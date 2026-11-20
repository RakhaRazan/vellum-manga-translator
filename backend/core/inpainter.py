"""LaMa ONNX Inpainter for erasing comic text masks and reconstructing clean backgrounds.

Uses local weights at `models/lama_fp32.onnx` (via `ModelManager`).
Implements SPEC.md Section 6 VRAM & Memory Management:
- Lazy ONNX Runtime session loading with automatic CUDA -> CPU OOM fallback.
- Tile-based ROI inpainting (crops around text bounding boxes with configurable padding,
  processed sequentially at `512x512` one tile at a time) to prevent peak memory/VRAM spikes
  while keeping all unmasked page pixels 100% lossless.
"""

import base64
from collections.abc import Sequence
import gc
import io
from pathlib import Path
import sys
from typing import Protocol, TypedDict, cast

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.detector import (  # noqa: E402
    ImageInput,
    NDArrayF32,
    NDArrayU8,
    load_rgb_image,
)
from backend.core.model_manager import (  # noqa: E402
    ModelManager,
    create_onnx_session,
    get_execution_provider,
)


class _OrtSession(Protocol):
    """Protocol for an ONNX Runtime InferenceSession running LaMa."""

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: dict[str, NDArrayF32],
    ) -> Sequence[NDArrayF32]: ...


class InpaintResult(TypedDict):
    """Structured result returned by `LamaInpainter.inpaint_with_metadata()`."""

    image: NDArrayU8
    tiles_processed: int
    execution_provider: str


def load_mask_image(mask: ImageInput, target_hw: tuple[int, int] | None = None) -> NDArrayU8:
    """Load a binary mask from path, bytes, PIL Image, or ndarray into a 2D `(H, W)` uint8 array (`0` or `255`)."""
    if isinstance(mask, (str, Path)):
        path = Path(mask).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Mask file not found: {path}")
        with Image.open(path) as pil_img:
            gray = np.asarray(pil_img.convert("L"), dtype=np.uint8)
    elif isinstance(mask, bytes):
        with Image.open(io.BytesIO(mask)) as pil_img:
            gray = np.asarray(pil_img.convert("L"), dtype=np.uint8)
    elif isinstance(mask, Image.Image):
        gray = np.asarray(mask.convert("L"), dtype=np.uint8)
    elif mask.ndim == 2:
        gray = np.asarray(mask, dtype=np.uint8)
    elif mask.ndim == 3:
        gray = np.asarray(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported mask array shape: {mask.shape!r}")

    if target_hw is not None and (gray.shape[0] != target_hw[0] or gray.shape[1] != target_hw[1]):
        gray = np.asarray(
            cv2.resize(
                gray,
                (target_hw[1], target_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            ),
            dtype=np.uint8,
        )

    return np.asarray(np.where(gray > 127, 255, 0), dtype=np.uint8)


def _extract_bboxes_from_mask(mask: NDArrayU8) -> list[list[int]]:
    """Derive bounding boxes `[x1, y1, x2, y2]` from connected active regions in a binary mask."""
    if not np.any(mask > 0):
        return []

    # Dilate to group nearby characters/lines within the same bubble into a single ROI
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    grouped: NDArrayU8 = np.asarray(
        cv2.dilate(mask, kernel, iterations=1),
        dtype=np.uint8,
    )
    num_labels, _labels, stats_raw, _centroids = cv2.connectedComponentsWithStats(
        grouped,
        connectivity=8,
    )
    if num_labels <= 1:
        return []

    stats = np.asarray(stats_raw, dtype=np.int32)
    stats_rows = cast(list[list[int]], stats.tolist())
    h, w = mask.shape[0], mask.shape[1]

    bboxes: list[list[int]] = []
    for row in stats_rows[1:]:
        bx = max(0, row[0])
        by = max(0, row[1])
        bw = row[2]
        bh = row[3]
        area = row[4]
        if area < 16 or bw < 4 or bh < 4:
            continue
        bboxes.append([bx, by, min(w, bx + bw), min(h, by + bh)])
    return bboxes


def _refine_bubble_mask(
    rgb_image: NDArrayU8,
    base_mask: NDArrayU8,
    bboxes: Sequence[Sequence[int]],
    dilate_kernel_size: int = 7,
) -> NDArrayU8:
    """Ensure all interior text strokes and punctuation inside each detected `bbox` are covered by the mask.

    Preserves speech bubble outlines that intersect the outer perimeter of `bbox` while capturing
    any small punctuation dots (`…`, `!`, `?`) and JPEG ringing halos around characters.
    """
    refined: NDArrayU8 = np.asarray(base_mask.copy(), dtype=np.uint8)
    orig_h, orig_w = rgb_image.shape[0], rgb_image.shape[1]

    for bbox in bboxes:
        if len(bbox) < 4:
            continue
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(orig_w, int(bbox[2]))
        y2 = min(orig_h, int(bbox[3]))
        bh = y2 - y1
        bw = x2 - x1
        if bh < 10 or bw < 10:
            continue

        crop_rgb: NDArrayU8 = np.asarray(rgb_image[y1:y2, x1:x2], dtype=np.uint8)
        gray: NDArrayU8 = np.asarray(
            cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY),
            dtype=np.uint8,
        )
        ink_mask: NDArrayU8 = np.asarray(gray < 210, dtype=np.uint8)
        num_labels, labels_raw = cv2.connectedComponents(ink_mask, connectivity=8)

        if num_labels > 1:
            labels = np.asarray(labels_raw, dtype=np.int32)
            top_row = cast(list[int], np.asarray(labels[0, :], dtype=np.int32).tolist())
            bottom_row = cast(
                list[int],
                np.asarray(labels[bh - 1, :], dtype=np.int32).tolist(),
            )
            left_col = cast(list[int], np.asarray(labels[:, 0], dtype=np.int32).tolist())
            right_col = cast(
                list[int],
                np.asarray(labels[:, bw - 1], dtype=np.int32).tolist(),
            )
            border_labels: set[int] = {
                val
                for val in (*top_row, *bottom_row, *left_col, *right_col)
                if val != 0
            }
            interior_ink = (labels > 0) & (~np.isin(labels, list(border_labels)))
            crop_mask: NDArrayU8 = np.asarray(refined[y1:y2, x1:x2], dtype=np.uint8)
            crop_mask[interior_ink] = 255
            refined[y1:y2, x1:x2] = crop_mask

    if dilate_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_kernel_size, dilate_kernel_size),
        )
        refined = np.asarray(
            cv2.dilate(refined, kernel, iterations=1),
            dtype=np.uint8,
        )

    return refined


class LamaInpainter:
    """Memory-efficient tile-based LaMa ONNX inpainter (`models/lama_fp32.onnx`)."""

    TILE_SIZE: int = 512

    model_manager: ModelManager
    model_path: Path
    _custom_model_path: Path | None
    tile_padding: int
    dilate_kernel_size: int
    force_cpu: bool
    _session: _OrtSession | None
    _active_provider: str | None
    _last_provider: str | None

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        model_path: Path | str | None = None,
        tile_padding: int = 32,
        dilate_kernel_size: int = 9,
        force_cpu: bool = False,
    ) -> None:
        self.model_manager = model_manager or ModelManager()
        self._custom_model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else None
        )
        self.model_path = (
            self._custom_model_path
            if self._custom_model_path is not None
            else self.model_manager.get_lama_path()
        )
        self.tile_padding = tile_padding
        self.dilate_kernel_size = dilate_kernel_size
        self.force_cpu = force_cpu
        self._session = None
        self._active_provider = None
        self._last_provider = None

    def _get_session(self, fallback_cpu: bool = False) -> _OrtSession:
        """Lazily initialize the ONNX Runtime session with CUDA / DirectML -> CPU fallback."""
        if self._session is not None and not fallback_cpu:
            return self._session

        if fallback_cpu and self._session is not None:
            self.unload()

        self.model_path = (
            self._custom_model_path
            if self._custom_model_path is not None
            else self.model_manager.get_lama_path()
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"LaMa Inpainting ONNX model not found at {self.model_path}. "
                + "Run ModelManager.download_missing_models() first."
            )

        raw_session, active_provider = create_onnx_session(
            self.model_path,
            force_cpu=(self.force_cpu or fallback_cpu),
        )
        session = cast(_OrtSession, cast(object, raw_session))
        self._session = session
        self._active_provider = active_provider
        self._last_provider = active_provider
        return session

    @property
    def is_loaded(self) -> bool:
        """Return True if the LaMa ONNX session is currently resident in memory."""
        return self._session is not None

    @property
    def execution_provider(self) -> str:
        """Return the active or most recently used ONNX Runtime execution provider."""
        if self._active_provider is not None:
            return self._active_provider
        if self._last_provider is not None:
            return self._last_provider
        return get_execution_provider(force_cpu=self.force_cpu)

    def unload(self) -> None:
        """Release the ONNX session from memory/VRAM and run garbage collection."""
        self._session = None
        self._active_provider = None
        _ = gc.collect()

    def _run_tile_512(
        self,
        tile_rgb: NDArrayU8,
        tile_mask: NDArrayU8,
    ) -> NDArrayU8:
        """Run `lama_fp32.onnx` on a single `512x512` RGB tile and binary mask."""
        img_tensor: NDArrayF32 = (
            np.asarray(
                tile_rgb.transpose(2, 0, 1)[np.newaxis, ...],
                dtype=np.float32,
            )
            / np.float32(255.0)
        )
        mask_tensor: NDArrayF32 = np.asarray(
            np.where(tile_mask > 127, 1.0, 0.0)[np.newaxis, np.newaxis, ...],
            dtype=np.float32,
        )

        session = self._get_session()
        try:
            outputs = session.run(
                None,
                {"image": img_tensor, "mask": mask_tensor},
            )
        except Exception:
            if self._active_provider == "CPUExecutionProvider":
                raise
            # Automatic safe CPU fallback if CUDA/DirectML fails or OOMs during inference
            session = self._get_session(fallback_cpu=True)
            outputs = session.run(
                None,
                {"image": img_tensor, "mask": mask_tensor},
            )

        out_nchw: NDArrayF32 = np.asarray(outputs[0], dtype=np.float32)
        out_chw: NDArrayF32 = np.reshape(
            out_nchw,
            (3, self.TILE_SIZE, self.TILE_SIZE),
        )
        out_hwc: NDArrayF32 = np.asarray(out_chw.transpose(1, 2, 0), dtype=np.float32)
        clipped: NDArrayU8 = np.asarray(
            np.clip(np.rint(out_hwc), 0.0, 255.0),
            dtype=np.uint8,
        )
        return clipped

    def _inpaint_roi(
        self,
        canvas: NDArrayU8,
        full_mask: NDArrayU8,
        bbox: Sequence[int],
        padding: int,
    ) -> bool:
        """Inpaint a single padded ROI `[x1, y1, x2, y2]` and patch masked pixels onto `canvas`."""
        orig_h, orig_w = canvas.shape[0], canvas.shape[1]
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(orig_w, int(bbox[2]) + padding)
        y2 = min(orig_h, int(bbox[3]) + padding)

        roi_w = x2 - x1
        roi_h = y2 - y1
        if roi_w <= 0 or roi_h <= 0:
            return False

        roi_mask_check: NDArrayU8 = np.asarray(full_mask[y1:y2, x1:x2], dtype=np.uint8)
        if not np.any(roi_mask_check > 0):
            return False

        ts = self.TILE_SIZE

        # Case 1: The padded ROI fits inside a native 512x512 window (1:1 pixel resolution, zero resize blur)
        if roi_w <= ts and roi_h <= ts and orig_w >= ts and orig_h >= ts:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            wx1 = max(0, min(orig_w - ts, cx - ts // 2))
            wy1 = max(0, min(orig_h - ts, cy - ts // 2))
            wx2 = wx1 + ts
            wy2 = wy1 + ts

            tile_rgb: NDArrayU8 = np.asarray(
                canvas[wy1:wy2, wx1:wx2].copy(),
                dtype=np.uint8,
            )
            # Restrict the tile mask to the current padded ROI so neighboring bubbles are handled in their own tiles
            tile_mask: NDArrayU8 = np.zeros((ts, ts), dtype=np.uint8)
            rel_x1 = x1 - wx1
            rel_y1 = y1 - wy1
            rel_x2 = x2 - wx1
            rel_y2 = y2 - wy1
            tile_mask[rel_y1:rel_y2, rel_x1:rel_x2] = roi_mask_check

            inpainted_tile = self._run_tile_512(tile_rgb, tile_mask)

            # Composite only the masked pixels back onto the canvas for 100% lossless backgrounds
            active = tile_mask > 0
            target_view: NDArrayU8 = np.asarray(canvas[wy1:wy2, wx1:wx2], dtype=np.uint8)
            target_view[active] = inpainted_tile[active]
            canvas[wy1:wy2, wx1:wx2] = target_view
            return True

        # Case 2: Padded ROI exceeds 512px (or image is smaller than 512px) -> resize ROI crop to 512x512
        roi_rgb: NDArrayU8 = np.asarray(canvas[y1:y2, x1:x2].copy(), dtype=np.uint8)
        resized_rgb: NDArrayU8 = np.asarray(
            cv2.resize(roi_rgb, (ts, ts), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )
        resized_mask: NDArrayU8 = np.asarray(
            cv2.resize(roi_mask_check, (ts, ts), interpolation=cv2.INTER_NEAREST),
            dtype=np.uint8,
        )

        inpainted_512 = self._run_tile_512(resized_rgb, resized_mask)
        restored_roi: NDArrayU8 = np.asarray(
            cv2.resize(inpainted_512, (roi_w, roi_h), interpolation=cv2.INTER_CUBIC),
            dtype=np.uint8,
        )

        active_roi = roi_mask_check > 0
        roi_rgb[active_roi] = restored_roi[active_roi]
        canvas[y1:y2, x1:x2] = roi_rgb
        return True

    def inpaint_with_metadata(
        self,
        image: ImageInput,
        mask: ImageInput,
        bboxes: Sequence[Sequence[int]] | None = None,
        tile_padding: int | None = None,
        refine_mask: bool = True,
        auto_unload: bool = True,
    ) -> InpaintResult:
        """Erase text regions indicated by `mask` (and optional `bboxes`) using tile-based LaMa inpainting."""
        rgb = load_rgb_image(image)
        orig_h, orig_w = rgb.shape[0], rgb.shape[1]
        bin_mask = load_mask_image(mask, target_hw=(orig_h, orig_w))

        roi_boxes: list[list[int]] = (
            [list(b) for b in bboxes]
            if bboxes is not None
            else _extract_bboxes_from_mask(bin_mask)
        )

        working_mask = (
            _refine_bubble_mask(
                rgb_image=rgb,
                base_mask=bin_mask,
                bboxes=roi_boxes,
                dilate_kernel_size=self.dilate_kernel_size,
            )
            if refine_mask and roi_boxes
            else bin_mask
        )

        canvas: NDArrayU8 = np.asarray(rgb.copy(), dtype=np.uint8)
        pad = self.tile_padding if tile_padding is None else tile_padding

        tiles_count = 0
        provider_used = get_execution_provider(force_cpu=self.force_cpu)
        try:
            for box in roi_boxes:
                if self._inpaint_roi(
                    canvas=canvas,
                    full_mask=working_mask,
                    bbox=box,
                    padding=pad,
                ):
                    tiles_count += 1
            if self._active_provider is not None:
                provider_used = self._active_provider
        finally:
            if auto_unload:
                self.unload()

        return {
            "image": canvas,
            "tiles_processed": tiles_count,
            "execution_provider": provider_used,
        }

    def inpaint(
        self,
        image: ImageInput,
        mask: ImageInput,
        bboxes: Sequence[Sequence[int]] | None = None,
        tile_padding: int | None = None,
    ) -> NDArrayU8:
        """Erase masked text regions and return the cleaned RGB image array `(H, W, 3)` uint8."""
        res = self.inpaint_with_metadata(
            image=image,
            mask=mask,
            bboxes=bboxes,
            tile_padding=tile_padding,
        )
        return res["image"]

    @staticmethod
    def image_to_png_bytes(rgb_image: NDArrayU8) -> bytes:
        """Encode an RGB `(H, W, 3)` uint8 array into lossless PNG bytes."""
        pil_img = Image.fromarray(rgb_image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    @classmethod
    def image_to_base64(cls, rgb_image: NDArrayU8) -> str:
        """Encode an RGB `(H, W, 3)` uint8 array as a `data:image/png;base64,...` data URI (SPEC.md 5.3)."""
        png_bytes = cls.image_to_png_bytes(rgb_image)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"
