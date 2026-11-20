"""Comic Text Detector (ONNX) for detecting speech bubble text bounding boxes and binary text masks.

Uses local weights at `models/comictextdetector.pt.onnx` (via `ModelManager`).
Outputs:
1. Sorted text bounding boxes (`bbox`: `[x1, y1, x2, y2]`) and 4-point polygons (`polygon`)
   following comic reading order (`ja`: Right-to-Left / Top-to-Bottom; `ko`/`zh`: Left-to-Right / Top-to-Bottom).
2. Binary pixel-level text segmentation mask (`mask`: `np.ndarray` of shape `(H, W)`, `uint8`, values `0` or `255`)
   ready for downstream LaMa inpainting.
"""

import base64
from collections.abc import Sequence
import gc
import io
from pathlib import Path
import sys
from typing import Literal, Protocol, TypedDict, cast

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.model_manager import (  # noqa: E402
    ModelManager,
    create_onnx_session,
    get_execution_provider,
)

NDArrayU8 = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
NDArrayF32 = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
TextDirection = Literal["horizontal", "vertical"]
ImageInput = str | Path | bytes | Image.Image | NDArrayU8
BoxXYXY = tuple[float, float, float, float]


class _OrtInputNode(Protocol):
    """Protocol for an ONNX Runtime input node metadata object."""

    @property
    def name(self) -> str: ...


class _OrtSession(Protocol):
    """Protocol for an ONNX Runtime InferenceSession."""

    def get_inputs(self) -> Sequence[_OrtInputNode]: ...

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: dict[str, NDArrayF32],
    ) -> Sequence[NDArrayF32]: ...


class DetectedBubble(TypedDict):
    """Detected text block matching SPEC.md Section 5.1 Bubble Definition."""

    id: str
    order: int
    bbox: list[int]
    polygon: list[list[int]]
    confidence: float
    text_direction: TextDirection


class ImageDimensions(TypedDict):
    """Image width and height in pixels."""

    width: int
    height: int


class DetectionResult(TypedDict):
    """Complete output from ComicTextDetector.detect()."""

    dimensions: ImageDimensions
    execution_provider: str
    bubbles: list[DetectedBubble]
    mask: NDArrayU8


def load_rgb_image(image: ImageInput) -> NDArrayU8:
    """Convert a file path, raw bytes, PIL Image, or numpy array into an RGB uint8 array `(H, W, 3)`."""
    if isinstance(image, (str, Path)):
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as pil_img:
            return np.asarray(pil_img.convert("RGB"), dtype=np.uint8)

    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as pil_img:
            return np.asarray(pil_img.convert("RGB"), dtype=np.uint8)

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    if image.ndim == 2:
        return np.asarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        return np.asarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB), dtype=np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.asarray(image, dtype=np.uint8)

    raise ValueError(f"Unsupported ndarray image shape: {image.shape!r}")


def _nms_xyxy(
    boxes: Sequence[BoxXYXY],
    scores: Sequence[float],
    iou_threshold: float,
) -> list[int]:
    """Perform greedy Non-Maximum Suppression on `(x1, y1, x2, y2)` boxes."""
    if not boxes:
        return []

    areas: list[float] = [
        max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        for b in boxes
    ]
    order: list[int] = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )

    keep: list[int] = []
    while order:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        bx1, by1, bx2, by2 = boxes[i]
        area_i = areas[i]
        next_order: list[int] = []

        for j in order[1:]:
            ox1, oy1, ox2, oy2 = boxes[j]
            xx1 = max(bx1, ox1)
            yy1 = max(by1, oy1)
            xx2 = min(bx2, ox2)
            yy2 = min(by2, oy2)

            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            area_j = areas[j]
            union = area_i + area_j - inter + 1e-6
            iou = inter / union

            # Also suppress boxes that are largely contained inside a higher-scoring box
            min_area = min(area_i, area_j) + 1e-6
            io_min = inter / min_area

            if iou <= iou_threshold and io_min <= 0.75:
                next_order.append(j)

        order = next_order

    return keep


def sort_bubbles_reading_order(
    raw_boxes: list[tuple[list[int], float, TextDirection]],
    source_language: str = "ja",
    page_height: int = 1400,
) -> list[DetectedBubble]:
    """Sort detected bounding boxes in comic reading order and assign sequential IDs.

    - `ja` (Manga): Right-to-Left within each horizontal row band, Top-to-Bottom across rows.
    - `ko` / `zh` (Manhwa / Manhua): Left-to-Right within each horizontal row band, Top-to-Bottom across rows.
    """
    if not raw_boxes:
        return []

    rtl = source_language.lower().strip() == "ja"

    # Estimate row grouping tolerance from median box height
    heights = sorted(max(1, b[0][3] - b[0][1]) for b in raw_boxes)
    median_h = float(heights[len(heights) // 2]) if heights else 60.0
    row_tolerance = max(median_h * 0.8, page_height * 0.05)

    # First sort by vertical center (cy) top-to-bottom
    indexed = sorted(
        raw_boxes,
        key=lambda item: (item[0][1] + item[0][3]) / 2.0,
    )

    rows: list[list[tuple[list[int], float, TextDirection]]] = []
    for item in indexed:
        cy = (item[0][1] + item[0][3]) / 2.0
        if not rows:
            rows.append([item])
            continue

        last_row = rows[-1]
        current_row_cy = sum((r[0][1] + r[0][3]) / 2.0 for r in last_row) / len(last_row)
        if abs(cy - current_row_cy) <= row_tolerance:
            last_row.append(item)
        else:
            rows.append([item])

    sorted_bubbles: list[DetectedBubble] = []
    order_idx = 1
    for row in rows:
        row_sorted = sorted(
            row,
            key=lambda item: (item[0][0] + item[0][2]) / 2.0,
            reverse=rtl,
        )
        for bbox, conf, direction in row_sorted:
            x1, y1, x2, y2 = bbox
            polygon = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]
            sorted_bubbles.append(
                {
                    "id": f"bubble_{order_idx:03d}",
                    "order": order_idx,
                    "bbox": [x1, y1, x2, y2],
                    "polygon": polygon,
                    "confidence": round(float(conf), 4),
                    "text_direction": direction,
                }
            )
            order_idx += 1

    return sorted_bubbles


class ComicTextDetector:
    """ONNX Comic Text Detector wrapper with lazy session loading, GPU/DirectML -> CPU fallback, and clean unload."""

    INPUT_SIZE: int = 1024

    model_manager: ModelManager
    model_path: Path
    _custom_model_path: Path | None
    conf_threshold: float
    nms_threshold: float
    mask_threshold: float
    box_padding: int
    dilate_kernel_size: int
    force_cpu: bool
    _session: _OrtSession | None
    _active_provider: str | None
    _last_provider: str | None

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        model_path: Path | str | None = None,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.40,
        mask_threshold: float = 0.35,
        box_padding: int = 6,
        dilate_kernel_size: int = 5,
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
            else self.model_manager.get_detector_path()
        )
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.mask_threshold = mask_threshold
        self.box_padding = box_padding
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
            else self.model_manager.get_detector_path()
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Comic Text Detector ONNX model not found at {self.model_path}. "
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
        """Return True if the Comic Text Detector ONNX session is currently resident in memory."""
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
        """Release the ONNX Runtime session from RAM/VRAM and run garbage collection."""
        self._session = None
        self._active_provider = None
        _ = gc.collect()

    def _preprocess(
        self,
        rgb_image: NDArrayU8,
    ) -> tuple[NDArrayF32, float, int, int]:
        """Letterbox-pad an RGB image to `(1, 3, 1024, 1024)` float32 in `[0.0, 1.0]`."""
        orig_h, orig_w = rgb_image.shape[0], rgb_image.shape[1]
        scale = min(self.INPUT_SIZE / orig_h, self.INPUT_SIZE / orig_w)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        resized = cv2.resize(
            rgb_image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        tensor = np.asarray(
            canvas.transpose(2, 0, 1)[np.newaxis, ...],
            dtype=np.float32,
        ) / np.float32(255.0)
        return tensor, scale, new_w, new_h

    def _decode_blocks(
        self,
        blk_output: NDArrayF32,
        scale: float,
        orig_w: int,
        orig_h: int,
        conf_threshold: float,
        nms_threshold: float,
        box_padding: int,
    ) -> list[tuple[list[int], float, TextDirection]]:
        """Decode YOLOv5 text block predictions (`blk`: `[1, 64512, 7]`) into original image coordinates."""
        # Fast pre-filter in NumPy on objectness confidence column (index 4) before Python unpacking
        raw_2d: NDArrayF32 = np.reshape(blk_output, (-1, 7))
        filtered_arr: NDArrayF32 = np.asarray(
            raw_2d[raw_2d[:, 4] >= conf_threshold],
            dtype=np.float32,
        )
        rows = cast(list[list[float]], filtered_arr.tolist())
        if not rows:
            return []

        boxes_xyxy: list[BoxXYXY] = []
        scores: list[float] = []
        cls_ids: list[int] = []

        for row in rows:
            obj_conf = row[4]
            cls0 = row[5]
            cls1 = row[6]
            cls_id = 0 if cls0 >= cls1 else 1
            cls_max = cls0 if cls0 >= cls1 else cls1
            score = obj_conf * cls_max
            if score < conf_threshold:
                continue

            cx = row[0] / scale
            cy = row[1] / scale
            bw = row[2] / scale
            bh = row[3] / scale

            boxes_xyxy.append(
                (
                    cx - bw / 2.0,
                    cy - bh / 2.0,
                    cx + bw / 2.0,
                    cy + bh / 2.0,
                )
            )
            scores.append(score)
            cls_ids.append(cls_id)

        if not boxes_xyxy:
            return []

        keep_indices = _nms_xyxy(
            boxes_xyxy,
            scores,
            iou_threshold=nms_threshold,
        )

        decoded: list[tuple[list[int], float, TextDirection]] = []
        for idx in keep_indices:
            x1_f, y1_f, x2_f, y2_f = boxes_xyxy[idx]
            bx1 = max(0, int(round(x1_f)) - box_padding)
            by1 = max(0, int(round(y1_f)) - box_padding)
            bx2 = min(orig_w, int(round(x2_f)) + box_padding)
            by2 = min(orig_h, int(round(y2_f)) + box_padding)

            if (bx2 - bx1) < 8 or (by2 - by1) < 8:
                continue

            # cls 0 is vertical comic text, cls 1 is horizontal comic text
            is_vertical = cls_ids[idx] == 0 and (by2 - by1) > (bx2 - bx1) * 1.2
            direction: TextDirection = "vertical" if is_vertical else "horizontal"
            decoded.append(([bx1, by1, bx2, by2], scores[idx], direction))

        return decoded

    def _build_binary_mask(
        self,
        seg_output: NDArrayF32,
        det_output: NDArrayF32,
        new_w: int,
        new_h: int,
        orig_w: int,
        orig_h: int,
        bubbles: list[DetectedBubble],
        mask_threshold: float,
        dilate_kernel_size: int,
        restrict_to_bboxes: bool = True,
    ) -> NDArrayU8:
        """Decode `seg` (`[1, 1, 1024, 1024]`) and `det` (`[1, 2, 1024, 1024]`) into a binary `(H, W)` uint8 mask."""
        seg_2d: NDArrayF32 = np.reshape(seg_output, (self.INPUT_SIZE, self.INPUT_SIZE))
        det_3d: NDArrayF32 = np.reshape(det_output, (2, self.INPUT_SIZE, self.INPUT_SIZE))
        seg_crop: NDArrayF32 = np.asarray(seg_2d[:new_h, :new_w], dtype=np.float32)
        det_prob_crop: NDArrayF32 = np.asarray(
            det_3d[0, :new_h, :new_w],
            dtype=np.float32,
        )

        # Fuse pixel segmentation map with DBNet text probability map
        fused_crop: NDArrayF32 = np.asarray(
            np.maximum(seg_crop, det_prob_crop * np.float32(0.85)),
            dtype=np.float32,
        )
        resized_prob: NDArrayF32 = np.asarray(
            cv2.resize(
                fused_crop,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            ),
            dtype=np.float32,
        )

        binary_mask: NDArrayU8 = np.asarray(
            np.where(resized_prob >= mask_threshold, 255, 0),
            dtype=np.uint8,
        )

        if dilate_kernel_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_kernel_size, dilate_kernel_size),
            )
            binary_mask = np.asarray(
                cv2.dilate(binary_mask, kernel, iterations=1),
                dtype=np.uint8,
            )

        if restrict_to_bboxes and bubbles:
            roi_mask: NDArrayU8 = np.zeros_like(binary_mask)
            pad = max(8, self.box_padding)
            for b in bubbles:
                x1, y1, x2, y2 = b["bbox"]
                rx1 = max(0, x1 - pad)
                ry1 = max(0, y1 - pad)
                rx2 = min(orig_w, x2 + pad)
                ry2 = min(orig_h, y2 + pad)
                roi_mask[ry1:ry2, rx1:rx2] = 255
            binary_mask = np.asarray(
                cv2.bitwise_and(binary_mask, roi_mask),
                dtype=np.uint8,
            )

        return binary_mask

    def detect(
        self,
        image: ImageInput,
        source_language: str = "ja",
        conf_threshold: float | None = None,
        nms_threshold: float | None = None,
        mask_threshold: float | None = None,
        box_padding: int | None = None,
        dilate_kernel_size: int | None = None,
        restrict_mask_to_bboxes: bool = True,
        auto_unload: bool = True,
    ) -> DetectionResult:
        """Run Comic Text Detector on `image` and return sorted bounding boxes and binary text mask."""
        rgb = load_rgb_image(image)
        orig_h, orig_w = rgb.shape[0], rgb.shape[1]

        conf_t = self.conf_threshold if conf_threshold is None else conf_threshold
        nms_t = self.nms_threshold if nms_threshold is None else nms_threshold
        mask_t = self.mask_threshold if mask_threshold is None else mask_threshold
        pad_px = self.box_padding if box_padding is None else box_padding
        dilate_k = (
            self.dilate_kernel_size
            if dilate_kernel_size is None
            else dilate_kernel_size
        )

        input_tensor, scale, new_w, new_h = self._preprocess(rgb)

        try:
            session = self._get_session()
            input_name = session.get_inputs()[0].name
            try:
                raw_outputs = session.run(
                    None,
                    {input_name: input_tensor},
                )
            except Exception:
                if self._active_provider == "CPUExecutionProvider":
                    raise
                # Automatic safe CPU fallback if CUDA/DirectML fails during inference
                session = self._get_session(fallback_cpu=True)
                input_name = session.get_inputs()[0].name
                raw_outputs = session.run(
                    None,
                    {input_name: input_tensor},
                )

            provider_used = self._active_provider or "CPUExecutionProvider"
            blk_output = np.asarray(raw_outputs[0], dtype=np.float32)
            seg_output = np.asarray(raw_outputs[1], dtype=np.float32)
            det_output = np.asarray(raw_outputs[2], dtype=np.float32)
        finally:
            if auto_unload:
                self.unload()

        raw_boxes = self._decode_blocks(
            blk_output=blk_output,
            scale=scale,
            orig_w=orig_w,
            orig_h=orig_h,
            conf_threshold=conf_t,
            nms_threshold=nms_t,
            box_padding=pad_px,
        )

        bubbles = sort_bubbles_reading_order(
            raw_boxes=raw_boxes,
            source_language=source_language,
            page_height=orig_h,
        )

        binary_mask = self._build_binary_mask(
            seg_output=seg_output,
            det_output=det_output,
            new_w=new_w,
            new_h=new_h,
            orig_w=orig_w,
            orig_h=orig_h,
            bubbles=bubbles,
            mask_threshold=mask_t,
            dilate_kernel_size=dilate_k,
            restrict_to_bboxes=restrict_mask_to_bboxes,
        )

        return {
            "dimensions": {"width": orig_w, "height": orig_h},
            "execution_provider": provider_used,
            "bubbles": bubbles,
            "mask": binary_mask,
        }

    def detect_bboxes_and_mask(
        self,
        image: ImageInput,
        source_language: str = "ja",
    ) -> tuple[list[list[int]], NDArrayU8]:
        """Convenience method returning `(bboxes, binary_mask)`."""
        result = self.detect(image=image, source_language=source_language)
        bboxes = [b["bbox"] for b in result["bubbles"]]
        return bboxes, result["mask"]

    @staticmethod
    def mask_to_png_bytes(mask: NDArrayU8) -> bytes:
        """Encode a binary `(H, W)` uint8 mask into PNG bytes."""
        pil_mask = Image.fromarray(mask)
        buf = io.BytesIO()
        pil_mask.save(buf, format="PNG")
        return buf.getvalue()

    @classmethod
    def mask_to_base64(cls, mask: NDArrayU8) -> str:
        """Encode a binary `(H, W)` uint8 mask as a `data:image/png;base64,...` data URI."""
        png_bytes = cls.mask_to_png_bytes(mask)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"


def _verify_samples() -> int:
    """CLI verification runner testing ComicTextDetector on `sample_ja.jpg`, `sample_ko.jpg`, and `sample_zh.jpg`."""
    workspace_root = Path(__file__).resolve().parents[2]
    samples_dir = workspace_root / "samples"
    detector = ComicTextDetector()

    sample_cases = [
        ("ja", samples_dir / "sample_ja.jpg"),
        ("ko", samples_dir / "sample_ko.jpg"),
        ("zh", samples_dir / "sample_zh.jpg"),
    ]

    print(f"=== ComicTextDetector Verification (model: {detector.model_path}) ===")
    all_passed = True

    for lang, sample_path in sample_cases:
        if not sample_path.is_file():
            print(f"[ERROR] Missing sample image: {sample_path}", file=sys.stderr)
            all_passed = False
            continue

        result = detector.detect(sample_path, source_language=lang)
        bubbles = result["bubbles"]
        mask = result["mask"]
        dims = result["dimensions"]
        nonzero_pixels = int(np.count_nonzero(mask))
        total_pixels = dims["width"] * dims["height"]
        coverage_pct = (nonzero_pixels / max(1, total_pixels)) * 100.0

        print(
            f"\n[{lang.upper()}] {sample_path.name} "
            + f"({dims['width']}x{dims['height']}, provider={result['execution_provider']})"
        )
        print(
            f"  Detected bubbles: {len(bubbles)} | "
            + f"Mask shape: {mask.shape}, dtype: {mask.dtype}, "
            + f"unique values: {np.unique(mask).tolist()}, "
            + f"active text pixels: {nonzero_pixels} ({coverage_pct:.2f}%)"
        )

        if len(bubbles) < 4 or nonzero_pixels == 0:
            all_passed = False

        for b in bubbles:
            x1, y1, x2, y2 = b["bbox"]
            box_mask: NDArrayU8 = np.asarray(mask[y1:y2, x1:x2], dtype=np.uint8)
            box_nonzero = int(np.count_nonzero(box_mask))
            print(
                f"    - {b['id']} (order={b['order']}): bbox={b['bbox']}, "
                + f"conf={b['confidence']:.3f}, dir={b['text_direction']}, "
                + f"mask_px_in_box={box_nonzero}"
            )
            if box_nonzero == 0:
                all_passed = False

    if not all_passed:
        print("\nERROR: Detector verification failed on one or more sample pages.", file=sys.stderr)
        return 1

    print("\nSUCCESS: All 3 synthetic comic pages detected bounding boxes and binary text masks accurately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_verify_samples())
