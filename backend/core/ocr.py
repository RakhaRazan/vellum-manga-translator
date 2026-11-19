"""Multi-Language OCR Router for Japanese (`ja`), Korean (`ko`), and Chinese (`zh`) comics.

Routes text recognition per language using local offline weights in `models/`:
- `ja`: `manga-ocr` (`kha-white/manga-ocr-base` at `models/manga-ocr-base/`) supporting vertical
  Japanese tategaki text, furigana suppression, horizontal yokogaki lines, and Kanji.
- `ko`: `rapidocr_onnxruntime` using Korean PP-OCRv4 recognition (`korean_PP-OCRv4_rec_infer.onnx`)
  and Multilingual PP-OCRv3 detection (`Multilingual_PP-OCRv3_det_infer.onnx`).
- `zh`: `rapidocr_onnxruntime` using Chinese PP-OCRv4 recognition (`ch_PP-OCRv4_rec_infer.onnx`)
  and PP-OCRv4 detection (`ch_PP-OCRv4_det_infer.onnx`).
"""

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys
from typing import Literal, Protocol, TypedDict, cast

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.detector import (  # noqa: E402
    ComicTextDetector,
    DetectedBubble,
    ImageInput,
    NDArrayU8,
    TextDirection,
    load_rgb_image,
)
from backend.core.model_manager import ModelManager  # noqa: E402

SourceLanguage = Literal["ja", "ko", "zh"]
SUPPORTED_LANGUAGES: tuple[SourceLanguage, ...] = ("ja", "ko", "zh")


class OcrBubbleResult(TypedDict):
    """Bubble metadata enriched with extracted `source_text` matching SPEC.md Section 5.1."""

    id: str
    order: int
    bbox: list[int]
    polygon: list[list[int]]
    source_text: str
    confidence: float
    text_direction: TextDirection


class _MangaOcrEngine(Protocol):
    """Callable protocol for a loaded `MangaOcr` instance."""

    def __call__(self, img: Image.Image) -> str: ...


class _MangaOcrFactory(Protocol):
    """Constructor protocol for `manga_ocr.MangaOcr`."""

    def __call__(
        self,
        pretrained_model_name_or_path: str = ...,
        force_cpu: bool = ...,
    ) -> _MangaOcrEngine: ...


class _RapidOcrEngine(Protocol):
    """Callable protocol for a loaded `RapidOCR` instance."""

    def __call__(
        self,
        img_content: NDArrayU8,
        use_det: bool | None = None,
        use_cls: bool | None = None,
        use_rec: bool | None = None,
    ) -> tuple[object, object]: ...


class _RapidOcrFactory(Protocol):
    """Constructor protocol for `rapidocr_onnxruntime.RapidOCR`."""

    def __call__(
        self,
        *,
        det_model_path: str,
        rec_model_path: str,
        cls_model_path: str,
        det_use_cuda: bool,
        cls_use_cuda: bool,
        rec_use_cuda: bool,
    ) -> _RapidOcrEngine: ...


def _image_hw(arr: NDArrayU8) -> tuple[int, int]:
    """Return `(height, width)` of an image array as concrete `int` values."""
    return int(arr.shape[0]), int(arr.shape[1])


def _to_float(val: object, default: float = 0.0) -> float:
    """Safely convert a runtime scalar value to `float` without using `Any`."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return default
    return default


def normalize_language(source_lang: str) -> SourceLanguage:
    """Validate and normalize a source language code (`'ja'`, `'ko'`, or `'zh'`)."""
    cleaned = source_lang.lower().strip()
    if cleaned == "ja":
        return "ja"
    if cleaned == "ko":
        return "ko"
    if cleaned == "zh":
        return "zh"
    raise ValueError(
        f"Unsupported source_lang {source_lang!r}. Must be one of {SUPPORTED_LANGUAGES}."
    )


def crop_bbox_with_padding(
    rgb_image: NDArrayU8,
    bbox: Sequence[int],
    context_pad: int = 2,
    border_pad: int = 12,
) -> NDArrayU8:
    """Crop `[x1, y1, x2, y2]` from an RGB image and add a clean white border for OCR stability."""
    if len(bbox) < 4:
        raise ValueError(f"Expected 4-element bbox [x1, y1, x2, y2], got {bbox!r}")

    orig_h, orig_w = _image_hw(rgb_image)
    x1 = max(0, int(bbox[0]) - context_pad)
    y1 = max(0, int(bbox[1]) - context_pad)
    x2 = min(orig_w, int(bbox[2]) + context_pad)
    y2 = min(orig_h, int(bbox[3]) + context_pad)

    if x2 <= x1 or y2 <= y1:
        return np.full((32, 32, 3), 255, dtype=np.uint8)

    crop: NDArrayU8 = np.asarray(rgb_image[y1:y2, x1:x2].copy(), dtype=np.uint8)
    crop = _erase_border_touching_ink(crop)
    if border_pad > 0:
        crop = np.asarray(
            cv2.copyMakeBorder(
                crop,
                border_pad,
                border_pad,
                border_pad,
                border_pad,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            ),
            dtype=np.uint8,
        )
    return crop


def _erase_border_touching_ink(
    rgb_crop: NDArrayU8,
) -> NDArrayU8:
    """Remove speech-bubble outline strokes that intersect the outer perimeter of `rgb_crop`."""
    h, w = _image_hw(rgb_crop)
    if h < 12 or w < 12:
        return rgb_crop

    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    ink_mask = np.asarray(gray < 210, dtype=np.uint8)
    num_labels, labels_raw = cv2.connectedComponents(ink_mask, connectivity=8)
    if num_labels <= 1:
        return rgb_crop

    labels: np.ndarray[tuple[int, ...], np.dtype[np.int32]] = np.asarray(
        labels_raw,
        dtype=np.int32,
    )
    top_row = cast(list[int], np.asarray(labels[0, :], dtype=np.int32).tolist())
    bottom_row = cast(
        list[int],
        np.asarray(labels[h - 1, :], dtype=np.int32).tolist(),
    )
    left_col = cast(list[int], np.asarray(labels[:, 0], dtype=np.int32).tolist())
    right_col = cast(
        list[int],
        np.asarray(labels[:, w - 1], dtype=np.int32).tolist(),
    )

    border_labels: set[int] = {
        val
        for val in (*top_row, *bottom_row, *left_col, *right_col)
        if val != 0
    }
    if not border_labels:
        return rgb_crop

    cleaned: NDArrayU8 = np.asarray(rgb_crop.copy(), dtype=np.uint8)
    remove_mask = np.isin(labels, list(border_labels))
    # Dilate slightly to also clear anti-aliased gray halo around the bubble border
    remove_u8: NDArrayU8 = np.asarray(
        cv2.dilate(
            np.asarray(remove_mask, dtype=np.uint8) * 255,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ),
        dtype=np.uint8,
    )
    cleaned[remove_u8 > 0] = (255, 255, 255)
    return cleaned


def extract_horizontal_line_strips(
    rgb_crop: NDArrayU8,
    min_line_height: int = 16,
) -> list[NDArrayU8]:
    """Segment a horizontal multi-line bubble crop into individual horizontal line strips.

    If the bubble contains vertical tategaki text (height >= width or continuous vertical ink)
    or a single line of text, returns an empty list so the caller uses the full bubble crop.
    """
    h, w = _image_hw(rgb_crop)
    if w < h * 1.1:
        # Vertical or square bubble crop (e.g. Japanese tategaki) -> keep whole crop intact
        return []

    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    # Focus on the central 80% horizontal region when computing row ink density
    # so curved speech bubble borders at the far left/right corners are ignored.
    margin_x = max(1, int(w * 0.10))
    central_ink = np.asarray(gray[:, margin_x : w - margin_x] < 180, dtype=np.uint8)
    row_counts = cast(list[int], central_ink.sum(axis=1).tolist())

    in_line = False
    start_y = 0
    raw_spans: list[tuple[int, int]] = []
    for y, count in enumerate(row_counts):
        if count >= 3 and not in_line:
            in_line = True
            start_y = y
        elif count < 3 and in_line:
            in_line = False
            if (y - start_y) >= min_line_height:
                raw_spans.append((start_y, y))
    if in_line and (h - start_y) >= min_line_height:
        raw_spans.append((start_y, h))

    if len(raw_spans) < 2:
        return []

    full_ink: NDArrayU8 = np.asarray(gray < 180, dtype=np.uint8)
    strips: list[NDArrayU8] = []
    for sy1, sy2 in raw_spans:
        y1 = max(0, sy1 - 5)
        y2 = min(h, sy2 + 5)
        sub_ink: NDArrayU8 = np.asarray(full_ink[y1:y2, :], dtype=np.uint8)
        col_sums = cast(list[int], sub_ink.sum(axis=0).tolist())
        active_cols = [col_idx for col_idx, val in enumerate(col_sums) if val > 0]
        if not active_cols:
            continue
        x1 = max(0, active_cols[0] - 8)
        x2 = min(w, active_cols[-1] + 8)
        if (x2 - x1) < 16:
            continue

        strip: NDArrayU8 = np.asarray(rgb_crop[y1:y2, x1:x2], dtype=np.uint8)
        padded_strip = np.asarray(
            cv2.copyMakeBorder(
                strip,
                10,
                10,
                16,
                16,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            ),
            dtype=np.uint8,
        )
        strips.append(padded_strip)

    return strips if len(strips) >= 2 else []


def _parse_rapid_rec_only(raw_obj: object) -> list[tuple[str, float]]:
    """Parse `RapidOCR(..., use_det=False, use_rec=True)` output into `(text, score)` pairs."""
    if not isinstance(raw_obj, (list, tuple)):
        return []
    seq = cast(Sequence[object], raw_obj)
    parsed: list[tuple[str, float]] = []
    for entry in seq:
        if not isinstance(entry, (list, tuple)):
            continue
        items = cast(Sequence[object], entry)
        if len(items) < 1:
            continue
        txt = str(items[0]).strip()
        score = _to_float(items[1], 1.0) if len(items) >= 2 else 1.0
        if txt:
            parsed.append((txt, score))
    return parsed


def _parse_rapid_det_rec(raw_obj: object) -> list[tuple[float, float, str]]:
    """Parse `RapidOCR(image)` full detection+recognition output into `(cy, cx, text)` tuples."""
    if not isinstance(raw_obj, (list, tuple)):
        return []
    seq = cast(Sequence[object], raw_obj)
    lines: list[tuple[float, float, str]] = []
    for entry in seq:
        if not isinstance(entry, (list, tuple)):
            continue
        entry_items = cast(Sequence[object], entry)
        if len(entry_items) < 2:
            continue
        box_obj = entry_items[0]
        text_val = str(entry_items[1]).strip()
        if not text_val:
            continue

        cy = 0.0
        cx = 0.0
        if isinstance(box_obj, (list, tuple)):
            pts = cast(Sequence[object], box_obj)
            xs: list[float] = []
            ys: list[float] = []
            for pt_obj in pts:
                if isinstance(pt_obj, (list, tuple)):
                    coords = cast(Sequence[object], pt_obj)
                    if len(coords) >= 2:
                        xs.append(_to_float(coords[0]))
                        ys.append(_to_float(coords[1]))
            if xs and ys:
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)

        lines.append((cy, cx, text_val))
    return lines


class OCRRouter:
    """Multi-language OCR router with lazy initialization for `ja`, `ko`, and `zh`."""

    model_manager: ModelManager
    force_cpu: bool
    _manga_ocr: _MangaOcrEngine | None
    _rapid_ocr_engines: dict[SourceLanguage, _RapidOcrEngine]

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        force_cpu: bool = False,
    ) -> None:
        self.model_manager = model_manager or ModelManager()
        self.force_cpu = force_cpu
        self._manga_ocr = None
        self._rapid_ocr_engines = {}

    def _get_manga_ocr(self) -> _MangaOcrEngine:
        """Lazily initialize `MangaOcr` from local weights in `models/manga-ocr-base/`."""
        if self._manga_ocr is not None:
            return self._manga_ocr

        manga_ocr_dir = self.model_manager.get_manga_ocr_dir()
        if not (manga_ocr_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Japanese manga-ocr weights not found at {manga_ocr_dir}. "
                + "Run ModelManager.download_missing_models() first."
            )

        manga_ocr_mod = importlib.import_module("manga_ocr")
        manga_ocr_cls = cast(
            _MangaOcrFactory,
            getattr(manga_ocr_mod, "MangaOcr"),
        )

        try:
            instance = manga_ocr_cls(
                pretrained_model_name_or_path=str(manga_ocr_dir),
                force_cpu=self.force_cpu,
            )
        except Exception:
            instance = manga_ocr_cls(
                pretrained_model_name_or_path=str(manga_ocr_dir),
                force_cpu=True,
            )
        self._manga_ocr = instance
        return instance

    def _get_rapid_ocr(self, lang: Literal["ko", "zh"]) -> _RapidOcrEngine:
        """Lazily initialize `RapidOCR` with local Korean or Chinese PP-OCRv4 ONNX weights."""
        if lang in self._rapid_ocr_engines:
            return self._rapid_ocr_engines[lang]

        paths = self.model_manager.get_rapidocr_paths(language=lang)
        required_paths: tuple[tuple[str, Path], ...] = (
            ("det_model_path", paths["det_model_path"]),
            ("rec_model_path", paths["rec_model_path"]),
            ("cls_model_path", paths["cls_model_path"]),
        )
        for key, path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(
                    f"RapidOCR ({lang}) model file {key} not found at {path}. "
                    + "Run ModelManager.download_missing_models() first."
                )

        rapidocr_mod = importlib.import_module("rapidocr_onnxruntime")
        rapidocr_cls = cast(
            _RapidOcrFactory,
            getattr(rapidocr_mod, "RapidOCR"),
        )

        use_cuda = (
            not self.force_cpu
            and self.model_manager.get_status()["execution_provider"]
            == "CUDAExecutionProvider"
        )
        engine = rapidocr_cls(
            det_model_path=str(paths["det_model_path"]),
            rec_model_path=str(paths["rec_model_path"]),
            cls_model_path=str(paths["cls_model_path"]),
            det_use_cuda=use_cuda,
            cls_use_cuda=use_cuda,
            rec_use_cuda=use_cuda,
        )
        self._rapid_ocr_engines[lang] = engine
        return engine

    def _recognize_ja(
        self,
        raw_crop: NDArrayU8,
        text_direction: TextDirection = "horizontal",
    ) -> str:
        """Run Japanese `manga-ocr` supporting vertical tategaki and multi-line horizontal text."""
        engine = self._get_manga_ocr()

        if text_direction == "horizontal":
            line_strips = extract_horizontal_line_strips(raw_crop)
            if line_strips:
                line_texts: list[str] = []
                for strip in line_strips:
                    pil_strip = Image.fromarray(strip)
                    t = engine(pil_strip).strip()
                    if t:
                        line_texts.append(t)
                if line_texts:
                    return " ".join(line_texts)

        padded = np.asarray(
            cv2.copyMakeBorder(
                raw_crop,
                8,
                8,
                8,
                8,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            ),
            dtype=np.uint8,
        )
        pil_crop = Image.fromarray(padded)
        return engine(pil_crop).strip()

    def _recognize_rapid(
        self,
        raw_crop: NDArrayU8,
        lang: Literal["ko", "zh"],
    ) -> str:
        """Run Korean or Chinese `RapidOCR` using PP-OCRv4 ONNX models."""
        engine = self._get_rapid_ocr(lang)

        # 1. If the bubble cleanly segments into horizontal lines, run PP-OCRv4 recognition
        # directly on each line strip so DBNet never clips characters at bubble edges.
        line_strips = extract_horizontal_line_strips(raw_crop)
        if line_strips:
            strip_texts: list[str] = []
            strip_scores: list[float] = []
            for strip in line_strips:
                rec_res_obj, _ = engine(
                    strip,
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )
                parsed_rec = _parse_rapid_rec_only(rec_res_obj)
                if parsed_rec:
                    txt, score = parsed_rec[0]
                    strip_texts.append(txt)
                    strip_scores.append(score)
            if strip_texts and (sum(strip_scores) / len(strip_scores)) >= 0.70:
                return " ".join(strip_texts).strip()

        # 2. Full-crop detection + recognition pipeline fallback
        padded_crop = np.asarray(
            cv2.copyMakeBorder(
                raw_crop,
                18,
                18,
                18,
                18,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            ),
            dtype=np.uint8,
        )
        raw_result_obj, _ = engine(padded_crop)
        lines = _parse_rapid_det_rec(raw_result_obj)

        # 3. Single-line recognition fallback if detector returned no boxes on a tight crop
        if not lines:
            rec_only_obj, _ = engine(
                padded_crop,
                use_det=False,
                use_cls=False,
                use_rec=True,
            )
            for txt, _score in _parse_rapid_rec_only(rec_only_obj):
                lines.append((0.0, 0.0, txt))

        if not lines:
            return ""

        lines.sort(key=lambda item: (round(item[0] / 18.0), item[1]))
        return " ".join(item[2] for item in lines).strip()

    def recognize_crop(
        self,
        crop_image: ImageInput,
        source_lang: str = "ja",
        text_direction: TextDirection = "horizontal",
    ) -> str:
        """Extract text from a single pre-cropped speech bubble image."""
        lang = normalize_language(source_lang)
        rgb_crop = load_rgb_image(crop_image)
        if lang == "ja":
            return self._recognize_ja(rgb_crop, text_direction=text_direction)
        return self._recognize_rapid(rgb_crop, lang=lang)

    def recognize_bboxes(
        self,
        image: ImageInput,
        bboxes: Sequence[Sequence[int]],
        source_lang: str = "ja",
    ) -> list[str]:
        """Crop each `[x1, y1, x2, y2]` bounding box from `image` and extract text."""
        lang = normalize_language(source_lang)
        rgb_image = load_rgb_image(image)
        texts: list[str] = []

        for bbox in bboxes:
            raw_crop = crop_bbox_with_padding(
                rgb_image,
                bbox=bbox,
                context_pad=0,
                border_pad=0,
            )
            if lang == "ja":
                h, w = _image_hw(raw_crop)
                direction: TextDirection = "vertical" if h > w * 1.2 else "horizontal"
                texts.append(self._recognize_ja(raw_crop, text_direction=direction))
            else:
                texts.append(self._recognize_rapid(raw_crop, lang=lang))

        return texts

    def recognize_bubbles(
        self,
        image: ImageInput,
        bubbles: Sequence[DetectedBubble],
        source_lang: str = "ja",
    ) -> list[OcrBubbleResult]:
        """Run OCR across a sequence of `DetectedBubble` items and return enriched bubble objects."""
        lang = normalize_language(source_lang)
        rgb_image = load_rgb_image(image)
        results: list[OcrBubbleResult] = []

        for bubble in bubbles:
            raw_crop = crop_bbox_with_padding(
                rgb_image,
                bbox=bubble["bbox"],
                context_pad=0,
                border_pad=0,
            )
            if lang == "ja":
                text = self._recognize_ja(
                    raw_crop,
                    text_direction=bubble["text_direction"],
                )
            else:
                text = self._recognize_rapid(raw_crop, lang=lang)

            results.append(
                {
                    "id": bubble["id"],
                    "order": bubble["order"],
                    "bbox": list(bubble["bbox"]),
                    "polygon": [list(pt) for pt in bubble["polygon"]],
                    "source_text": text,
                    "confidence": bubble["confidence"],
                    "text_direction": bubble["text_direction"],
                }
            )
        return results


def _verify_ocr_on_samples() -> int:
    """CLI verification test running ComicTextDetector + OCRRouter on all 3 sample pages."""
    samples_dir = PROJECT_ROOT / "samples"
    detector = ComicTextDetector()
    ocr_router = OCRRouter()

    sample_cases: list[tuple[SourceLanguage, Path]] = [
        ("ja", samples_dir / "sample_ja.jpg"),
        ("ko", samples_dir / "sample_ko.jpg"),
        ("zh", samples_dir / "sample_zh.jpg"),
    ]

    print("=== Multi-Language OCR Router Verification (ja / ko / zh) ===")
    all_passed = True

    for lang, sample_path in sample_cases:
        if not sample_path.is_file():
            print(f"[ERROR] Missing sample image: {sample_path}", file=sys.stderr)
            all_passed = False
            continue

        det_result = detector.detect(sample_path, source_language=lang)
        ocr_bubbles = ocr_router.recognize_bubbles(
            image=sample_path,
            bubbles=det_result["bubbles"],
            source_lang=lang,
        )

        print(f"\n[{lang.upper()}] {sample_path.name} (bubbles={len(ocr_bubbles)}):")
        if not ocr_bubbles:
            all_passed = False

        for b in ocr_bubbles:
            text = b["source_text"]
            print(
                f"  - {b['id']} (order={b['order']}, bbox={b['bbox']}): {text!r}"
            )
            if not text:
                all_passed = False

    if not all_passed:
        print("\nERROR: OCR verification failed on one or more bubbles.", file=sys.stderr)
        return 1

    print("\nSUCCESS: Extracted text accurately for Japanese (ja), Korean (ko), and Chinese (zh).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_verify_ocr_on_samples())
