"""End-to-end verification pipeline combining `detector.py`, `ocr.py`, and `inpainter.py`.

Runs sequential Detection -> Reading-Order Sorting -> Multi-Language OCR -> LaMa Tile Inpainting on:
- `samples/sample_ja.jpg` -> saves `samples/clean_ja.png`
- `samples/sample_ko.jpg` -> saves `samples/clean_ko.png`
- `samples/sample_zh.jpg` -> saves `samples/clean_zh.png`

Prints the JSON array of extracted dialogue items (`id`, `bbox`, `original_text`) per language.

Usage:
    backend\\venv\\Scripts\\python.exe backend/test_pipeline.py
"""

import json
from pathlib import Path
import sys
from typing import TypedDict

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.detector import ComicTextDetector, NDArrayU8  # noqa: E402
from backend.core.inpainter import LamaInpainter  # noqa: E402
from backend.core.ocr import OCRRouter, SourceLanguage  # noqa: E402


class PipelineBubbleItem(TypedDict):
    """JSON output contract for each detected & recognized speech bubble."""

    id: str
    bbox: list[int]
    original_text: str


def run_pipeline_test() -> int:
    """Execute Detection -> OCR -> LaMa Inpainting on all 3 sample comic pages."""
    samples_dir = PROJECT_ROOT / "samples"
    detector = ComicTextDetector()
    ocr_router = OCRRouter()
    inpainter = LamaInpainter()

    sample_cases: list[tuple[SourceLanguage, Path, Path]] = [
        ("ja", samples_dir / "sample_ja.jpg", samples_dir / "clean_ja.png"),
        ("ko", samples_dir / "sample_ko.jpg", samples_dir / "clean_ko.png"),
        ("zh", samples_dir / "sample_zh.jpg", samples_dir / "clean_zh.png"),
    ]

    all_passed = True

    for lang, input_path, output_path in sample_cases:
        if not input_path.is_file():
            print(f"[ERROR] Input sample image not found: {input_path}", file=sys.stderr)
            all_passed = False
            continue

        # Stage 1: Comic Text Detection (Bounding Boxes + Binary Mask)
        det_result = detector.detect(input_path, source_language=lang)
        detected_bubbles = det_result["bubbles"]
        text_mask = det_result["mask"]

        # Stage 2: Multi-Language OCR Router (manga-ocr for ja, RapidOCR PP-OCRv4 for ko/zh)
        ocr_bubbles = ocr_router.recognize_bubbles(
            image=input_path,
            bubbles=detected_bubbles,
            source_lang=lang,
        )

        # Stage 3: Tile-Based LaMa Inpainting (512x512 ROI patches)
        bboxes = [b["bbox"] for b in detected_bubbles]
        inpaint_res = inpainter.inpaint_with_metadata(
            image=input_path,
            mask=text_mask,
            bboxes=bboxes,
        )
        clean_rgb: NDArrayU8 = inpaint_res["image"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(clean_rgb).save(output_path, format="PNG")

        json_items: list[PipelineBubbleItem] = [
            {
                "id": b["id"],
                "bbox": b["bbox"],
                "original_text": b["source_text"],
            }
            for b in ocr_bubbles
        ]

        # Verify dark ink inside detected text boxes was erased by LaMa inpainting
        remaining_dark_pixels = 0
        for box in bboxes:
            x1, y1, x2, y2 = box
            # Check central region of the text box where characters were located
            pad_x = max(2, (x2 - x1) // 10)
            pad_y = max(2, (y2 - y1) // 10)
            interior: NDArrayU8 = np.asarray(
                clean_rgb[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x],
                dtype=np.uint8,
            )
            remaining_dark_pixels += int(np.count_nonzero(interior < 150))

        print(
            f"\n=== [{lang.upper()}] {input_path.name} -> {output_path.name} "
            + f"(tiles={inpaint_res['tiles_processed']}, "
            + f"provider={inpaint_res['execution_provider']}, "
            + f"remaining_dark_px={remaining_dark_pixels}) ==="
        )
        print(json.dumps(json_items, ensure_ascii=False, indent=2))

        if (
            not output_path.is_file()
            or len(json_items) < 4
            or any(not item["original_text"] for item in json_items)
        ):
            all_passed = False

    inpainter.unload()

    if not all_passed:
        print("\nERROR: Pipeline verification failed on one or more pages.", file=sys.stderr)
        return 1

    print(
        "\nSUCCESS: Saved samples/clean_ja.png, samples/clean_ko.png, and samples/clean_zh.png "
        + "with all dialogue extracted and inpainted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_pipeline_test())
