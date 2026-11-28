#!/usr/bin/env python3
"""
Vellum Detector Engine v2 - Static ONNX Exporter & Quantizer.

Exports a trained YOLO11-Seg .pt checkpoint to static 1024x1024 ONNX (opset=17,
simplify=True, nms=False) and emits 3 deployment-ready weights into `models/`:
  - vellum-det-yolo11m-seg-1024-fp32.onnx (reference)
  - vellum-det-yolo11m-seg-1024-fp16.onnx (primary GPU CUDA/DirectML)
  - vellum-det-yolo11m-seg-1024-int8.onnx (static QDQ INT8 calibrated on 300-500 comic pages)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _letterbox_nchw_fp32(img_path: Path, size: int = 1024):
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(size / float(w), size / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pl, pt = (size - nw) // 2, (size - nh) // 2
    canvas[pt:pt + nh, pl:pl + nw] = resized
    return (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]


def convert_fp32_to_fp16(fp32_path: Path, fp16_path: Path) -> None:
    import onnx  # type: ignore
    from onnxconverter_common import float16  # type: ignore

    model = onnx.load(str(fp32_path))
    fp16_model = float16.convert_float_to_float16(model, keep_io_types=False)
    onnx.save(fp16_model, str(fp16_path))


def quantize_fp32_to_int8(
    fp32_path: Path,
    int8_path: Path,
    calib_dir: Path,
    max_calib_images: int = 500,
    imgsz: int = 1024,
) -> None:
    import onnxruntime as ort  # type: ignore
    from onnxruntime.quantization import (  # type: ignore
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_paths = sorted(p for p in calib_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not img_paths:
        raise RuntimeError(f"No calibration images found in {calib_dir}")
    img_paths = img_paths[:max_calib_images]

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    class ComicCalibReader(CalibrationDataReader):
        def __init__(self, paths: List[Path]):
            self._paths = iter(paths)

        def get_next(self):
            for p in self._paths:
                arr = _letterbox_nchw_fp32(p, imgsz)
                if arr is not None:
                    return {input_name: arr}
            return None

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=ComicCalibReader(img_paths),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_onnx.py",
        description="Vellum Detector Engine v2 - Static 1024x1024 ONNX Exporter (FP32 / FP16 / QDQ INT8)",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained YOLO11-Seg .pt checkpoint.")
    parser.add_argument("--out-dir", type=str, default="models", help="Output directory (default: models).")
    parser.add_argument(
        "--prefix",
        type=str,
        default="vellum-det-yolo11m-seg-1024",
        help="Output filename prefix (default: vellum-det-yolo11m-seg-1024).",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Static export resolution (default: 1024).")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17).")
    parser.add_argument(
        "--calib-dir",
        type=str,
        default="",
        help="Directory of 300-500 real comic pages for static QDQ INT8 calibration.",
    )
    parser.add_argument("--calib-max", type=int, default=500, help="Max calibration images (default: 500).")
    parser.add_argument("--skip-fp16", action="store_true", help="Skip FP16 conversion.")
    parser.add_argument("--skip-int8", action="store_true", help="Skip QDQ INT8 quantization.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        print(f"[export_onnx] Missing ultralytics ({exc}). Install requirements.txt first.", file=sys.stderr)
        return 1

    ckpt = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_out = out_dir / f"{args.prefix}-fp32.onnx"
    fp16_out = out_dir / f"{args.prefix}-fp16.onnx"
    int8_out = out_dir / f"{args.prefix}-int8.onnx"

    print(f"[export_onnx] Exporting {ckpt} -> static {args.imgsz}x{args.imgsz} FP32 ONNX (opset={args.opset}, nms=False)...")
    model = YOLO(str(ckpt))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=True,
            nms=False,
            dynamic=False,
            half=False,
        )
    )
    shutil.move(str(exported_path), str(fp32_out))
    print(f"[export_onnx] Wrote FP32: {fp32_out} ({fp32_out.stat().st_size / (1024*1024):.2f} MB)")

    if not args.skip_fp16:
        try:
            convert_fp32_to_fp16(fp32_out, fp16_out)
            print(f"[export_onnx] Wrote FP16: {fp16_out} ({fp16_out.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as exc:
            print(f"[export_onnx] FP16 conversion failed ({exc}); try installing onnxconverter-common.", file=sys.stderr)

    if not args.skip_int8:
        if not args.calib_dir:
            print("[export_onnx] Skipping INT8 quantization (no --calib-dir provided).")
        else:
            quantize_fp32_to_int8(
                fp32_out,
                int8_out,
                Path(args.calib_dir),
                max_calib_images=args.calib_max,
                imgsz=args.imgsz,
            )
            print(f"[export_onnx] Wrote INT8: {int8_out} ({int8_out.stat().st_size / (1024*1024):.2f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
