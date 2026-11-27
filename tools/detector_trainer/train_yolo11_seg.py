#!/usr/bin/env python3
"""
Vellum Detector Engine v2 - 2-Stage YOLO11-Seg Training Script.

Supports 2-stage transfer learning at imgsz=1024:
  --stage preadapt : 35 epochs on combined public datasets (Manga109 + COO + DCM772).
  --stage finetune : 80 epochs on curated 9-class polygon dataset with manga-safe
                     hyperparameters (AdamW, cos_lr, close_mosaic=15, flipud=0.0,
                     hsv_h=0.01, copy_paste=0.35, scale=0.5, overlap_mask=False).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


def build_train_kwargs(args: argparse.Namespace) -> dict:
    common = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name or f"vellum_{args.stage}",
        "workers": args.workers,
        "seed": args.seed,
        "exist_ok": True,
    }

    if args.stage == "preadapt":
        return {
            **common,
            "epochs": args.epochs if args.epochs is not None else 35,
            "optimizer": "AdamW",
            "cos_lr": True,
            "lr0": 1e-3,
            "lrf": 0.05,
            "flipud": 0.0,
            "hsv_h": 0.01,
            "scale": 0.5,
            "overlap_mask": False,
        }

    # --stage finetune: 80 epochs on curated 9-class polygon dataset
    return {
        **common,
        "epochs": args.epochs if args.epochs is not None else 80,
        "optimizer": "AdamW",
        "cos_lr": True,
        "lr0": 5e-4,
        "lrf": 0.01,
        "close_mosaic": 15,
        "flipud": 0.0,       # Preserve comic gravity & speech tails
        "hsv_h": 0.01,
        "copy_paste": 0.35,  # Boost minority whisper/shout/borderless recall
        "scale": 0.5,
        "overlap_mask": False,  # Retain independent instance masks for overlapping bubbles
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_yolo11_seg.py",
        description="Vellum Detector Engine v2 - 2-Stage YOLO11-Seg Trainer",
    )
    parser.add_argument(
        "--stage",
        choices=["preadapt", "finetune"],
        required=True,
        help="Training stage: 'preadapt' (35ep public datasets) or 'finetune' (80ep curated 9-class).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m-seg.pt",
        help="Base checkpoint or pretrained stage-1 weights (e.g. yolo11m-seg.pt or yolo11l-seg.pt).",
    )
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML (e.g. vellum_9class.yaml).")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size (default: 1024).")
    parser.add_argument("--epochs", type=int, default=None, help="Override stage default epochs (35 or 80).")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--device", type=str, default="0", help="CUDA device or 'cpu' (default: 0).")
    parser.add_argument("--project", type=str, default="runs/detector_v2", help="Output project directory.")
    parser.add_argument("--name", type=str, default="", help="Run name (defaults to vellum_<stage>).")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved training config without running.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    train_kwargs = build_train_kwargs(args)

    if args.dry_run:
        print(f"[train_yolo11_seg] Dry run: model={args.model}")
        for k, v in train_kwargs.items():
            print(f"  {k}: {v}")
        return 0

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        print(f"[train_yolo11_seg] Missing ultralytics ({exc}). Install requirements.txt first.", file=sys.stderr)
        return 1

    model = YOLO(args.model)
    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", Path(args.project) / train_kwargs["name"])
    print(f"[train_yolo11_seg] Training complete. Artifacts saved under: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
