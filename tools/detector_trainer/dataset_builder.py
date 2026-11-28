#!/usr/bin/env python3
"""
Vellum Detector Engine v2 - Offline Dataset Curation & QA Toolkit.

Subcommands:
  convert-public : Convert Manga109 XML, COO JSON, and eBDtheque SVG/XML into unified COCO JSON.
  prelabel-sam2  : Semi-automated pre-labeling via ONNX/YOLO base detector + optional SAM 2 mask refiner.
  qa-check       : Audit COCO JSON polygons (>=3 vertices, non-degenerate, area >= 144 px^2),
                   duplicate same-class IoU > 0.85, and minority class quotas.
  split-dataset  : Title-level stratified split (75% train / 15% val / 10% test) grouped by series
                   folder name, exporting Ultralytics YOLO-Seg .txt labels + vellum_9class.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CLASSES: List[Tuple[int, str]] = [
    (0, "speech_bubble"),
    (1, "thought_bubble"),
    (2, "narration_box"),
    (3, "shout_bubble"),
    (4, "whisper_bubble"),
    (5, "borderless_dialogue"),
    (6, "text_region"),
    (7, "panel"),
    (8, "sfx_ignore"),
]

CLASS_NAME_TO_ID: Dict[str, int] = {name: cid for cid, name in CLASSES}

MINORITY_QUOTAS: Dict[str, int] = {
    "thought_bubble": 800,
    "narration_box": 1200,
    "shout_bubble": 800,
    "whisper_bubble": 600,
    "borderless_dialogue": 1000,
    "sfx_ignore": 3000,
}

MIN_POLY_VERTICES = 3
MIN_AREA_PX2 = 144.0
DUPLICATE_IOU_THRESHOLD = 0.85


def coco_categories() -> List[Dict[str, Any]]:
    return [
        {"id": cid, "name": name, "supercategory": "comic_layout"}
        for cid, name in CLASSES
    ]


def polygon_area_and_bbox(flat_poly: Sequence[float]) -> Tuple[float, List[float]]:
    """Compute shoelace area and [x, y, w, h] COCO bbox for a flat [x1, y1, ...] polygon."""
    n = len(flat_poly) // 2
    if n < 3:
        return 0.0, [0.0, 0.0, 0.0, 0.0]
    xs = [float(flat_poly[2 * i]) for i in range(n)]
    ys = [float(flat_poly[2 * i + 1]) for i in range(n)]
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += xs[i] * ys[j] - xs[j] * ys[i]
    area = abs(area2) * 0.5
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return area, [min_x, min_y, max(0.0, max_x - min_x), max(0.0, max_y - min_y)]


def rect_to_flat_poly(x1: float, y1: float, x2: float, y2: float) -> List[float]:
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def bbox_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return (inter / union) if union > 0.0 else 0.0


def is_valid_polygon(flat_poly: Sequence[float], min_area: float = MIN_AREA_PX2) -> Tuple[bool, str]:
    if len(flat_poly) % 2 != 0 or len(flat_poly) < MIN_POLY_VERTICES * 2:
        return False, f"too_few_vertices({len(flat_poly) // 2})"
    n = len(flat_poly) // 2
    pts = [(float(flat_poly[2 * i]), float(flat_poly[2 * i + 1])) for i in range(n)]
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in pts):
        return False, "non_finite_coords"
    unique_pts = {(round(x, 3), round(y, 3)) for x, y in pts}
    if len(unique_pts) < MIN_POLY_VERTICES:
        return False, "degenerate_colocated_vertices"
    area, bbox = polygon_area_and_bbox(flat_poly)
    if area < min_area or bbox[2] <= 0.0 or bbox[3] <= 0.0:
        return False, f"area_below_min({area:.1f}<{min_area:.1f})"
    return True, "ok"


# ------------------------------------------------------------------------------
# 1. convert-public: Manga109 XML, COO JSON, eBDtheque SVG/XML -> COCO JSON
# ------------------------------------------------------------------------------


def _ingest_manga109(
    root: Path,
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    next_img_id: int,
    next_ann_id: int,
) -> Tuple[int, int]:
    ann_dir = root / "annotations" if (root / "annotations").is_dir() else root
    xml_files = sorted(ann_dir.glob("*.xml"))
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        book = tree.getroot()
        series_title = book.attrib.get("title", xml_path.stem)
        pages_node = book.find("pages")
        if pages_node is None:
            continue
        for page in pages_node.findall("page"):
            idx = page.attrib.get("index", "0")
            w = int(page.attrib.get("width", "0"))
            h = int(page.attrib.get("height", "0"))
            rel_name = f"{series_title}/{int(idx):03d}.jpg" if idx.isdigit() else f"{series_title}/{idx}.jpg"
            img_id = next_img_id
            next_img_id += 1
            images.append({
                "id": img_id,
                "file_name": rel_name,
                "width": w,
                "height": h,
                "series": series_title,
                "source": "manga109",
            })

            for tag, cls_name in (("frame", "panel"), ("text", "text_region")):
                cid = CLASS_NAME_TO_ID[cls_name]
                for elem in page.findall(tag):
                    x1 = float(elem.attrib.get("xmin", 0))
                    y1 = float(elem.attrib.get("ymin", 0))
                    x2 = float(elem.attrib.get("xmax", 0))
                    y2 = float(elem.attrib.get("ymax", 0))
                    poly = rect_to_flat_poly(x1, y1, x2, y2)
                    area, bbox = polygon_area_and_bbox(poly)
                    if area <= 0.0:
                        continue
                    annotations.append({
                        "id": next_ann_id,
                        "image_id": img_id,
                        "category_id": cid,
                        "segmentation": [poly],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                    })
                    next_ann_id += 1
    return next_img_id, next_ann_id


def _ingest_coo(
    root: Path,
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    next_img_id: int,
    next_ann_id: int,
) -> Tuple[int, int]:
    json_files = sorted(root.rglob("*.json"))
    sfx_cid = CLASS_NAME_TO_ID["sfx_ignore"]
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        # Support both COO COCO-style dicts and per-page lists/dicts
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            id_map: Dict[Any, int] = {}
            for im in data["images"]:
                img_id = next_img_id
                next_img_id += 1
                id_map[im["id"]] = img_id
                fname = im.get("file_name", f"{jf.stem}.jpg")
                series = Path(fname).parts[0] if len(Path(fname).parts) > 1 else jf.parent.name
                images.append({
                    "id": img_id,
                    "file_name": fname,
                    "width": int(im.get("width", 0)),
                    "height": int(im.get("height", 0)),
                    "series": series,
                    "source": "coo",
                })
            for ann in data["annotations"]:
                mapped_img = id_map.get(ann.get("image_id"))
                if mapped_img is None:
                    continue
                seg = ann.get("segmentation") or []
                if not seg or not isinstance(seg, list):
                    bx = ann.get("bbox")
                    if bx and len(bx) == 4:
                        seg = [rect_to_flat_poly(bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3])]
                if not seg:
                    continue
                area, bbox = polygon_area_and_bbox(seg[0])
                annotations.append({
                    "id": next_ann_id,
                    "image_id": mapped_img,
                    "category_id": sfx_cid,
                    "segmentation": seg,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                })
                next_ann_id += 1
    return next_img_id, next_ann_id


def _parse_svg_points(points_str: str) -> List[float]:
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", points_str)]
    return nums if len(nums) >= 6 and len(nums) % 2 == 0 else []


def _ingest_ebdtheque(
    root: Path,
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    next_img_id: int,
    next_ann_id: int,
) -> Tuple[int, int]:
    svg_files = sorted(root.rglob("*.svg"))
    for svg_path in svg_files:
        tree = ET.parse(svg_path)
        svg_root = tree.getroot()
        w = int(float(svg_root.attrib.get("width", "0").rstrip("px") or 0))
        h = int(float(svg_root.attrib.get("height", "0").rstrip("px") or 0))
        series = svg_path.parent.name
        img_id = next_img_id
        next_img_id += 1
        images.append({
            "id": img_id,
            "file_name": f"{series}/{svg_path.stem}.jpg",
            "width": w,
            "height": h,
            "series": series,
            "source": "ebdtheque",
        })

        for elem in svg_root.iter():
            tag = elem.tag.split("}")[-1].lower()
            cls_hint = (elem.attrib.get("class", "") + " " + elem.attrib.get("id", "")).lower()
            if "panel" in cls_hint:
                cid = CLASS_NAME_TO_ID["panel"]
            elif "balloon" in cls_hint or "bubble" in cls_hint:
                cid = CLASS_NAME_TO_ID["speech_bubble"]
            else:
                continue

            poly: List[float] = []
            if tag == "polygon" and "points" in elem.attrib:
                poly = _parse_svg_points(elem.attrib["points"])
            elif tag == "rect":
                rx = float(elem.attrib.get("x", 0))
                ry = float(elem.attrib.get("y", 0))
                rw = float(elem.attrib.get("width", 0))
                rh = float(elem.attrib.get("height", 0))
                poly = rect_to_flat_poly(rx, ry, rx + rw, ry + rh)
            elif tag == "path" and "d" in elem.attrib:
                poly = _parse_svg_points(elem.attrib["d"])

            if not poly:
                continue
            area, bbox = polygon_area_and_bbox(poly)
            if area <= 0.0:
                continue
            annotations.append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": cid,
                "segmentation": [poly],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            next_ann_id += 1
    return next_img_id, next_ann_id


def cmd_convert_public(args: argparse.Namespace) -> int:
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    next_img_id = 1
    next_ann_id = 1

    if args.manga109:
        next_img_id, next_ann_id = _ingest_manga109(
            Path(args.manga109), images, annotations, next_img_id, next_ann_id
        )
    if args.coo:
        next_img_id, next_ann_id = _ingest_coo(
            Path(args.coo), images, annotations, next_img_id, next_ann_id
        )
    if args.ebdtheque:
        next_img_id, next_ann_id = _ingest_ebdtheque(
            Path(args.ebdtheque), images, annotations, next_img_id, next_ann_id
        )

    coco = {
        "info": {"description": "Vellum 9-Class Unified Public Dataset", "version": "2.0"},
        "categories": coco_categories(),
        "images": images,
        "annotations": annotations,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"[convert-public] Wrote {len(images)} images, {len(annotations)} annotations -> {out_path}")
    return 0


# ------------------------------------------------------------------------------
# 2. prelabel-sam2: Base ONNX/YOLO detector + optional SAM 2 mask refiner
# ------------------------------------------------------------------------------


def cmd_prelabel_sam2(args: argparse.Namespace) -> int:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        print(f"[prelabel-sam2] Missing dependency ({exc}). Install requirements.txt first.", file=sys.stderr)
        return 1

    input_dir = Path(args.images)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not image_paths:
        print(f"[prelabel-sam2] No images found in {input_dir}", file=sys.stderr)
        return 1

    yolo_model = None
    ort_session = None
    if args.detector.endswith(".onnx"):
        import onnxruntime as ort  # type: ignore
        ort_session = ort.InferenceSession(args.detector, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    else:
        from ultralytics import YOLO  # type: ignore
        yolo_model = YOLO(args.detector)

    sam2_predictor = None
    if args.sam2_checkpoint:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
            sam2_predictor = SAM2ImagePredictor.from_pretrained(args.sam2_checkpoint)
        except Exception as exc:
            print(f"[prelabel-sam2] SAM 2 unavailable ({exc}); falling back to base detector masks.", file=sys.stderr)

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1

    for img_id, img_path in enumerate(image_paths, start=1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rel = img_path.relative_to(input_dir).as_posix()
        series = img_path.relative_to(input_dir).parts[0] if len(img_path.relative_to(input_dir).parts) > 1 else input_dir.name
        images.append({"id": img_id, "file_name": rel, "width": w, "height": h, "series": series})

        boxes_xyxy: List[Tuple[int, float, List[float]]] = []
        if yolo_model is not None:
            res = yolo_model.predict(bgr, conf=args.conf, verbose=False)[0]
            if res.boxes is not None:
                for b in res.boxes:
                    cls_id = int(b.cls.item())
                    conf = float(b.conf.item())
                    xyxy = [float(v) for v in b.xyxy[0].tolist()]
                    boxes_xyxy.append((cls_id, conf, xyxy))
        elif ort_session is not None:
            # Generic 1024x1024 letterbox ONNX inference
            scale = min(1024.0 / w, 1024.0 / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (nw, nh))
            canvas = np.full((1024, 1024, 3), 114, dtype=np.uint8)
            pl, pt = (1024 - nw) // 2, (1024 - nh) // 2
            canvas[pt:pt + nh, pl:pl + nw] = resized
            inp = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            out0 = ort_session.run(None, {ort_session.get_inputs()[0].name: inp})[0][0]
            # out0: [4 + C + 32, anchors]
            num_cls = max(0, out0.shape[0] - 4 - 32)
            scores = out0[4:4 + num_cls, :]
            cls_ids = np.argmax(scores, axis=0)
            confs = np.max(scores, axis=0)
            keep = np.where(confs >= args.conf)[0]
            for k in keep[:150]:
                cx, cy, bw, bh = out0[0:4, k]
                x1 = max(0.0, float((cx - 0.5 * bw - pl) / scale))
                y1 = max(0.0, float((cy - 0.5 * bh - pt) / scale))
                x2 = min(float(w), float((cx + 0.5 * bw - pl) / scale))
                y2 = min(float(h), float((cy + 0.5 * bh - pt) / scale))
                boxes_xyxy.append((int(cls_ids[k]), float(confs[k]), [x1, y1, x2, y2]))

        if sam2_predictor is not None and boxes_xyxy:
            sam2_predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        for cls_id, _conf, xyxy in boxes_xyxy:
            poly = rect_to_flat_poly(*xyxy)
            if sam2_predictor is not None:
                masks, _, _ = sam2_predictor.predict(box=np.array(xyxy), multimask_output=False)
                if len(masks) > 0:
                    contours, _ = cv2.findContours(masks[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        eps = 0.005 * cv2.arcLength(largest, True)
                        approx = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2)
                        if len(approx) >= 3:
                            poly = [float(v) for pt in approx for v in pt]

            area, bbox = polygon_area_and_bbox(poly)
            if area < MIN_AREA_PX2:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "segmentation": [poly],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": "Vellum Pre-labeled COCO Dataset", "version": "2.0"},
        "categories": coco_categories(),
        "images": images,
        "annotations": annotations,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"[prelabel-sam2] Exported {len(images)} images, {len(annotations)} prelabels -> {out_path}")
    return 0


# ------------------------------------------------------------------------------
# 3. qa-check: Polygon validity, duplicate same-class IoU > 0.85, class quotas
# ------------------------------------------------------------------------------


def audit_coco_dataset(coco: Dict[str, Any]) -> Dict[str, Any]:
    anns_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    class_counts: Counter[int] = Counter()
    invalid_polygons: List[Dict[str, Any]] = []
    duplicate_pairs: List[Dict[str, Any]] = []

    for ann in coco.get("annotations", []):
        ann_id = ann.get("id")
        img_id = ann.get("image_id")
        cid = int(ann.get("category_id", -1))
        class_counts[cid] += 1

        seg = ann.get("segmentation") or []
        if not seg or not isinstance(seg, list) or not isinstance(seg[0], list):
            invalid_polygons.append({"annotation_id": ann_id, "image_id": img_id, "reason": "missing_segmentation"})
            continue

        ok, reason = is_valid_polygon(seg[0], MIN_AREA_PX2)
        if not ok:
            invalid_polygons.append({"annotation_id": ann_id, "image_id": img_id, "reason": reason})

        _, bbox = polygon_area_and_bbox(seg[0])
        anns_by_img[img_id].append({"id": ann_id, "category_id": cid, "bbox": bbox})

    for img_id, items in anns_by_img.items():
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                if items[i]["category_id"] != items[j]["category_id"]:
                    continue
                iou = bbox_iou_xywh(items[i]["bbox"], items[j]["bbox"])
                if iou > DUPLICATE_IOU_THRESHOLD:
                    duplicate_pairs.append({
                        "image_id": img_id,
                        "ann_a": items[i]["id"],
                        "ann_b": items[j]["id"],
                        "category_id": items[i]["category_id"],
                        "iou": round(iou, 4),
                    })

    quota_status: Dict[str, Dict[str, Any]] = {}
    for cls_name, target in MINORITY_QUOTAS.items():
        cid = CLASS_NAME_TO_ID[cls_name]
        count = class_counts.get(cid, 0)
        quota_status[cls_name] = {"count": count, "target": target, "met": count >= target}

    return {
        "total_images": len(coco.get("images", [])),
        "total_annotations": len(coco.get("annotations", [])),
        "invalid_polygons": invalid_polygons,
        "duplicate_pairs": duplicate_pairs,
        "class_counts": {name: class_counts.get(cid, 0) for cid, name in CLASSES},
        "quota_status": quota_status,
    }


def cmd_qa_check(args: argparse.Namespace) -> int:
    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    report = audit_coco_dataset(coco)

    print(f"[qa-check] Images: {report['total_images']}, Annotations: {report['total_annotations']}")
    print(f"[qa-check] Invalid polygons : {len(report['invalid_polygons'])}")
    for item in report["invalid_polygons"][:10]:
        print(f"  - ann={item['annotation_id']} img={item['image_id']} reason={item['reason']}")
    print(f"[qa-check] Duplicate pairs  : {len(report['duplicate_pairs'])}")
    for dup in report["duplicate_pairs"][:10]:
        print(f"  - img={dup['image_id']} anns=({dup['ann_a']},{dup['ann_b']}) cls={dup['category_id']} IoU={dup['iou']}")

    print("[qa-check] Class counts & minority quotas:")
    for cls_name, cnt in report["class_counts"].items():
        q = report["quota_status"].get(cls_name)
        if q:
            status = "OK" if q["met"] else f"BELOW TARGET ({q['target']})"
            print(f"  - {cls_name:<20}: {cnt:>6}  [{status}]")
        else:
            print(f"  - {cls_name:<20}: {cnt:>6}")

    if args.strict and (report["invalid_polygons"] or report["duplicate_pairs"]):
        return 2
    return 0


# ------------------------------------------------------------------------------
# 4. split-dataset: Title-level stratified 75/15/10 split -> YOLO-Seg + YAML
# ------------------------------------------------------------------------------


def _extract_series_key(img_entry: Dict[str, Any]) -> str:
    if img_entry.get("series"):
        return str(img_entry["series"])
    parts = Path(img_entry["file_name"]).parts
    return parts[0] if len(parts) > 1 else "__default__"


def split_coco_by_series(
    coco: Dict[str, Any],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.75, 0.15, 0.10),
) -> Dict[str, List[Dict[str, Any]]]:
    """Group images strictly by series folder so zero pages from the same mangaka leak across splits."""
    by_series: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for img in coco.get("images", []):
        by_series[_extract_series_key(img)].append(img)

    series_items = sorted(by_series.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    rng = random.Random(seed)
    # Deterministic shuffle within equal-size groups
    rng.shuffle(series_items)
    series_items.sort(key=lambda kv: -len(kv[1]))

    total_imgs = sum(len(v) for _, v in series_items)
    targets = {
        "train": ratios[0] * total_imgs,
        "val": ratios[1] * total_imgs,
        "test": ratios[2] * total_imgs,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for _series, imgs in series_items:
        # Assign entire series to the split furthest below its target ratio
        best_split = max(
            ("train", "val", "test"),
            key=lambda s: (targets[s] - counts[s]) / max(1.0, targets[s]),
        )
        splits[best_split].extend(imgs)
        counts[best_split] += len(imgs)

    return splits


def cmd_split_dataset(args: argparse.Namespace) -> int:
    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    out_dir = Path(args.output)
    images_root = Path(args.images_root) if args.images_root else None

    anns_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    splits = split_coco_by_series(coco, seed=args.seed)

    for split_name, split_imgs in splits.items():
        lbl_dir = out_dir / "labels" / split_name
        img_dir = out_dir / "images" / split_name
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)

        for img in split_imgs:
            w = max(1, int(img.get("width", 1)))
            h = max(1, int(img.get("height", 1)))
            rel = Path(img["file_name"])
            safe_stem = "__".join(rel.with_suffix("").parts)
            lines: List[str] = []
            for ann in anns_by_img.get(img["id"], []):
                cid = int(ann["category_id"])
                seg = ann.get("segmentation") or []
                if not seg or not isinstance(seg[0], list):
                    continue
                poly = seg[0]
                if len(poly) < 6:
                    continue
                norm_coords: List[str] = []
                for i in range(0, len(poly), 2):
                    nx = min(1.0, max(0.0, float(poly[i]) / w))
                    ny = min(1.0, max(0.0, float(poly[i + 1]) / h))
                    norm_coords.append(f"{nx:.6f} {ny:.6f}")
                lines.append(f"{cid} " + " ".join(norm_coords))

            (lbl_dir / f"{safe_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            if images_root is not None:
                src_img = images_root / rel
                if src_img.is_file():
                    shutil.copy2(src_img, img_dir / f"{safe_stem}{src_img.suffix}")

    yaml_lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(CLASSES)}",
        "names:",
    ]
    for cid, name in CLASSES:
        yaml_lines.append(f"  {cid}: {name}")
    yaml_path = out_dir / "vellum_9class.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # Verify zero series overlap
    split_series = {
        s: {_extract_series_key(im) for im in imgs}
        for s, imgs in splits.items()
    }
    leak_tv = split_series["train"] & split_series["val"]
    leak_tt = split_series["train"] & split_series["test"]
    leak_vt = split_series["val"] & split_series["test"]
    if leak_tv or leak_tt or leak_vt:
        print(f"[split-dataset] ERROR: series leakage detected: {leak_tv | leak_tt | leak_vt}", file=sys.stderr)
        return 2

    print(
        f"[split-dataset] train={len(splits['train'])} ({len(split_series['train'])} series), "
        f"val={len(splits['val'])} ({len(split_series['val'])} series), "
        f"test={len(splits['test'])} ({len(split_series['test'])} series) -> {yaml_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset_builder.py",
        description="Vellum Detector Engine v2 - Offline Dataset Curation & QA Toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_conv = sub.add_parser("convert-public", help="Convert Manga109/COO/eBDtheque datasets into unified COCO JSON")
    p_conv.add_argument("--manga109", type=str, default="", help="Path to Manga109 root/annotations directory")
    p_conv.add_argument("--coo", type=str, default="", help="Path to COO SFX JSON directory")
    p_conv.add_argument("--ebdtheque", type=str, default="", help="Path to eBDtheque SVG/XML directory")
    p_conv.add_argument("--output", "-o", type=str, required=True, help="Output COCO JSON file path")
    p_conv.set_defaults(func=cmd_convert_public)

    p_pre = sub.add_parser("prelabel-sam2", help="Pre-label images via ONNX/YOLO base detector + optional SAM 2")
    p_pre.add_argument("--detector", type=str, required=True, help="Path to base .onnx or .pt detector")
    p_pre.add_argument("--images", type=str, required=True, help="Input images directory")
    p_pre.add_argument("--sam2-checkpoint", type=str, default="", help="Optional SAM 2 model/checkpoint ID")
    p_pre.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p_pre.add_argument("--output", "-o", type=str, required=True, help="Output pre-labeled COCO JSON path")
    p_pre.set_defaults(func=cmd_prelabel_sam2)

    p_qa = sub.add_parser("qa-check", help="Run automated polygon, duplicate IoU, and class-quota QA checks")
    p_qa.add_argument("--coco", type=str, required=True, help="Input COCO JSON file path")
    p_qa.add_argument("--strict", action="store_true", help="Exit non-zero if invalid polygons or duplicates found")
    p_qa.set_defaults(func=cmd_qa_check)

    p_split = sub.add_parser("split-dataset", help="Title-level stratified 75/15/10 split to YOLO-Seg + YAML")
    p_split.add_argument("--coco", type=str, required=True, help="Input COCO JSON file path")
    p_split.add_argument("--images-root", type=str, default="", help="Optional root directory of source images to copy")
    p_split.add_argument("--output", "-o", type=str, required=True, help="Output YOLO-Seg dataset directory")
    p_split.add_argument("--seed", type=int, default=42, help="Random seed")
    p_split.set_defaults(func=cmd_split_dataset)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
