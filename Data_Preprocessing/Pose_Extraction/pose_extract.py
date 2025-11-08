#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import List

import numpy as np
import cv2
from tqdm import tqdm

# Ultralytics (YOLOv8 Pose)
from ultralytics import YOLO

# COCO 17-keypoint skeleton connectivity (pairs of indices)
COCO_SKELETON = [
    (5, 7), (7, 9),      # Left arm: L-Shoulder -> L-Elbow -> L-Wrist
    (6, 8), (8, 10),     # Right arm: R-Shoulder -> R-Elbow -> R-Wrist
    (5, 6),              # Shoulders
    (5, 11), (6, 12),    # Torso connections to hips
    (11, 12),            # Hips
    (11, 13), (13, 15),  # Left leg: L-Hip -> L-Knee -> L-Ankle
    (12, 14), (14, 16),  # Right leg: R-Hip -> R-Knee -> R-Ankle
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (1, 6)  # Head/ears/eyes and neck to shoulders
]

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

def list_images(input_dir: Path):
    return sorted([p for p in input_dir.rglob("*") if is_image_file(p)])

def draw_skeleton(img: np.ndarray, keypoints: np.ndarray, confs: np.ndarray, min_kpt_conf: float = 0.2) -> np.ndarray:
    """Draw COCO-17 skeleton on a copy of img. keypoints shape: (17,2), confs shape: (17,)"""
    out = img.copy()
    # Draw joints
    for i, (x, y) in enumerate(keypoints):
        if confs[i] >= min_kpt_conf:
            cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    # Draw bones
    for a, b in COCO_SKELETON:
        if 0 <= a < 17 and 0 <= b < 17 and confs[a] >= min_kpt_conf and confs[b] >= min_kpt_conf:
            xa, ya = keypoints[a]
            xb, yb = keypoints[b]
            cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)
    return out

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def save_csv_header(csv_path: Path):
    header = ["image_path", "person_idx", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    for i in range(17):
        header += [f"kpt_{i}_x", f"kpt_{i}_y", f"kpt_{i}_conf"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def append_row(csv_path: Path, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def process_single_image(model: YOLO, image_path: Path, out_csv: Path, draw_dir: Path, conf: float, draw: bool):
    # Run inference
    results = model(source=str(image_path), conf=conf, verbose=False)
    for r in results:
        if r.keypoints is None or len(r.keypoints) == 0:
            return
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4), dtype=np.float32)
        # (N,17,3) -> x, y, conf
        kpts = r.keypoints.data.cpu().numpy()

        img = None
        if draw:
            img = cv2.imread(str(image_path))
            if img is None:
                draw = False

        for idx in range(kpts.shape[0]):
            bbox = boxes[idx] if idx < boxes.shape[0] else np.array([np.nan]*4)
            xy = kpts[idx][:, :2]
            kc = kpts[idx][:, 2]

            row = [str(image_path), idx, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            for i in range(17):
                row += [float(xy[i, 0]), float(xy[i, 1]), float(kc[i])]
            append_row(out_csv, row)

            if draw:
                drawn = draw_skeleton(img, xy, kc, min_kpt_conf=0.2)
                if not np.isnan(bbox).any():
                    x1, y1, x2, y2 = map(int, bbox.tolist())
                    cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 0, 255), 2)
                out_path = draw_dir / f"{image_path.stem}_person{idx}{image_path.suffix}"
                cv2.imwrite(str(out_path), drawn)

def main():
    ap = argparse.ArgumentParser(description="Extract human pose keypoints from images and save to CSV.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_dir", type=str, help="Directory containing images (png/jpg/jpeg).")
    g.add_argument("--input_image", type=str, help="Path to a single image.")
    ap.add_argument("--output_dir", type=str, default="./output", help="Where to write CSV and annotated images.")
    ap.add_argument("--model", type=str, default="yolov8m-pose.pt", help="YOLOv8 pose model weights.")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections.")
    ap.add_argument("--device", type=str, default="cpu", help="Compute device, e.g., 'cpu' or 'cuda:0'.")
    ap.add_argument("--draw", action="store_true", help="If set, save annotated images.")

    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    out_csv = output_dir / "poses.csv"
    save_csv_header(out_csv)
    draw_dir = output_dir / "annotated"
    if args.draw:
        ensure_dir(draw_dir)

    model = YOLO(args.model)
    model.to(args.device)

    images = []
    if args.input_dir:
        in_dir = Path(args.input_dir)
        if not in_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {in_dir}")
        images = [p for p in in_dir.iterdir() if is_image_file(p)]
        if len(images) == 0:
            images = [p for p in in_dir.rglob("*") if is_image_file(p)]
    else:
        p = Path(args.input_image)
        if not p.exists():
            raise FileNotFoundError(f"Input image not found: {p}")
        images = [p]

    for img_path in tqdm(images, desc="Processing images"):
        process_single_image(model, img_path, out_csv, draw_dir, args.conf, args.draw)

    print(f"Done. CSV saved to: {out_csv}")
    if args.draw:
        print(f"Annotated images saved to: {draw_dir}")

if __name__ == "__main__":
    main()
