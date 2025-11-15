#!/usr/bin/env python3
# RUN-ME.py (modified)
# Process sets of RGB + depth images under INPUT-RGBD and write per-set JSON results into OUTPUT-JSON.
# Optionally draw bboxes with confidence on RGB images.

from pathlib import Path
import json
from typing import Optional, Tuple, List, Dict
import argparse

import cv2
import numpy as np
from ultralytics import YOLO


class DepthConverter:
	"""Simple pinhole back-projection using camera intrinsics (meters)."""

	def __init__(self, fx: float, fy: float, cx: float, cy: float):
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy

	def pixel_to_3d(self, u: float, v: float, depth_mm: float) -> Optional[Tuple[float, float, float]]:
		if not np.isfinite(depth_mm) or depth_mm <= 0:
			return None
		z = float(depth_mm) / 1000.0
		x = (u - self.cx) * z / self.fx
		y = (v - self.cy) * z / self.fy
		return x, y, z

	def bbox_center_to_3d(self, bbox: np.ndarray, depth_map: np.ndarray, win: int = 5) -> Optional[Tuple[float, float, float]]:
		x1, y1, x2, y2 = bbox
		u = (x1 + x2) / 2.0
		v = (y1 + y2) / 2.0

		ui = int(round(u))
		vi = int(round(v))
		hw = win // 2

		x0 = max(0, ui - hw)
		x1i = min(depth_map.shape[1], ui + hw + 1)
		y0 = max(0, vi - hw)
		y1i = min(depth_map.shape[0], vi + hw + 1)

		patch = depth_map[y0:y1i, x0:x1i]
		if patch.size == 0:
			return None
		vals = patch[np.isfinite(patch)]
		if vals.size == 0:
			return None
		depth_mm = float(np.median(vals))
		return self.pixel_to_3d(u, v, depth_mm)


def find_depth_for_rgb_in_set(rgb_path: Path, set_dir: Path) -> Optional[Path]:
	"""
	Try to find an associated depth TIFF for this RGB under the same set directory.
	Supports patterns like:
	  name.png -> name_depth.tiff
	  name_rgb.png -> name_depth.tiff
	  name_rgb_aug000.png -> name_depth.tiff
	  name_aug000_rgb.png -> name_depth.tiff
	Also looks in common subfolders like 'depth', 'depths', 'depth_maps'.
	"""
	stem = rgb_path.stem
	# remove trailing _rgb if present
	if stem.endswith("_rgb"):
		stem = stem[: -4]
	parts = stem.split("_")
	if parts and parts[-1].startswith("aug"):
		parts = parts[:-1]
	stem = "_".join(parts)

	candidates = [
		f"{stem}_depth.tiff",
		f"{stem}_depth.tif",
		f"{stem}_depth.png",
		f"{stem}.tiff",
		f"{stem}.tif",
	]
	# first check alongside rgb
	for cand in candidates:
		p = set_dir / cand
		if p.exists():
			return p

	# check common subfolders
	for sub in ("depth", "depths", "depth_maps", "Depth", "depth_tiff"):
		for cand in candidates:
			p = set_dir / sub / cand
			if p.exists():
				return p

	# fallback: search for any file in set_dir with same stem and tiff extension (recursive)
	for ext in ("*.tiff", "*.tif"):
		for p in set_dir.rglob(ext):
			if p.stem.startswith(stem):
				return p
	return None


def draw_boxes_on_image(img: np.ndarray, boxes: List[np.ndarray], confs: List[float], color=(0, 255, 0)):
	out = img.copy()
	for bbox, conf in zip(boxes, confs):
		x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
		cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
		txt = f"{conf*100:.2f}%"
		cv2.putText(out, txt, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
	return out


def process_set(
	set_dir: Path,
	model: YOLO,
	conv: DepthConverter,
	output_json_dir: Path,
	conf_threshold: float = 0.01,
	draw: bool = False,
):
	images = sorted(set_dir.glob("*.png"))
	if not images:
		return None

	results = {
		"set_name": set_dir.name,
		"camera_intrinsics": {"fx": conv.fx, "fy": conv.fy, "cx": conv.cx, "cy": conv.cy, "resolution": [1920, 1080]},
		"summary": {
			"total_images": len(images),
			"images_with_detections": 0,
			"total_tips_detected": 0,
			"images_with_depth": 0,
			"tips_with_3d_coords": 0,
		},
		"images": [],
	}

	annotated_dir = output_json_dir / f"{set_dir.name}_annotated"
	if draw:
		annotated_dir.mkdir(parents=True, exist_ok=True)

	for i, rgb_path in enumerate(images, start=1):
		depth_path = find_depth_for_rgb_in_set(rgb_path, set_dir)
		depth_map = None
		if depth_path:
			results["summary"]["images_with_depth"] += 1
			depth_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

		dets = model(str(rgb_path), conf=conf_threshold, verbose=False)
		boxes = dets[0].boxes
		if len(boxes) == 0:
			continue

		results["summary"]["images_with_detections"] += 1
		results["summary"]["total_tips_detected"] += len(boxes)

		img_entry = {
			"image": rgb_path.name,
			"image_path": str(rgb_path),
			"depth_available": depth_path is not None,
			"depth_path": str(depth_path) if depth_path else None,
			"num_tips": len(boxes),
			"tips": [],
		}

		# optionally load rgb for drawing
		rgb_img = None
		drawn_boxes = []
		drawn_confs = []

		for tidx, b in enumerate(boxes, 1):
			bbox_xyxy = b.xyxy[0].cpu().numpy()
			conf = float(b.conf[0])
			u = float((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0)
			v = float((bbox_xyxy[1] + bbox_xyxy[3]) / 2.0)

			tip = {
				"tip_id": tidx,
				"confidence": conf,
				"bbox": {
					"x1": float(bbox_xyxy[0]),
					"y1": float(bbox_xyxy[1]),
					"x2": float(bbox_xyxy[2]),
					"y2": float(bbox_xyxy[3]),
				},
				"pixel_coords": {"u": u, "v": v},
				"world_coords_meters": None,
			}

			if depth_map is not None:
				coords = conv.bbox_center_to_3d(bbox_xyxy, depth_map)
				if coords is not None:
					x, y, z = coords
					tip["world_coords_meters"] = {"x": x, "y": y, "z": z}
					results["summary"]["tips_with_3d_coords"] += 1

			img_entry["tips"].append(tip)

			if draw:
				if rgb_img is None:
					rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
				drawn_boxes.append(bbox_xyxy)
				drawn_confs.append(conf)

		results["images"].append(img_entry)

		if draw and rgb_img is not None and drawn_boxes:
			annotated = draw_boxes_on_image(rgb_img, drawn_boxes, drawn_confs)
			out_path = annotated_dir / rgb_path.name
			cv2.imwrite(str(out_path), annotated)

	out_json = output_json_dir / f"{set_dir.name}.json"
	output_json_dir.mkdir(parents=True, exist_ok=True)
	out_json.write_text(json.dumps(results, indent=2))
	return out_json


def find_sets_under_root(input_root: Path) -> List[Path]:
	# Treat immediate subdirectories that contain PNGs as sets. If root itself has PNGs, include root.
	sets = []
	if any(input_root.glob("*.png")):
		sets.append(input_root)
	for child in sorted(input_root.iterdir()):
		if child.is_dir() and any(child.glob("*.png")):
			sets.append(child)
	return sets


def main():
	p = argparse.ArgumentParser(description="Process INPUT-RGBD sets and produce JSON outputs.")
	p.add_argument("input_root", help="Path to INPUT-RGBD folder containing one or more sets (subfolders) of images")
	p.add_argument("output_json_dir", help="Folder to write per-set JSON results (OUTPUT-JSON)")
	p.add_argument("--model", default="runs/train/tip_detector3/weights/best.pt", help="YOLO model path")
	p.add_argument("--conf", type=float, default=0.01, help="detection confidence threshold")
	p.add_argument("--draw", action="store_true", help="draw bounding boxes and save annotated images")
	p.add_argument("--fx", type=float, default=1067.0)
	p.add_argument("--fy", type=float, default=1067.0)
	p.add_argument("--cx", type=float, default=960.0)
	p.add_argument("--cy", type=float, default=540.0)
	args = p.parse_args()

	input_root = Path(args.input_root)
	output_json_dir = Path(args.output_json_dir)
	model = YOLO(args.model)
	conv = DepthConverter(args.fx, args.fy, args.cx, args.cy)

	sets = find_sets_under_root(input_root)
	if not sets:
		print("No image sets found under", input_root)
		return

	print(f"Found {len(sets)} set(s) to process.")
	for s in sets:
		print("Processing set:", s)
		out = process_set(s, model, conv, output_json_dir, conf_threshold=args.conf, draw=args.draw)
		if out:
			print("Wrote:", out)

	print("Done.")


if __name__ == "__main__":
	main()
