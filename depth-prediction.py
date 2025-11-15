#!/usr/bin/env python3
"""
Evaluate 3D (x, y, z) predictions by adding depth information to RGB predictions.
This script:
1. Loads RGB predictions from OUTPUT-JSON
2. Finds corresponding depth maps
3. Converts pixel coordinates to 3D world coordinates
4. Analyzes depth estimation quality
"""

import json
import numpy as np
from pathlib import Path
import cv2
from typing import Optional, Tuple


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
        z = float(depth_mm) / 1000.0  # Convert mm to meters
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return x, y, z
    
    def point_to_3d(self, u: float, v: float, depth_map: np.ndarray, win: int = 5) -> Optional[Tuple[float, float, float]]:
        """Get 3D coordinates for a pixel point using depth map."""
        ui = int(round(u))
        vi = int(round(v))
        hw = win // 2
        
        x0 = max(0, ui - hw)
        x1 = min(depth_map.shape[1], ui + hw + 1)
        y0 = max(0, vi - hw)
        y1 = min(depth_map.shape[0], vi + hw + 1)
        
        patch = depth_map[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        vals = patch[np.isfinite(patch)]
        if vals.size == 0:
            return None
        depth_mm = float(np.median(vals))
        return self.pixel_to_3d(u, v, depth_mm)


def find_depth_for_rgb(rgb_filename: str, depth_dir: Path) -> Optional[Path]:
    """Find corresponding depth TIFF for RGB image."""
    # Remove _rgb and _aug* suffixes
    stem = Path(rgb_filename).stem
    
    # Remove _rgb suffix
    if stem.endswith("_rgb"):
        stem = stem[:-4]
    
    # Remove augmentation suffix (_aug000, etc.)
    parts = stem.split("_")
    if parts and parts[-1].startswith("aug"):
        parts = parts[:-1]
    stem = "_".join(parts)
    
    # Try different depth file patterns
    candidates = [
        f"{stem}_depth.tiff",
        f"{stem}_depth.tif",
    ]
    
    for cand in candidates:
        depth_path = depth_dir / cand
        if depth_path.exists():
            return depth_path
    
    return None


def add_depth_to_predictions(predictions_json: str, depth_dir: str, output_json: str):
    """
    Add 3D world coordinates to predictions using depth maps.
    
    Args:
        predictions_json: Path to RGB predictions JSON
        depth_dir: Directory containing depth TIFF files
        output_json: Where to save predictions with 3D coordinates
    """
    print("="*70)
    print("Adding 3D Depth Information to Predictions")
    print("="*70)
    
    # Load predictions
    print(f"\nLoading predictions from: {predictions_json}")
    with open(predictions_json, 'r') as f:
        data = json.load(f)
    
    predictions = data['images']
    print(f"  Loaded {len(predictions)} predictions")
    
    # Initialize depth converter (ZED X camera intrinsics)
    # These are typical values - adjust if you have specific calibration
    fx = 1059.0  # focal length x
    fy = 1059.0  # focal length y  
    cx = 960.0   # principal point x (half of 1920)
    cy = 540.0   # principal point y (half of 1080)
    
    conv = DepthConverter(fx, fy, cx, cy)
    print(f"\nCamera intrinsics:")
    print(f"  fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    depth_dir = Path(depth_dir)
    print(f"\nDepth directory: {depth_dir}")
    
    # Process predictions
    stats = {
        'total_predictions': len(predictions),
        'images_with_detections': 0,
        'detections_with_depth_found': 0,
        'detections_with_valid_3d': 0,
        'depth_files_found': 0,
        'depth_files_missing': 0
    }
    
    results_with_3d = []
    depth_values = []
    
    for pred in predictions:
        filename = pred['image']
        tips = pred['tips']
        
        if not tips:
            results_with_3d.append(pred)
            continue
        
        stats['images_with_detections'] += 1
        
        # Find corresponding depth file
        depth_path = find_depth_for_rgb(filename, depth_dir)
        
        if depth_path is None:
            stats['depth_files_missing'] += 1
            # Add prediction without 3D coords
            for tip in tips:
                tip['world_coords_meters'] = None
                tip['depth_available'] = False
            results_with_3d.append(pred)
            continue
        
        stats['depth_files_found'] += 1
        
        # Load depth map
        depth_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            print(f"  Warning: Could not load depth map: {depth_path}")
            for tip in tips:
                tip['world_coords_meters'] = None
                tip['depth_available'] = False
            results_with_3d.append(pred)
            continue
        
        # Add 3D coordinates to each tip
        for tip in tips:
            u = tip['pixel_coords']['u']
            v = tip['pixel_coords']['v']
            
            coords_3d = conv.point_to_3d(u, v, depth_map)
            
            if coords_3d is not None:
                x, y, z = coords_3d
                tip['world_coords_meters'] = {
                    'x': float(x),
                    'y': float(y),
                    'z': float(z)
                }
                tip['depth_available'] = True
                stats['detections_with_valid_3d'] += 1
                stats['detections_with_depth_found'] += 1
                depth_values.append(z)
            else:
                tip['world_coords_meters'] = None
                tip['depth_available'] = True  # Depth file exists but no valid depth
                stats['detections_with_depth_found'] += 1
        
        results_with_3d.append(pred)
    
    # Save results
    output_data = {
        'camera_intrinsics': {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'resolution': [1920, 1080]
        },
        'statistics': stats,
        'images': results_with_3d
    }
    
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"3D CONVERSION RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Detection Statistics:")
    print(f"  Total predictions:        {stats['total_predictions']}")
    print(f"  Images with detections:   {stats['images_with_detections']}")
    
    print(f"\nDepth Map Availability:")
    print(f"  Depth files found:        {stats['depth_files_found']}")
    print(f"  Depth files missing:      {stats['depth_files_missing']}")
    miss_pct = stats['depth_files_missing'] / stats['images_with_detections'] * 100 if stats['images_with_detections'] > 0 else 0
    print(f"  Missing rate:             {miss_pct:.1f}%")
    
    print(f"\n3D Coordinate Generation:")
    print(f"  Detections with depth:    {stats['detections_with_depth_found']}")
    print(f"  Valid 3D coordinates:     {stats['detections_with_valid_3d']}")
    
    if stats['detections_with_depth_found'] > 0:
        success_rate = stats['detections_with_valid_3d'] / stats['detections_with_depth_found'] * 100
        print(f"  3D success rate:          {success_rate:.1f}%")
    
    if depth_values:
        print(f"\nDepth Statistics (Z-coordinate in meters):")
        print(f"  Mean depth:    {np.mean(depth_values):.3f} m")
        print(f"  Median depth:  {np.median(depth_values):.3f} m")
        print(f"  Std deviation: {np.std(depth_values):.3f} m")
        print(f"  Min depth:     {np.min(depth_values):.3f} m")
        print(f"  Max depth:     {np.max(depth_values):.3f} m")
        
        # Show depth distribution
        print(f"\nDepth Distribution:")
        bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 100]
        hist, _ = np.histogram(depth_values, bins=bins)
        for i in range(len(bins)-1):
            if i == len(bins)-2:
                label = f"  >{bins[i]:.1f}m:"
            else:
                label = f"  {bins[i]:.1f}-{bins[i+1]:.1f}m:"
            count = hist[i]
            pct = count / len(depth_values) * 100
            bar = "█" * int(pct / 2)
            print(f"{label:12s} {count:3d} ({pct:5.1f}%) {bar}")
    
    print(f"\n✅ Results saved to: {output_json}")
    
    return output_data


if __name__ == '__main__':
    # Use the validation predictions we just generated
    PREDICTIONS_JSON = 'OUTPUT-JSON/validation_predictions.json'
    DEPTH_DIR = 'data/1-OG/OG-framed-only'
    OUTPUT_JSON = 'OUTPUT-JSON/validation_predictions_3d.json'
    
    # Check files exist
    if not Path(PREDICTIONS_JSON).exists():
        print(f"❌ Error: Predictions file not found: {PREDICTIONS_JSON}")
        print(f"   Please run predict_and_evaluate.py first")
        exit(1)
    
    if not Path(DEPTH_DIR).exists():
        print(f"❌ Error: Depth directory not found: {DEPTH_DIR}")
        exit(1)
    
    # Add depth information
    results = add_depth_to_predictions(
        PREDICTIONS_JSON,
        DEPTH_DIR,
        OUTPUT_JSON
    )
    
    print("\n" + "="*70)
    print("3D Evaluation Complete!")
    print("="*70)
