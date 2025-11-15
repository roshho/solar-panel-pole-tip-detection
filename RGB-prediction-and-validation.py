#!/usr/bin/env python3
"""
Run predictions on validation dataset and evaluate results.
This script:
1. Loads the trained YOLO model
2. Runs inference on validation images
3. Converts YOLO predictions to evaluation format
4. Analyzes results with metrics
"""

import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
from collections import defaultdict


def predict_on_validation_set(model_path, val_images_dir, output_json_path, conf_threshold=0.25):
    """
    Run YOLO predictions on validation images and save to JSON.
    
    Args:
        model_path: Path to trained model weights
        val_images_dir: Directory containing validation images
        output_json_path: Where to save predictions JSON
        conf_threshold: Confidence threshold for detections
    """
    print("="*70)
    print("Running predictions on validation set")
    print("="*70)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Get all validation images
    val_images = sorted(Path(val_images_dir).glob("*.png"))
    print(f"Found {len(val_images)} validation images")
    
    # Run predictions
    print(f"\nRunning inference with confidence threshold: {conf_threshold}")
    predictions = []
    
    for i, img_path in enumerate(val_images, 1):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(val_images)}...")
        
        # Run inference
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        
        # Extract detections
        image_pred = {
            'image': img_path.name,
            'tips': []
        }
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # Get bounding box center
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                
                # Get confidence
                conf = float(box.conf[0].cpu().numpy())
                
                image_pred['tips'].append({
                    'pixel_coords': {
                        'u': float(u),
                        'v': float(v)
                    },
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        predictions.append(image_pred)
    
    # Save predictions
    output_data = {'images': predictions}
    
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Predictions saved to: {output_json_path}")
    print(f"   Total images: {len(predictions)}")
    detected = sum(1 for p in predictions if len(p['tips']) > 0)
    print(f"   Images with detections: {detected} ({detected/len(predictions)*100:.1f}%)")
    
    return output_data


def load_yolo_labels(labels_dir, images):
    """
    Load YOLO format ground truth labels.
    
    Returns:
        dict: {filename: {'x': float, 'y': float}} in normalized coordinates
    """
    ground_truth = {}
    
    for img_path in images:
        label_path = Path(labels_dir) / (img_path.stem + '.txt')
        
        if not label_path.exists():
            continue
        
        # Read YOLO label (class x_center y_center width height)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            continue
        
        # Take first line (assuming single class)
        parts = lines[0].strip().split()
        if len(parts) >= 5:
            # YOLO format: class x_center y_center width height (all normalized 0-1)
            x_norm = float(parts[1])
            y_norm = float(parts[2])
            
            ground_truth[img_path.name] = {
                'x_norm': x_norm,
                'y_norm': y_norm
            }
    
    return ground_truth


def evaluate_yolo_predictions(predictions_json, labels_dir, val_images_dir, distance_threshold_px=50):
    """
    Evaluate YOLO predictions against ground truth labels.
    """
    print("\n" + "="*70)
    print("Evaluating predictions")
    print("="*70)
    
    # Load predictions
    with open(predictions_json, 'r') as f:
        pred_data = json.load(f)
    
    predictions = {p['image']: p for p in pred_data['images']}
    
    # Load ground truth
    val_images = list(Path(val_images_dir).glob("*.png"))
    ground_truth = load_yolo_labels(labels_dir, val_images)
    
    print(f"\nDataset statistics:")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Ground truth labels: {len(ground_truth)}")
    print(f"  Predictions: {len(predictions)}")
    
    # Get image dimensions (assuming all images same size)
    sample_img = cv2.imread(str(val_images[0]))
    img_height, img_width = sample_img.shape[:2]
    print(f"  Image dimensions: {img_width}x{img_height}")
    
    # Evaluate
    common_files = set(ground_truth.keys()) & set(predictions.keys())
    print(f"  Common files: {len(common_files)}")
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    distances = []
    results_per_image = []
    
    for filename in sorted(common_files):
        gt = ground_truth[filename]
        pred = predictions[filename]
        
        # Convert ground truth normalized coords to pixels
        gt_x_px = gt['x_norm'] * img_width
        gt_y_px = gt['y_norm'] * img_height
        
        if not pred['tips']:
            # No detection
            false_negatives += 1
            results_per_image.append({
                'filename': filename,
                'status': 'FN',
                'distance': None,
                'gt_x': gt_x_px,
                'gt_y': gt_y_px,
                'pred_x': None,
                'pred_y': None
            })
        else:
            # Use first (highest confidence) detection
            tip = pred['tips'][0]
            pred_x_px = tip['pixel_coords']['u']
            pred_y_px = tip['pixel_coords']['v']
            conf = tip['confidence']
            
            # Calculate distance
            distance = np.sqrt((pred_x_px - gt_x_px)**2 + (pred_y_px - gt_y_px)**2)
            distances.append(distance)
            
            if distance <= distance_threshold_px:
                true_positives += 1
                status = 'TP'
            else:
                false_positives += 1
                status = 'FP'
            
            results_per_image.append({
                'filename': filename,
                'status': status,
                'distance': float(distance),
                'gt_x': float(gt_x_px),
                'gt_y': float(gt_y_px),
                'pred_x': float(pred_x_px),
                'pred_y': float(pred_y_px),
                'confidence': float(conf)
            })
    
    # Calculate metrics
    total = len(common_files)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = true_positives / total if total > 0 else 0
    
    # Distance statistics
    if distances:
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        percentile_95 = np.percentile(distances, 95)
    else:
        mean_dist = median_dist = std_dist = min_dist = max_dist = percentile_95 = None
    
    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS (Threshold: {distance_threshold_px}px)")
    print(f"{'='*70}\n")
    
    print(f"Detection Metrics:")
    print(f"  True Positives (TP):  {true_positives:4d} / {total} ({true_positives/total*100:5.1f}%)")
    print(f"  False Positives (FP): {false_positives:4d} / {total} ({false_positives/total*100:5.1f}%)")
    print(f"  False Negatives (FN): {false_negatives:4d} / {total} ({false_negatives/total*100:5.1f}%)")
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    if distances:
        print(f"\nLocalization Error (pixels):")
        print(f"  Mean:        {mean_dist:6.2f} px")
        print(f"  Median:      {median_dist:6.2f} px")
        print(f"  Std Dev:     {std_dist:6.2f} px")
        print(f"  Min:         {min_dist:6.2f} px")
        print(f"  Max:         {max_dist:6.2f} px")
        print(f"  95th %ile:   {percentile_95:6.2f} px")
    
    # Show worst cases
    print(f"\n{'='*70}")
    print(f"Top 10 Worst Localization Errors:")
    print(f"{'='*70}")
    
    worst_cases = sorted([r for r in results_per_image if r['distance'] is not None],
                        key=lambda x: x['distance'], reverse=True)[:10]
    
    for i, case in enumerate(worst_cases, 1):
        print(f"\n{i:2d}. {case['filename']}")
        print(f"    Distance:  {case['distance']:6.2f} px")
        print(f"    GT:        ({case['gt_x']:7.1f}, {case['gt_y']:7.1f})")
        print(f"    Pred:      ({case['pred_x']:7.1f}, {case['pred_y']:7.1f})")
        print(f"    Conf:      {case['confidence']:.4f}")
    
    # Show false negatives
    fn_cases = [r for r in results_per_image if r['status'] == 'FN']
    if fn_cases:
        print(f"\n{'='*70}")
        print(f"False Negatives (No Detection): {len(fn_cases)}")
        print(f"{'='*70}")
        for i, case in enumerate(fn_cases[:10], 1):
            print(f"{i:2d}. {case['filename']}")
            print(f"    GT: ({case['gt_x']:7.1f}, {case['gt_y']:7.1f})")
    
    # Save detailed results
    metrics_output = {
        'total_images': total,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_distance_px': mean_dist,
        'median_distance_px': median_dist,
        'std_distance_px': std_dist,
        'min_distance_px': min_dist,
        'max_distance_px': max_dist,
        'percentile_95_px': percentile_95,
        'distance_threshold_px': distance_threshold_px,
        'results_per_image': results_per_image
    }
    
    return metrics_output


if __name__ == '__main__':
    # Configuration
    MODEL_PATH = 'runs/train/tip_detector3/weights/best.pt'
    VAL_IMAGES_DIR = 'data/4-yolo_dataset/images/val'
    VAL_LABELS_DIR = 'data/4-yolo_dataset/labels/val'
    PREDICTIONS_JSON = 'OUTPUT-JSON/validation_predictions.json'
    RESULTS_JSON = 'OUTPUT-JSON/validation_evaluation_results.json'
    CONF_THRESHOLD = 0.25
    DISTANCE_THRESHOLD = 50  # pixels
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print(f"   Please train the model first using train.py")
        exit(1)
    
    # Run predictions
    print("\n" + "="*70)
    print("STEP 1: Generate Predictions")
    print("="*70)
    predictions = predict_on_validation_set(
        MODEL_PATH,
        VAL_IMAGES_DIR,
        PREDICTIONS_JSON,
        conf_threshold=CONF_THRESHOLD
    )
    
    # Evaluate predictions
    print("\n" + "="*70)
    print("STEP 2: Evaluate Against Ground Truth")
    print("="*70)
    metrics = evaluate_yolo_predictions(
        PREDICTIONS_JSON,
        VAL_LABELS_DIR,
        VAL_IMAGES_DIR,
        distance_threshold_px=DISTANCE_THRESHOLD
    )
    
    # Save results
    with open(RESULTS_JSON, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Evaluation results saved to: {RESULTS_JSON}")
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
