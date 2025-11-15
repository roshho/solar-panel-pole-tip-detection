# Davis Mechatronics Interview Challenge

![Title](title.png)

## Overview
![Pre-Configuration](pre-config.png)

![Model Fine-Tuning](model-finetuning.png)

## Results

![Example Labeling](example-labeling.png)
### RGB Accuracies

#### Localization Error (pixels)
| Metric | Value |
|--------|-------|
| Mean | 0.90 px |
| Median | 0.69 px |
| Std Dev | 0.69 px |
| Min | 0.06 px |
| Max | 4.25 px |
| 95th percentile | 2.43 px |

#### Detection Metrics
| Metric | Count | Percentage |
|--------|-------|------------|
| **True Positives (TP)** | 246 / 306 | 80.4% |
| **False Positives (FP)** | 0 / 306 | 0.0% |
| **False Negatives (FN)** | 60 / 306 | 19.6% (primarily augmented) |

#### Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 80.39% | Overall correctness |
| **Precision** | 100.00% | ✅ Correct ID-ing (no false positives) |
| **Recall** | 80.39% | Detection rate when pole present |
| **F1-Score** | 89.13% | Harmonic mean of precision/recall |

> **Note**: False negatives primarily occur on augmented images. Confidence threshold: 0.01

### Depth Estimation

![Depth Estimation Distribution](depth-estimation-piechart.png)

| Metric | Value |
|--------|-------|
| Mean depth | 2.587 m |
| Median depth | 2.546 m |
| Std deviation | 0.324 m |
| Min depth | 1.895 m |
| Max depth | 3.414 m |



## Challenge requirements
- input: pictures, output: x,y,z
- front and back is different, pictures are combined. can assume camera is fixed. 
- camera used is zed X
- possible that there's no pole too, e.g. 20250826_111428_front_frame000123_rgb, but not a primary concern
- to be run real time on nvidia orin

## Setup
1. Create a virtualenv (optional but recommended): 
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies: 
   ```
   pip install -r requirements.txt
   ```

## Usage

### Production Inference (RGBD → JSON)
Run predictions on RGB+Depth image pairs and output 3D coordinates:
```
python RGBD-prediction.py INPUT-RGBD OUTPUT-JSON --draw
```
or without visualization:
```
python RGBD-prediction.py INPUT-RGBD OUTPUT-JSON
```

**Output**: JSON files with 2D pixel coordinates (u, v) and 3D world coordinates (x, y, z) in meters.

### Evaluation Scripts

#### 1. RGB Predictions & Validation
Evaluate model on validation dataset (2D pixel accuracy):
```
python RGB-prediction-and-validation.py
```
**Outputs**:
- `OUTPUT-JSON/validation_predictions.json` - RGB predictions with pixel coordinates
- `OUTPUT-JSON/validation_evaluation_results.json` - Accuracy metrics
- `VALIDATION_ANALYSIS.md` - Detailed analysis report

**Metrics**: Precision, Recall, F1-score, mean pixel error, false positive/negative analysis

#### 2. Depth Predictions (3D)
Add 3D world coordinates to RGB predictions using depth maps:
```
python depth-prediction.py
```
**Outputs**:
- `OUTPUT-JSON/validation_predictions_3d.json` - Predictions with (x, y, z) coordinates
- `3D_EVALUATION_RESULTS.md` - Depth analysis report

**Requirements**: Depth maps in `data/1-OG/OG-framed-only` (only original frames have depth)
