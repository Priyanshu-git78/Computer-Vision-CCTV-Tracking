# Detailed Model Performance & Test Analytics Report

## Project Ecosystem: AI-Powered Intelligent Security Ecosystem

This report contains the test set validation results, inference speed statistics, and error analysis for both the custom Specialist Fire Detector and the General Person Detector.

### 1. Overall Model Performance Matrix
| Model Identifier | Dataset | Precision | Recall | mAP@50 | mAP@50-95 | Fitness Value | Evaluation Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Specialist Fire Detector** | Home-fire-dataset (Val Split) | 0.7033 | 0.6714 | 0.6950 | 0.3953 | 0.3953 | 5.46s |
| **General Person Detector** | COCO128 (Person Split) | 0.7780 | 0.3307 | 0.3733 | 0.2954 | 0.2954 | 2.63s |

### 2. Inference Speed & Latency Breakdown (CPU)
| Model Identifier | Preprocess Latency | Inference Latency | Postprocess Latency | Total Latency (ms) | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Specialist Fire Detector** | 0.39 ms | 23.07 ms | 0.84 ms | 24.30 ms | 41.15 FPS |
| **General Person Detector** | 0.85 ms | 22.97 ms | 1.29 ms | 25.11 ms | 39.83 FPS |

### 3. Detailed Error & Confusion Matrix Analysis
#### A. Specialist Fire Detector
- **True Positives (TP)**: The model exhibits high sensitivity to active flames. Bright, localized fire pixels are mapped correctly with minimal centroid offset.
- **False Positives (FP)**: Minor confusion occurs under strong, warm-colored indoor light sources and reflections on glossy showroom panels.
- **False Negatives (FN)**: Thin, dispersed smoke streams without a visible flame core are occasionally missed at early stages. This validates our design choice of using fire and smoke as separate triggerable sub-classes.

#### B. General Person Detector
- **True Positives (TP)**: Bounding box overlaps are highly consistent for upright walking paths and normal poses.
- **False Positives (FP)**: Handbags, backpacks, and chairs are occasionally misclassified as persons in cluttered background arrangements.
- **False Negatives (FN)**: Heavy occlusions (e.g. person partially hidden behind columns or furniture) lead to bounding box dropout.

### 4. Saved Visualizations
Sample visual predictions displaying predicted bounding boxes and class confidence scores have been exported to:
- `project/outputs/test_predictions/`