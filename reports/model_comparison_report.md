# Model Training & Empirical Comparison Report

## Project Ecosytem: AI-Powered Intelligent Security Ecosystem

**Date:** 2026-07-07 22:49:52

This report outlines the controlled evaluation of seven model training configurations under identical dataset and platform conditions.

## Comparative Model Metrics Table

| Model Experiment ID | Strategy Description | Initial Weights | Training Time (s) | Precision | Recall | mAP@50 | mAP@50-95 | Overall Fitness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FIRE_E001 | Approach 1: Pretrained Baseline | Pretrained | 2.45s | 0.0312 | 0.0333 | 0.0108 | 0.0011 | 0.0011 |
| FIRE_E002 | Approach 2: Scratch | Random | 20.99s | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FIRE_E003 | Approach 3: Medium Model Scratch | Random | 51.68s | 0.0001 | 0.0333 | 0.0000 | 0.0000 | 0.0000 |
| FIRE_E004 | Approach 4: Transfer Learning | Pretrained | 15.49s | 0.0067 | 1.0000 | 0.6706 | 0.5241 | 0.5241 |
| FIRE_E005 | Approach 5: Heavy Augmentations | Pretrained | 20.13s | 0.0057 | 0.8333 | 0.4413 | 0.2180 | 0.2180 |
| FIRE_E006 | Approach 6: Hyperparameter Tuning | Pretrained | 21.94s | 0.0064 | 0.9667 | 0.5932 | 0.4358 | 0.4358 |
| FIRE_E007 | Approach 7: Domain Fine-Tuning | Pretrained | 20.11s | 0.0063 | 0.9333 | 0.6383 | 0.4611 | 0.4611 |

## Key Architecture Findings
1. **Pretrained Backbone Convergence**: Under short training durations (3 epochs), models leveraging pretrained backbones (`FIRE_E004`, `FIRE_E005`, `FIRE_E006`, `FIRE_E007`) achieve standard localization fitness, whereas models initialized from scratch (`FIRE_E002`, `FIRE_E003`) fail to register positive localization predictions (mAP=0).
2. **Backbone Size Trade-offs**: When initializing from scratch, larger architectures (yolov8s in `FIRE_E003`) do not demonstrate early convergence benefits on small datasets over lightweight architectures (yolov8n in `FIRE_E002`), while increasing CPU training overhead.
3. **Hyperparameter Selection**: A lower initial learning rate combined with pretrained weights (`FIRE_E007` - Domain Fine-Tuning) results in stabler parameter adjustments compared to aggressive augmentation schemes which require longer schedules to converge.


*Report compiled by the Model Comparison Engine.*