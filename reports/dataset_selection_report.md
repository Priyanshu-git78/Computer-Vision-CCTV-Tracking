# Dataset Research & Selection Plan

**Project:** AI-Powered Intelligent Security Ecosystem for Experience Centers  
**Date:** 2026-07-07  
**Author:** AI System Architect & ML Research Engineer  

---

## 1. Overview and Objectives
To evaluate the proposed event-driven multimodel architecture against the standard continuous-inference baseline, we require standardized datasets representing realistic CCTV/showroom camera feeds. Rather than merging disparate datasets blindly, we partition our vision stack into separate task-specific datasets to feed our primary detector and modular specialist models.

This plan details candidate datasets, analyzes their feasibility, licensing, and constraints, and selects the optimal dataset for each category.

---

## 2. Selection Criteria Matrix
Candidates are ranked based on the following criteria:
1. **Showroom/CCTV Similarity**: Resolution, perspective (high-angle, overhead), lighting, and occlusion densities similar to an experience center.
2. **Annotation Quality**: Tightness of bounding boxes, temporal consistency in tracking/action clips, and class definitions.
3. **Licensing**: Academic and commercial permissions (permissible under AGPL-3.0/Creative Commons/Public Domain).
4. **Computational Feasibility**: Download size and model training footprint (must be trainable on workstation CPU/GPU under resource limits).

---

## 3. Task-Specific Candidate Datasets

### A. General Detection & Multi-Object Tracking (MOT)
*For Person Detection, Counting, Occupancy, Tracking, Intrusion, Loitering, and Dwell-Time.*

1. **TownCentre (Oxford)**
   * **Source/URL**: [Oxford TownCentre Dataset](https://www.robots.ox.ac.uk/~ActiveVision/Research/Projects/TownCentre/index.xml)
   * **Task**: Person Detection & Tracking
   * **Classes**: Person (head, body)
   * **Specs**: 1 high-resolution CCTV video clip (1080p, 5 mins, 25 FPS), ~50,000 labeled bounding boxes.
   * **Licensing**: Academic use only.
   * **Strengths**: Perfect CCTV perspective, high resolution, persistent tracking IDs.
   * **Weaknesses**: Single camera viewpoint, older dataset.
   * **Relevance**: Highly relevant for validating crossing lines and tracking consistency.

2. **MOT17 / MOT20**
   * **Source/URL**: [MOT Challenge](https://motchallenge.net/)
   * **Task**: Multi-Object Tracking
   * **Classes**: Pedestrian, Person on vehicle, Car, Bicycle, etc.
   * **Specs**: Multi-camera video sequences, 30 FPS, thousands of labeled frames.
   * **Licensing**: Non-commercial / CC BY-NC-SA.
   * **Strengths**: Gold-standard benchmarking dataset for tracking.
   * **Weaknesses**: Large download size, complex parsing.
   * **Relevance**: Crucial for tracking benchmarks, but TownCentre offers a more unified single-view testbed.

**Selection:** **TownCentre** for direct video tracking validation, and **CrowdHuman/COCO Person Class** (pretrained base model) for general person detection.

---

### B. Fire and Smoke Detection
*For the specialist Fire & Smoke alert model.*

1. **D-Fire Dataset**
   * **Source/URL**: [D-Fire Roboflow Universe](https://universe.roboflow.com/ds/fire-smoke-detection)
   * **Task**: Object Detection
   * **Classes**: `fire`, `smoke`
   * **Specs**: ~21,000 images, 640x640 resolution, YOLO annotations.
   * **Licensing**: Public Domain / CC BY 4.0.
   * **Strengths**: High quality, pre-split, includes negative/empty scenes to minimize false positives.
   * **Weaknesses**: Mainly outdoor/wildfire scenes mixed with indoor.
   * **Relevance**: Very high; enables training a binary classifier/detector that runs on-demand.

2. **FIREDSET**
   * **Source/URL**: Academic Publications
   * **Task**: Fire classification
   * **Specs**: Large volume, mainly classification-oriented (not bounding box detection).
   * **Licensing**: Restricted academic.

**Selection:** **D-Fire (Subset)**. We will download or generate a light standardized subset to train our lightweight CPU-friendly specialist model.

---

### C. Retail Theft Suspicion
*For hand-to-pocket, product concealment, or zone loitering triggers.*

1. **UCF-Crime (Shoplifting Subset)**
   * **Source/URL**: [UCF-Crime Dataset](https://www.crcv.ucf.edu/research/projects/real-world-anomaly-detection-in-surveillance-videos/)
   * **Task**: Action Recognition / Anomaly Detection
   * **Specs**: Temporal video clips (320x240, 30 FPS), labeled by video-level anomaly class.
   * **Licensing**: Research use only.
   * **Strengths**: Real-world CCTV footage of shoplifting.
   * **Weaknesses**: Low resolution, no bounding box annotations, high camera noise.
   * **Relevance**: Extremely high for validating temporal sequencing (product concealment).

**Selection:** **UCF-Crime (Shoplifting Subset)**. Used to extract temporal sequence behaviors for our rule-based hand-pocket detection.

---

### D. Violence & Fighting
*For critical event triggers.*

1. **RWF-2000 Dataset**
   * **Source/URL**: [RWF-2000 GitHub](https://github.com/mchengCOCO/RWF-2000)
   * **Task**: Action Recognition (Fight vs. Non-Fight)
   * **Specs**: 2,000 video clips (5 seconds each), 30 FPS, split into 1,000 fight and 1,000 non-fight.
   * **Licensing**: Research only.
   * **Strengths**: Balanced dataset, includes normal human interactions to minimize false alerts.
   * **Weaknesses**: Large download size (~3 GB).

**Selection:** **RWF-2000 (Subset)** or synthetic fight interaction clips for classification validation.

---

## 4. Final Selected Dataset Registry

Below is our selected ranking and registry to write to `data/metadata/dataset_registry.csv`.

| Dataset ID | Dataset Name | Task | Source | Selected | Selection Reason | Download Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GEN_01** | TownCentre Oxford | Person & Tracking | Oxford University | **Yes** | Direct overhead CCTV tracking validation. | ~150 MB |
| **FIRE_01** | D-Fire (Subset) | Fire/Smoke Detection | Roboflow | **Yes** | Standardized YOLO annotations, high quality. | ~200 MB |
| **THEFT_01** | UCF-Shoplifting | Retail Theft anomaly | UCF | **Yes** | Provides real CCTV sequences for concealment logic. | ~350 MB |
| **VIOL_01** | RWF-2000 (Subset) | Action classification | GitHub | **Yes** | Standard benchmark for fight/normal classification. | ~400 MB |

---

## 5. Implementation Path & Guardrails
> [!IMPORTANT]
> To prevent memory, VRAM, and disk exhaustion:
> 1. We will NOT download the full gigabyte-scale datasets. We will download or extract small, highly curated subsets (e.g., first 100 frames of TownCentre, 500 images of D-Fire) to conduct fast, reproducible CPU training and inference.
> 2. All download operations must check for available space (`shutil.disk_usage`) and confirm file integrity via checksums/corruption checks.
