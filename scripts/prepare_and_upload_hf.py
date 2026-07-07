# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
import shutil
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_UPLOAD_DIR = "/mnt/hdd/ultralytics/project/hf_upload"
REPO_ID = "Priyanshu-68/yolov8-cctv-tracking-models"

# Source weights
WEIGHTS_MAP = {
    "person_detector_best.pt": "/mnt/hdd/ultralytics/project/outputs/train/person_detector_run-2/weights/best.pt",
    "fire_detector_best.pt": "/mnt/hdd/ultralytics/project/outputs/train/tuned_run/weights/best.pt",
    "fire_specialist_best.pt": "/mnt/hdd/ultralytics/project/outputs/train/real_fire_run/weights/best.pt"
}

README_CONTENT = """---
language: en
license: agpl-3.0
tags:
- yolo
- yolov8
- ultralytics
- object-detection
- surveillance
- cctv
- security
---

# YOLOv8 CCTV Tracking and Intelligent Security Models

This repository contains the fine-tuned YOLOv8 vision models developed for the research project: **"AI-Powered Intelligent Security Ecosystem for Experience Centers"**.

## Repository Link
The complete codebase, configurations, and scripts for this project are available at:
*   [GitHub Repository (Computer-Vision-CCTV-Tracking)](https://github.com/Priyanshu-git78/Computer-Vision-CCTV-Tracking)

## Included Models

### 1. Person Detector (`person_detector_best.pt`)
*   **Base Architecture:** YOLOv8n (nano)
*   **Task:** General person detection and tracking.
*   **Purpose:** Served as the lightweight baseline detector to manage entry, exit, occupancy analysis, and identify regions of interest.
*   **mAP50:** 0.3733 on the validation set.

### 2. Tuned Fire Detector (`fire_detector_best.pt`)
*   **Base Architecture:** YOLOv8n (nano)
*   **Task:** Specialist fire and smoke detection.
*   **Purpose:** Fine-tuned with domain-adapted learning rates and augmentation to conditionally trigger warnings upon detecting active combustion in critical zones.
*   **mAP50:** 0.6383 | **mAP50-95:** 0.4611.

### 3. Specialist Fire Detector (`fire_specialist_best.pt`)
*   **Base Architecture:** YOLOv8s (small)
*   **Task:** Deep specialist fire detection.
*   **Purpose:** A larger model used to validate detection consistency when active triggers are identified, balancing high recall requirements.
*   **mAP50:** 0.6950 | **mAP50-95:** 0.3953.

## Dataset Curation & Citing Sources
The models were trained, validated, and compared on standard datasets:
1.  **COCO128-Person Sub-dataset**: For person detection training.
2.  **Home-fire-dataset**: For fine-tuning fire and smoke detectors.
3.  **Oxford TownCentre Dataset**: For benchmarking multiple-object tracking (MOT) performance using ByteTrack and BoT-SORT algorithms.

## How to Load and Use

You can load these models directly using the official `ultralytics` library in Python:

```python
from ultralytics import YOLO

# Load the person detector
person_model = YOLO("person_detector_best.pt")

# Perform inference on an image or video stream
results = person_model("path/to/cctv_frame.jpg")

# Show or save the results
results[0].show()
results[0].save(filename="prediction.jpg")
```

To run inference on a live RTSP stream or local video file:
```python
results = person_model("rtsp://your_camera_ip/stream", stream=True)
for r in results:
    boxes = r.boxes  # Bounding boxes
    # Process detections...
```

## Licensing
These models and their weights are licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** in accordance with the underlying Ultralytics YOLOv8 library guidelines.
"""

def prepare_and_upload():
    # 1. Create upload folder
    os.makedirs(HF_UPLOAD_DIR, exist_ok=True)
    
    # 2. Copy weight files
    for dest_name, src_path in WEIGHTS_MAP.items():
        if os.path.exists(src_path):
            dest_path = os.path.join(HF_UPLOAD_DIR, dest_name)
            logger.info(f"Copying {src_path} to {dest_path}...")
            shutil.copy2(src_path, dest_path)
        else:
            logger.warning(f"Source weight file not found: {src_path}")
            
    # 3. Write README.md
    readme_path = os.path.join(HF_UPLOAD_DIR, "README.md")
    logger.info(f"Writing Model Card to {readme_path}...")
    with open(readme_path, "w") as f:
        f.write(README_CONTENT)
        
    # 4. Create repository on Hugging Face Hub (with exist-ok)
    logger.info(f"Creating Hugging Face repository {REPO_ID} (if it does not exist)...")
    try:
        subprocess.run([
            "hf", "repos", "create", REPO_ID, "--exist-ok", "--public"
        ], check=True)
        logger.info("Repository successfully created/verified on HF Hub.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create repository: {e}")
        sys.exit(1)
        
    # 5. Upload files to Hugging Face
    logger.info("Uploading weights and Model Card to Hugging Face Hub...")
    try:
        subprocess.run([
            "hf", "upload", REPO_ID, HF_UPLOAD_DIR, ".", "--commit-message", "feat: upload YOLOv8 CCTV tracking models and README card"
        ], check=True)
        logger.info("Upload completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upload files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    prepare_and_upload()
