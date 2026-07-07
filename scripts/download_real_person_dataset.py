# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
import zipfile
import urllib.request
import shutil
import time
import csv
import logging
from ultralytics import YOLO

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(project_root, "data")
COCO128_DIR = os.path.join(DATA_DIR, "coco128_dataset")
ZIP_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
ZIP_PATH = os.path.join(DATA_DIR, "coco128.zip")
METRICS_CSV = os.path.join(project_root, "results/person_model_metrics.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)

def download_and_extract():
    # 1. Download ZIP file
    if not os.path.exists(ZIP_PATH):
        logger.info(f"Downloading COCO128 dataset from {ZIP_URL}...")
        t_start = time.time()
        try:
            req = urllib.request.Request(ZIP_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response, open(ZIP_PATH, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            logger.info(f"Download completed in {time.time() - t_start:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            sys.exit(1)
    else:
        logger.info("COCO128 ZIP already exists locally. Skipping download.")

    # 2. Extract ZIP file
    logger.info("Extracting ZIP file...")
    # Clean previous extraction if exists
    if os.path.exists(COCO128_DIR):
        shutil.rmtree(COCO128_DIR)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(COCO128_DIR)
    logger.info("Extraction completed.")

    # 3. Filter label files to only keep class 0 (person)
    extracted_path = os.path.join(COCO128_DIR, "coco128")
    labels_path = os.path.join(extracted_path, "labels/train2017")
    images_path = os.path.join(extracted_path, "images/train2017")

    logger.info("Filtering labels for class 0 (person) only...")
    person_image_count = 0
    
    # List of images to keep (only those that actually contain a person)
    images_to_keep = []
    
    for filename in os.listdir(labels_path):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(labels_path, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Keep only lines representing class 0
        person_lines = [line for line in lines if line.strip().startswith("0 ")]
        
        base_name, _ = os.path.splitext(filename)
        img_filename = base_name + ".jpg"
        img_path = os.path.join(images_path, img_filename)
        
        if len(person_lines) > 0 and os.path.exists(img_path):
            # Save the filtered annotations
            with open(file_path, "w") as f:
                f.writelines(person_lines)
            images_to_keep.append(img_filename)
            person_image_count += 1
        else:
            # Delete annotations and image files that don't have persons
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(img_path):
                os.remove(img_path)

    logger.info(f"Filtering completed. Kept {person_image_count} images containing 'person'.")

    # 4. Create dataset.yaml configured for person-only (class ID 0)
    dataset_yaml = os.path.join(COCO128_DIR, "dataset.yaml")
    
    yaml_content = f"""path: {extracted_path}
train: images/train2017
val: images/train2017

names:
  0: person
"""
    with open(dataset_yaml, "w") as f:
        f.write(yaml_content)
    logger.info(f"Created dataset YAML at {dataset_yaml}")
    return dataset_yaml

def train_model(dataset_yaml: str):
    logger.info("Loading pretrained YOLOv8n weights...")
    model = YOLO("yolov8n.pt")
    
    # Train for 3 epochs on CPU with a small batch size for speed and safety
    logger.info("Starting training on COCO128 person dataset...")
    t_start = time.time()
    results = model.train(
        data=dataset_yaml,
        epochs=3,
        batch=8,
        device="cpu",
        project=os.path.join(project_root, "outputs/train"),
        name="person_detector_run",
        verbose=True
    )
    t_end = time.time()
    duration = t_end - t_start
    logger.info(f"Training finished in {duration:.2f}s.")

    # Evaluate the model on validation set to collect metrics
    logger.info("Evaluating model metrics...")
    metrics = model.val(device="cpu", verbose=False)
    
    precision = metrics.results_dict.get("metrics/precision(B)", 0.0)
    recall = metrics.results_dict.get("metrics/recall(B)", 0.0)
    map50 = metrics.results_dict.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)

    # Save metrics to CSV
    logger.info(f"Saving metrics to {METRICS_CSV}...")
    headers = ["timestamp", "dataset", "epochs", "duration_seconds", "precision", "recall", "mAP50", "mAP50_95"]
    file_exists = os.path.exists(METRICS_CSV)
    
    with open(METRICS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "COCO128_Person",
            3,
            f"{duration:.2f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{map50:.4f}",
            f"{map50_95:.4f}"
        ])
    logger.info("Metrics saved successfully.")

if __name__ == "__main__":
    yaml_path = download_and_extract()
    train_model(yaml_path)
