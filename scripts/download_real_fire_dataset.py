# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
import zipfile
import urllib.request
import shutil
import time
import logging
from typing import List
from ultralytics import YOLO

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(project_root, "data")
REAL_FIRE_DIR = os.path.join(DATA_DIR, "real_fire_dataset")
ZIP_URL = "https://github.com/PengBo0/Home-fire-dataset/releases/download/v1.0.0/val.zip"
ZIP_PATH = os.path.join(DATA_DIR, "val.zip")

os.makedirs(DATA_DIR, exist_ok=True)

def download_and_extract():
    # 1. Download ZIP file
    if not os.path.exists(ZIP_PATH):
        logger.info(f"Downloading real fire dataset from {ZIP_URL}...")
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
        logger.info("Dataset ZIP already exists locally. Skipping download.")

    # 2. Extract ZIP file
    logger.info("Extracting ZIP file...")
    extract_temp = os.path.join(DATA_DIR, "temp_extract")
    os.makedirs(extract_temp, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(extract_temp)
    logger.info("Extraction completed.")

    # 3. Reorganize into standard train/val splits
    # Home-fire-dataset has folders like images/ and labels/ in the val.zip
    # Let's inspect the temp extract directory structure
    logger.info("Reorganizing files into train/val splits...")
    
    images_src = None
    labels_src = None
    
    # Walk to find images and labels directories
    for root, dirs, files in os.walk(extract_temp):
        if "images" in dirs:
            images_src = os.path.join(root, "images")
        if "labels" in dirs:
            labels_src = os.path.join(root, "labels")

    if not images_src or not labels_src:
        # Fallback if the ZIP has flat structures or different folder names
        logger.error("Could not find 'images' or 'labels' directories inside the extracted zip.")
        sys.exit(1)

    # Setup target directories
    target_img_train = os.path.join(REAL_FIRE_DIR, "images/train")
    target_img_val = os.path.join(REAL_FIRE_DIR, "images/val")
    target_lbl_train = os.path.join(REAL_FIRE_DIR, "labels/train")
    target_lbl_val = os.path.join(REAL_FIRE_DIR, "labels/val")

    os.makedirs(target_img_train, exist_ok=True)
    os.makedirs(target_img_val, exist_ok=True)
    os.makedirs(target_lbl_train, exist_ok=True)
    os.makedirs(target_lbl_val, exist_ok=True)

    # Get all image files
    all_images = [f for f in os.listdir(images_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_images.sort()
    
    # Split: 80% train, 20% val
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    def copy_split(images_list: List[str], img_dest: str, lbl_dest: str):
        for img_name in images_list:
            base_name, _ = os.path.splitext(img_name)
            lbl_name = base_name + ".txt"
            
            src_img_path = os.path.join(images_src, img_name)
            src_lbl_path = os.path.join(labels_src, lbl_name)
            
            if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
                shutil.copy2(src_img_path, os.path.join(img_dest, img_name))
                shutil.copy2(src_lbl_path, os.path.join(lbl_dest, lbl_name))

    logger.info(f"Copying {len(train_images)} images to train split...")
    copy_split(train_images, target_img_train, target_lbl_train)
    
    logger.info(f"Copying {len(val_images)} images to val split...")
    copy_split(val_images, target_img_val, target_lbl_val)

    # Clean up temp extraction folder
    shutil.rmtree(extract_temp)
    logger.info("Reorganization finished and temporary files cleaned.")

    # 4. Create dataset.yaml
    dataset_yaml = os.path.join(REAL_FIRE_DIR, "dataset.yaml")
    yaml_content = f"""path: {REAL_FIRE_DIR}
train: images/train
val: images/val

names:
  0: fire
"""
    with open(dataset_yaml, "w") as f:
        f.write(yaml_content)
    logger.info(f"Created dataset YAML at {dataset_yaml}")
    return dataset_yaml

def train_model(dataset_yaml: str):
    logger.info("Loading pretrained YOLOv8n weights...")
    model = YOLO("yolov8n.pt")
    
    # Train for 3 epochs on CPU with a small batch size for speed and safety
    logger.info("Starting training on real fire dataset...")
    t_start = time.time()
    model.train(
        data=dataset_yaml,
        epochs=3,
        batch=8,
        device="cpu",
        project=os.path.join(project_root, "outputs/train"),
        name="real_fire_run",
        verbose=True
    )
    t_end = time.time()
    logger.info(f"Training finished in {t_end - t_start:.2f}s.")

if __name__ == "__main__":
    yaml_path = download_and_extract()
    train_model(yaml_path)
