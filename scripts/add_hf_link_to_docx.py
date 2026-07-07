# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
import docx
import logging

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOCX_PATHS = [
    "/mnt/hdd/ultralytics/Computer_Vision_CCTV_Tracking_Project_Report.docx",
    "/mnt/hdd/ultralytics/project/Computer_Vision_CCTV_Tracking_Project_Report.docx"
]

def add_hf_link():
    for doc_path in DOCX_PATHS:
        if not os.path.exists(doc_path):
            logger.warning(f"File not found: {doc_path}")
            continue

        logger.info(f"Adding Hugging Face link in: {doc_path}")
        doc = docx.Document(doc_path)
        
        # Check paragraph 447 (where repository link is)
        p447 = doc.paragraphs[447]
        if "Repository Link" in p447.text and "Hugging Face" not in p447.text:
            p447.text = (
                "Repository Link: https://github.com/Priyanshu-git78/Computer-Vision-CCTV-Tracking\n"
                "Hugging Face Model Hub: https://huggingface.co/Priyanshu-68/yolov8-cctv-tracking-models"
            )
            logger.info("Successfully added Hugging Face link.")

        doc.save(doc_path)
        logger.info(f"Saved document to: {doc_path}")

if __name__ == "__main__":
    add_hf_link()
