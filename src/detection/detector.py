import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from ultralytics import YOLO
from src.config import PrimaryDetectorConfig, SpecialistModelConfig

logger = logging.getLogger(__name__)

class DetectionResult:
    """Standardized representation of an object detection box."""
    def __init__(self, bbox: Tuple[float, float, float, float], confidence: float, class_id: int):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


class YOLODetector:
    """Wrapper class for loading and running YOLO models on video frames."""
    
    def __init__(self, config: PrimaryDetectorConfig):
        self.model_path = config.path
        self.conf_threshold = config.conf_threshold
        self.iou_threshold = config.iou_threshold
        self.classes = config.classes
        self.device = config.device
        
        logger.info(f"Loading YOLO model from: {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Runs inference on a single frame and returns standardized detection results.
        
        Args:
            frame: OpenCV image frame (BGR format)
            
        Returns:
            List of DetectionResult objects
        """
        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False
        )
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append(DetectionResult(
                    bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    confidence=conf,
                    class_id=cls
                ))
                
        return detections
