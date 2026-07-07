import logging
from typing import List, Tuple, Optional, Any
import numpy as np
from ultralytics import YOLO
from src.config import PrimaryDetectorConfig, TrackerConfig

logger = logging.getLogger(__name__)

class TrackedObject:
    """Standardized representation of a tracked object, including its unique persistent ID."""
    
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], confidence: float, class_id: int):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        # Compute centroid
        self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


class YOLOTracker:
    """Uses Ultralytics YOLO's built-in tracking engine (ByteTrack/BoT-SORT)."""
    
    def __init__(self, detector_config: PrimaryDetectorConfig, tracker_config: TrackerConfig):
        self.model_path = detector_config.path
        self.conf_threshold = detector_config.conf_threshold
        self.iou_threshold = detector_config.iou_threshold
        self.classes = detector_config.classes
        self.device = detector_config.device
        self.tracker_type = tracker_config.tracker_type
        
        logger.info(f"Initializing YOLOTracker using {self.tracker_type} with detector {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)
        
        # Determine tracking config filename (bytetrack.yaml or botsort.yaml)
        self.tracker_cfg_file = "bytetrack.yaml" if self.tracker_type.lower() == "bytetrack" else "botsort.yaml"

    def track(self, frame: np.ndarray) -> List[TrackedObject]:
        """Performs tracking on a single frame and returns TrackedObject instances.
        
        Args:
            frame: OpenCV image frame (BGR format)
            
        Returns:
            List of TrackedObject instances with active tracks
        """
        # Note: persist=True runs tracking across frames in a stateful loop inside YOLO
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            tracker=self.tracker_cfg_file,
            verbose=False
        )
        
        tracked_objects = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Only include boxes that have been successfully tracked and assigned IDs
                if box.id is not None:
                    track_id = int(box.id[0])
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    tracked_objects.append(TrackedObject(
                        track_id=track_id,
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                        class_id=cls
                    ))
                    
        return tracked_objects
