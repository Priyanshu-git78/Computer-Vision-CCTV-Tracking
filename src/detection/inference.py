import cv2
import numpy as np
from typing import List, Tuple
from src.detection.detector import DetectionResult

def draw_detections(
    frame: np.ndarray, 
    detections: List[DetectionResult], 
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_labels: bool = True,
    class_names: List[str] = None
) -> np.ndarray:
    """Draws bounding boxes and labels for the given detections on the frame."""
    annotated_frame = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        if show_labels:
            class_name = class_names[det.class_id] if class_names and det.class_id < len(class_names) else f"Class {det.class_id}"
            label = f"{class_name}: {det.confidence:.2f}"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - 20), 
                (x1 + w, y1), 
                color, 
                -1
            )
            cv2.putText(
                annotated_frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
    return annotated_frame
