import os
import cv2
import time
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class VideoSource:
    """Handles reading frames from a video file, webcam, or RTSP stream at a target FPS."""
    
    def __init__(self, camera_id: str, source: str, target_fps: int = 10):
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.width = 0
        self.height = 0
        self.source_fps = 0.0
        self.total_frames = 0
        self.is_file = False
        
        # Frame rate control
        self.frame_count = 0
        self.processed_frame_count = 0
        self.frame_interval = 1.0  # Default skip interval
        self.last_frame_time = 0.0
        self.running = False
        
        self._initialize_source()

    def _initialize_source(self):
        """Initializes OpenCV VideoCapture and extracts metadata."""
        self.is_synthetic = False
        
        # Check if source is a file path
        self.is_file = not (
            isinstance(self.source, str) and 
            (self.source.startswith("rtsp://") or self.source.startswith("http://") or self.source.isdigit())
        )
        
        if self.is_file and not os.path.exists(self.source):
            # Locate bus.jpg
            bus_path = "ultralytics/assets/bus.jpg"
            if not os.path.exists(bus_path):
                # Try relative to the parent of project_root
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                bus_path = os.path.join(parent_dir, "ultralytics/assets/bus.jpg")
                if not os.path.exists(bus_path):
                    bus_path = "/mnt/hdd/ultralytics/ultralytics/assets/bus.jpg"
                    
            if os.path.exists(bus_path):
                logger.warning(
                    f"[{self.camera_id}] Video source file '{self.source}' not found. "
                    f"Initializing synthetic generator using '{bus_path}' to simulate stream."
                )
                self.is_synthetic = True
                self.synthetic_base_frame = cv2.imread(bus_path)
                if self.synthetic_base_frame is not None:
                    self.height, self.width = self.synthetic_base_frame.shape[:2]
                    self.source_fps = 30.0
                    self.total_frames = 10000
                    self.frame_interval = max(1, round(self.source_fps / self.target_fps))
                    return
                else:
                    logger.error(f"[{self.camera_id}] Failed to load synthetic base image: {bus_path}")
        
        # If it's a digit string, convert to int for webcam
        source_val: Any = self.source
        if isinstance(self.source, str) and self.source.isdigit():
            source_val = int(self.source)
            
        self.cap = cv2.VideoCapture(source_val)
        if not self.cap.isOpened():
            logger.error(f"[{self.camera_id}] Failed to open source: {self.source}")
            raise ValueError(f"Could not open video source: {self.source}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.source_fps <= 0:
            self.source_fps = 30.0  # fallback default
            
        # Determine frame skip interval for files
        if self.is_file:
            self.frame_interval = max(1, round(self.source_fps / self.target_fps))
        else:
            self.frame_interval = 1
            
        logger.info(
            f"[{self.camera_id}] Initialized source. "
            f"Resolution: {self.width}x{self.height}, Source FPS: {self.source_fps:.2f}, "
            f"Target FPS: {self.target_fps}, Frame Interval: {self.frame_interval}"
        )

    def start(self):
        """Starts the video source reading loop."""
        self.running = True
        self.frame_count = 0
        self.processed_frame_count = 0
        self.last_frame_time = time.time()
        logger.info(f"[{self.camera_id}] Started stream reading.")

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Reads the next valid frame matching the target FPS.
        
        Returns:
            Tuple[success, frame, metadata]
        """
        if not self.running:
            return False, None, None

        if getattr(self, "is_synthetic", False):
            # Synthetic loop: generate frame with translation to simulate movement
            self.frame_count += 1
            self.processed_frame_count += 1
            
            # Apply slight translation using warpAffine to simulate person/camera movement
            dx = int(15 * np.sin(self.frame_count * 0.15))
            dy = int(8 * np.cos(self.frame_count * 0.15))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(self.synthetic_base_frame, M, (self.width, self.height))
            
            timestamp = self.frame_count * (1.0 / self.target_fps)
            metadata = {
                "frame_index": self.frame_count,
                "processed_index": self.processed_frame_count,
                "timestamp": timestamp,
                "camera_id": self.camera_id,
                "width": self.width,
                "height": self.height,
            }
            # Simulate processing frame interval delay
            time.sleep(1.0 / self.target_fps)
            return True, frame, metadata

        if self.cap is None:
            return False, None, None

        if self.is_file:
            # File mode: skip frames to match the target FPS
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info(f"[{self.camera_id}] End of video file reached.")
                    return False, None, None
                
                self.frame_count += 1
                
                # Check if this frame should be processed based on skip interval
                if (self.frame_count - 1) % self.frame_interval == 0:
                    self.processed_frame_count += 1
                    timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    metadata = {
                        "frame_index": self.frame_count,
                        "processed_index": self.processed_frame_count,
                        "timestamp": timestamp,
                        "camera_id": self.camera_id,
                        "width": self.width,
                        "height": self.height,
                    }
                    return True, frame, metadata
        else:
            # Stream mode: read latest frame or rate limit to target_fps
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            target_period = 1.0 / self.target_fps
            
            # If we are reading too fast, sleep/wait
            if elapsed < target_period:
                time.sleep(target_period - elapsed)
                
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"[{self.camera_id}] Failed to grab frame from stream.")
                return False, None, None
                
            self.frame_count += 1
            self.processed_frame_count += 1
            self.last_frame_time = time.time()
            
            metadata = {
                "frame_index": self.frame_count,
                "processed_index": self.processed_frame_count,
                "timestamp": self.last_frame_time,
                "camera_id": self.camera_id,
                "width": self.width,
                "height": self.height,
            }
            return True, frame, metadata

    def stop(self):
        """Releases the video source resources."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info(f"[{self.camera_id}] Stopped stream reading and released source.")
