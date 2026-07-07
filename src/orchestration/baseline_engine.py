import os
import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from src.config import SystemConfig, CameraConfig
from src.ingestion.video_source import VideoSource
from src.tracking.tracker import YOLOTracker, TrackedObject
from src.tracking.track_manager import TrackManager
from src.analytics.line_crossing import LineCrossingDetector
from src.analytics.occupancy import OccupancyTracker
from src.analytics.dwell_time import ZoneAnalyticsManager
from src.monitoring.performance_monitor import PerformanceMonitor
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class BaselineEngine:
    """Runs the baseline pipeline: fixed FPS and continuous execution of all models."""
    
    def __init__(self, system_config: SystemConfig, camera_id: str):
        self.config = system_config
        self.camera_id = camera_id
        
        # Load camera configuration
        self.cam_config = self.config.get_camera_config(camera_id)
        if not self.cam_config:
            raise ValueError(f"Camera ID {camera_id} not found in configuration.")
            
        # Get zones and lines
        self.zones_config = self.config.get_camera_zones(camera_id)
        
        # Ingestion
        self.video_source = VideoSource(
            camera_id=self.camera_id,
            source=self.cam_config.source,
            target_fps=self.cam_config.target_fps
        )
        
        # Tracking
        self.tracker = YOLOTracker(self.config.models.primary_detector, self.config.models.tracker)
        self.track_manager = TrackManager()
        
        # Analytics
        self.line_crossing_detectors = [
            LineCrossingDetector(line) for line in self.zones_config.lines
        ]
        self.occupancy_tracker = OccupancyTracker()
        self.zone_analytics = ZoneAnalyticsManager(
            zones=self.zones_config.zones,
            loitering_threshold_seconds=self.config.events.loitering.dwell_time_threshold_seconds
        )
        
        # Performance Monitoring
        self.perf_monitor = PerformanceMonitor(
            camera_id=self.camera_id,
            monitor_interval_seconds=self.config.benchmark.monitor_interval_seconds
        )
        
        # Specialist Models - loaded in baseline for continuous execution
        self.specialists: Dict[str, YOLO] = {}
        self._load_specialist_models()

    def _load_specialist_models(self):
        """Loads enabled specialist models. Falls back to primary detector to simulate workload if weights missing."""
        for name, spec_cfg in self.config.models.specialist_models.items():
            if spec_cfg.enabled:
                try:
                    # Check if weights file exists, otherwise fallback to primary path for simulation
                    path_to_load = spec_cfg.path
                    if not os.path.exists(path_to_load):
                        logger.warning(
                            f"[Baseline] Specialist weights not found at {path_to_load}. "
                            f"Falling back to primary detector {self.config.models.primary_detector.path} to simulate GPU workload."
                        )
                        path_to_load = self.config.models.primary_detector.path
                        
                    self.specialists[name] = YOLO(path_to_load)
                    logger.info(f"[Baseline] Loaded specialist model '{name}' from {path_to_load}")
                except Exception as e:
                    logger.error(f"[Baseline] Failed to load specialist model '{name}': {e}")

    def run(self, max_frames: Optional[int] = None):
        """Runs the continuous baseline inference loop."""
        logger.info(f"[Baseline] Starting execution on camera {self.camera_id}")
        self.video_source.start()
        self.perf_monitor.start_time = time.time()
        
        processed_count = 0
        try:
            while self.video_source.running:
                # 1. Read Frame
                success, frame, metadata = self.video_source.read()
                if not success or frame is None or metadata is None:
                    break
                    
                self.perf_monitor.record_frame_received()
                t_start = time.time()
                
                # 2. Run Primary Detector + Tracker
                t_infer_start = time.time()
                tracked_objects = self.tracker.track(frame)
                t_infer_end = time.time()
                
                self.perf_monitor.record_detector_execution()
                self.perf_monitor.record_tracker_update()
                
                # Update trajectory history
                self.track_manager.update(tracked_objects, metadata["timestamp"])
                active_tracks = self.track_manager.get_active_tracks(metadata["timestamp"])
                
                # 3. Continuous Execution of Specialist Models (Baseline behavior)
                # In baseline, we execute all loaded specialists on EVERY frame to measure workload
                for name, model in self.specialists.items():
                    # Run inference on frame to simulate VRAM and compute load
                    _ = model.predict(source=frame, device=self.config.models.primary_detector.device, verbose=False)
                    self.perf_monitor.record_specialist_execution(name)
                
                # 4. Run Analytics Modules
                # Line Crossing
                crossings = []
                for lcd in self.line_crossing_detectors:
                    crossings.extend(lcd.update(active_tracks, metadata["timestamp"]))
                self.occupancy_tracker.update_from_crossing_events(crossings)
                
                # Zone Analytics (Dwell Time / Loitering)
                zone_events, security_events = self.zone_analytics.update(active_tracks, metadata["timestamp"])
                
                # Record events count
                for _ in range(len(crossings) + len(security_events)):
                    self.perf_monitor.record_event_detected()
                    
                # 5. Record performance metrics
                t_end = time.time()
                self.perf_monitor.record_frame_processed()
                self.perf_monitor.record_latency(
                    inference_time_sec=t_infer_end - t_infer_start,
                    e2e_time_sec=t_end - t_start
                )
                
                self.perf_monitor.update()
                
                # Clean up tracks history periodically
                self.track_manager.clean_stale_tracks(metadata["timestamp"])
                
                processed_count += 1
                if max_frames and processed_count >= max_frames:
                    logger.info(f"[Baseline] Reached max frames threshold: {max_frames}")
                    break
                    
        except KeyboardInterrupt:
            logger.info("[Baseline] Execution interrupted by user.")
        finally:
            self.video_source.stop()
            logger.info(f"[Baseline] Finished execution on camera {self.camera_id}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Returns the execution performance metrics summary."""
        return self.perf_monitor.get_summary()

    def save_results(self, output_dir: str):
        """Saves execution history and summary."""
        self.perf_monitor.save_results(output_dir, prefix="baseline")
