import os
import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from src.config import SystemConfig, CameraConfig
from src.ingestion.video_source import VideoSource
from src.tracking.tracker import YOLOTracker, TrackedObject
from src.tracking.track_manager import TrackManager
from src.analytics.line_crossing import LineCrossingDetector
from src.analytics.occupancy import OccupancyTracker
from src.analytics.dwell_time import ZoneAnalyticsManager
from src.monitoring.performance_monitor import PerformanceMonitor
from src.orchestration.model_registry import ModelRegistry
from src.orchestration.frame_scheduler import FrameScheduler
from src.orchestration.trigger_manager import TriggerManager

logger = logging.getLogger(__name__)

class AdaptiveEngine:
    """Runs the adaptive, event-driven pipeline which optimizes compute dynamically."""
    
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
        
        # Adaptive Orchestration
        self.model_registry = ModelRegistry(self.config)
        self.frame_scheduler = FrameScheduler(target_fps=self.cam_config.target_fps)
        self.trigger_manager = TriggerManager(self.config.events)
        
        # Evidence Clip Buffer (circular queue of frames to record preceding context)
        self.frame_buffer: List[np.ndarray] = []
        self.max_buffer_frames = self.cam_config.target_fps * 5  # 5 seconds buffer
        self.recording_active = False
        self.recorded_frames = []

    def _determine_specialists_to_run(self) -> List[str]:
        """Rules defining which specialist models should be active based on current state."""
        state = self.trigger_manager.state
        specialists = []
        
        # Only run specialist models in SUSPICIOUS_ACTIVITY or CRITICAL_EVENT states
        if state in ["SUSPICIOUS_ACTIVITY", "CRITICAL_EVENT"]:
            # If camera role is security, run fire/smoke detection
            if self.cam_config.role in ["security", "entrance"]:
                specialists.append("fire_smoke")
                
            # If we observe suspicious behavior, run action/theft models
            specialists.append("action_recognition")
            specialists.append("theft")
            
        return specialists

    def _save_evidence_clip(self, output_dir: str):
        """Saves cached frames to an MP4 video file when a critical event is verified."""
        if not self.recorded_frames:
            return
            
        os.makedirs(os.path.join(output_dir, "clips"), exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evidence_{self.camera_id}_{timestamp_str}.mp4"
        filepath = os.path.join(output_dir, "clips", filename)
        
        h, w = self.recorded_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.cam_config.target_fps, (w, h))
        
        for f in self.recorded_frames:
            out.write(f)
        out.release()
        logger.warning(f"[Evidence Clip] Saved critical event video evidence to {filepath}")
        self.recorded_frames = []

    def run(self, max_frames: Optional[int] = None):
        """Runs the adaptive frame rate and dynamic specialist model execution loop."""
        logger.info(f"[Adaptive] Starting execution on camera {self.camera_id}")
        self.video_source.start()
        self.perf_monitor.start_time = time.time()
        
        processed_count = 0
        grabbed_count = 0
        try:
            while self.video_source.running:
                # 1. Check frame scheduler based on current state
                grabbed_count += 1
                should_proc = self.frame_scheduler.should_process(grabbed_count, self.trigger_manager.state)
                
                if not should_proc:
                    # Skip frame: read and drop it from CV2 decoder to keep stream real-time
                    # (For file sources, we can skip frames by simply reading without processing)
                    if self.video_source.is_file:
                        # Grab next frame but do not run pipeline
                        ret = self.video_source.cap.grab()
                        if not ret:
                            break
                        self.video_source.frame_count += 1
                        self.perf_monitor.record_frame_skipped()
                    else:
                        success, frame, metadata = self.video_source.read()
                        if not success:
                            break
                        self.perf_monitor.record_frame_skipped()
                    continue
                
                # 2. Read Frame
                success, frame, metadata = self.video_source.read()
                if not success or frame is None or metadata is None:
                    break
                    
                self.perf_monitor.record_frame_received()
                t_start = time.time()
                
                # Update rolling frame buffer for evidence recording
                self.frame_buffer.append(frame.copy())
                if len(self.frame_buffer) > self.max_buffer_frames:
                    self.frame_buffer.pop(0)
                
                # 3. Run Primary Detector + Tracker
                t_infer_start = time.time()
                tracked_objects = self.tracker.track(frame)
                t_infer_end = time.time()
                
                self.perf_monitor.record_detector_execution()
                self.perf_monitor.record_tracker_update()
                
                # Update trajectory history
                self.track_manager.update(tracked_objects, metadata["timestamp"])
                active_tracks = self.track_manager.get_active_tracks(metadata["timestamp"])
                
                # 4. Run Analytics Modules
                # Line Crossing
                crossings = []
                for lcd in self.line_crossing_detectors:
                    crossings.extend(lcd.update(active_tracks, metadata["timestamp"]))
                self.occupancy_tracker.update_from_crossing_events(crossings)
                
                # Zone Analytics (Dwell Time / Loitering)
                zone_events, security_events = self.zone_analytics.update(active_tracks, metadata["timestamp"])
                
                # Compile current visits details to evaluate triggers
                # Key: (track_id, zone_name), Value: visit object
                active_visits = self.zone_analytics.active_visits
                
                # 5. Dynamic Specialist Execution
                specialist_dets = {}
                specialist_to_run = self._determine_specialists_to_run()
                
                # Deactivate models not in list to free up VRAM
                for loaded_model in list(self.model_registry.models.keys()):
                    if loaded_model not in specialist_to_run:
                        self.model_registry.release_model(loaded_model)
                
                # Run active specialists
                for model_name in specialist_to_run:
                    model = self.model_registry.get_model(model_name)
                    if model:
                        # Run inference
                        preds = model.predict(source=frame, device=self.config.models.primary_detector.device, verbose=False)
                        self.perf_monitor.record_specialist_execution(model_name)
                        
                        # Store prediction results for TriggerManager evaluation
                        if len(preds) > 0 and preds[0].boxes is not None:
                            specialist_dets[model_name] = preds[0].boxes.xyxy.tolist()
                        else:
                            specialist_dets[model_name] = []
                            
                # 6. State Evaluation & Transitions
                prev_state = self.trigger_manager.state
                new_state = self.trigger_manager.evaluate_state(
                    active_tracks=active_tracks,
                    active_visits=active_visits,
                    specialist_detections=specialist_dets,
                    timestamp=metadata["timestamp"]
                )
                
                # Record events count
                for _ in range(len(crossings) + len(security_events)):
                    self.perf_monitor.record_event_detected()
                
                # 7. Evidence Recording and Verification
                if new_state == "CRITICAL_EVENT":
                    if not self.recording_active:
                        self.recording_active = True
                        # Prepend the buffer to recording
                        self.recorded_frames = list(self.frame_buffer)
                    else:
                        self.recorded_frames.append(frame.copy())
                else:
                    if self.recording_active:
                        # Save the evidence clip once we transition out of CRITICAL_EVENT
                        self.recording_active = False
                        self._save_evidence_clip(self.config.benchmark.output_dir)
                
                # 8. Record performance metrics
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
                    logger.info(f"[Adaptive] Reached max frames threshold: {max_frames}")
                    break
                    
        except KeyboardInterrupt:
            logger.info("[Adaptive] Execution interrupted by user.")
        finally:
            self.video_source.stop()
            # If we were recording, close it out
            if self.recording_active:
                self._save_evidence_clip(self.config.benchmark.output_dir)
            logger.info(f"[Adaptive] Finished execution on camera {self.camera_id}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Returns the execution performance metrics summary."""
        return self.perf_monitor.get_summary()

    def save_results(self, output_dir: str):
        """Saves execution history and summary."""
        self.perf_monitor.save_results(output_dir, prefix="adaptive")
