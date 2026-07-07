import logging
from typing import Dict, Optional, List
from src.config import CameraConfig
from src.ingestion.video_source import VideoSource

logger = logging.getLogger(__name__)

class StreamManager:
    """Manages creation, lifecycle, and retrieval of multiple VideoSource streams."""
    
    def __init__(self):
        self.streams: Dict[str, VideoSource] = {}

    def add_stream(self, config: CameraConfig) -> VideoSource:
        """Creates and registers a VideoSource for the given configuration."""
        if not config.enabled:
            logger.info(f"[{config.camera_id}] Skipping disabled stream.")
            raise ValueError(f"Stream {config.camera_id} is disabled in configuration.")
            
        if config.camera_id in self.streams:
            logger.warning(f"[{config.camera_id}] Stream already exists. Stopping existing stream.")
            self.stop_stream(config.camera_id)
            
        stream = VideoSource(
            camera_id=config.camera_id,
            source=config.source,
            target_fps=config.target_fps
        )
        self.streams[config.camera_id] = stream
        return stream

    def get_stream(self, camera_id: str) -> Optional[VideoSource]:
        """Gets a registered stream by its ID."""
        return self.streams.get(camera_id)

    def start_stream(self, camera_id: str):
        """Starts a registered stream by its ID."""
        stream = self.get_stream(camera_id)
        if stream:
            stream.start()
        else:
            logger.error(f"[{camera_id}] Cannot start. Stream not registered.")

    def stop_stream(self, camera_id: str):
        """Stops a registered stream by its ID and removes it."""
        stream = self.streams.pop(camera_id, None)
        if stream:
            stream.stop()

    def start_all(self):
        """Starts all registered streams."""
        for cam_id in self.streams:
            self.start_stream(cam_id)

    def stop_all(self):
        """Stops all registered streams."""
        # Copy keys to avoid mutation issues during iteration
        for cam_id in list(self.streams.keys()):
            self.stop_stream(cam_id)
        logger.info("All streams stopped.")
