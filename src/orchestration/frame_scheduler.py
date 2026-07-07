import logging
from typing import Dict

logger = logging.getLogger(__name__)

class FrameScheduler:
    """Manages frame sampling frequency dynamically based on the current pipeline state."""
    
    def __init__(self, target_fps: int = 10):
        self.target_fps = target_fps
        
        # State to Frame Interval mapping (interval = process every Nth frame)
        # IDLE: Process 1 frame per second (1 in target_fps)
        # NORMAL_ACTIVITY: Process at regular target FPS (interval = 1)
        # ZONE_ACTIVITY: Process at regular target FPS (interval = 1)
        # SUSPICIOUS_ACTIVITY: Process at regular target FPS (interval = 1)
        # CRITICAL_EVENT: Process every frame (interval = 1)
        self.state_intervals: Dict[str, int] = {
            "IDLE": max(1, target_fps),
            "NORMAL_ACTIVITY": 1,
            "ZONE_ACTIVITY": 1,
            "SUSPICIOUS_ACTIVITY": 1,
            "CRITICAL_EVENT": 1
        }

    def should_process(self, processed_frame_index: int, current_state: str) -> bool:
        """Determines if the current frame should be processed or skipped.
        
        Args:
            processed_frame_index: Number of frames grabbed from the stream so far
            current_state: Current engine state (e.g. IDLE, NORMAL_ACTIVITY)
            
        Returns:
            True if the frame should be processed, False if it should be skipped
        """
        interval = self.state_intervals.get(current_state, 1)
        return (processed_frame_index - 1) % interval == 0

    def get_effective_fps(self, current_state: str) -> float:
        """Returns the effective processing frame rate for the current state."""
        interval = self.state_intervals.get(current_state, 1)
        return self.target_fps / interval
