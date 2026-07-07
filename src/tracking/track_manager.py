import logging
from typing import Dict, List, Tuple, Any, Optional
from src.tracking.tracker import TrackedObject

logger = logging.getLogger(__name__)

class TrackHistory:
    """Stores the history of a single tracked object."""
    
    def __init__(self, track_id: int, class_id: int, first_seen: float):
        self.track_id = track_id
        self.class_id = class_id
        self.first_seen = first_seen
        self.last_seen = first_seen
        
        self.centroids: List[Tuple[float, float]] = []
        self.bboxes: List[Tuple[float, float, float, float]] = []
        self.timestamps: List[float] = []
        self.confidences: List[float] = []

    def update(self, bbox: Tuple[float, float, float, float], confidence: float, timestamp: float):
        """Appends a new state observation to this track's history."""
        self.last_seen = timestamp
        self.bboxes.append(bbox)
        self.confidences.append(confidence)
        self.timestamps.append(timestamp)
        
        # Calculate and record centroid
        centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.centroids.append(centroid)


class TrackManager:
    """Tracks state and trajectory history for multiple active tracks."""
    
    def __init__(self, max_history_points: int = 100):
        self.tracks: Dict[int, TrackHistory] = {}
        self.max_history_points = max_history_points

    def update(self, tracked_objects: List[TrackedObject], timestamp: float):
        """Updates internal track history with new object coordinates from the tracker.
        
        Args:
            tracked_objects: List of TrackedObject from the tracker
            timestamp: Current frame timestamp in seconds
        """
        for obj in tracked_objects:
            if obj.track_id not in self.tracks:
                self.tracks[obj.track_id] = TrackHistory(
                    track_id=obj.track_id,
                    class_id=obj.class_id,
                    first_seen=timestamp
                )
            
            track = self.tracks[obj.track_id]
            track.update(obj.bbox, obj.confidence, timestamp)
            
            # Keep trajectory history bounded to prevent memory bloat
            if len(track.centroids) > self.max_history_points:
                track.centroids.pop(0)
                track.bboxes.pop(0)
                track.timestamps.pop(0)
                track.confidences.pop(0)

    def clean_stale_tracks(self, current_time: float, max_age_seconds: float = 10.0):
        """Removes tracks that have not been observed recently.
        
        Args:
            current_time: Current timestamp in seconds
            max_age_seconds: Maximum time since last observation before removing track
        """
        stale_ids = []
        for track_id, track in self.tracks.items():
            if current_time - track.last_seen > max_age_seconds:
                stale_ids.append(track_id)
                
        for track_id in stale_ids:
            self.tracks.pop(track_id)
            
        if stale_ids:
            logger.debug(f"Pruned {len(stale_ids)} stale tracks: {stale_ids}")

    def get_track(self, track_id: int) -> Optional[TrackHistory]:
        """Retrieves history for a specific track ID."""
        return self.tracks.get(track_id)

    def get_active_tracks(self, current_time: float, max_age_seconds: float = 2.0) -> Dict[int, TrackHistory]:
        """Returns tracks that have been updated recently."""
        active = {}
        for track_id, track in self.tracks.items():
            if current_time - track.last_seen <= max_age_seconds:
                active[track_id] = track
        return active
