import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.tracking.track_manager import TrackHistory
from src.config import AbandonedObjectConfig

logger = logging.getLogger(__name__)

class StationaryObjectState:
    """Tracks the state, duration, and owner association of a potential abandoned object."""
    
    def __init__(self, obj_id: int, bbox: Tuple[float, float, float, float], class_id: int, timestamp: float):
        self.obj_id = obj_id
        self.bbox = bbox
        self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.class_id = class_id
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.stationary_since = timestamp
        
        # Owner association
        self.owner_track_id: Optional[int] = None
        self.alert_triggered = False

    def update_position(self, bbox: Tuple[float, float, float, float], timestamp: float):
        """Updates position and records the last seen time."""
        self.bbox = bbox
        self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.last_seen = timestamp


class AbandonedObjectDetector:
    """Detects unattended luggage, backpacks, and packages by analyzing stationary durations and owner proximity."""
    
    def __init__(self, config: AbandonedObjectConfig):
        self.config = config
        
        # Track stationary items. Key: track_id of the object, Value: StationaryObjectState
        self.tracked_objects: Dict[int, StationaryObjectState] = {}
        
        # Helper counter for object IDs (if objects are not tracked by main tracker, 
        # but here we assume the primary detector/tracker also tracks these class IDs)
        # Note: If tracker only tracks people, we can associate object detections from detector directly.
        # We will support both tracker-tracked objects and raw detections.

    def _calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculates Euclidean distance between two points."""
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

    def update(
        self, 
        active_people_tracks: Dict[int, TrackHistory], 
        object_detections: List[Any],  # TrackedObject or DetectionResult instances for bags
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """Analyzes bags and people tracks to find abandoned packages.
        
        Args:
            active_people_tracks: Dict mapping track_id to TrackHistory for people
            object_detections: List of observed backpacks/bags in the current frame
            timestamp: Current timestamp in seconds
            
        Returns:
            List of abandoned object event alerts
        """
        alerts = []
        current_frame_obj_ids = set()
        
        for obj in object_detections:
            # Check if this object class is configured for monitoring (e.g. backpack, suitcase, handbag)
            if obj.class_id not in self.config.object_classes:
                continue
                
            obj_id = getattr(obj, "track_id", None)
            if obj_id is None:
                # If object is not tracked, we can skip or generate a pseudo-ID based on centroid proximity
                continue
                
            current_frame_obj_ids.add(obj_id)
            
            # 1. Register new object
            if obj_id not in self.tracked_objects:
                state = StationaryObjectState(obj_id, obj.bbox, obj.class_id, timestamp)
                
                # Associate Owner: find the closest person track at this moment
                closest_person_id = None
                min_dist = float('inf')
                
                for person_id, person_track in active_people_tracks.items():
                    if person_track.centroids:
                        dist = self._calculate_distance(state.centroid, person_track.centroids[-1])
                        if dist < min_dist and dist < 120.0:  # Proximity threshold to associate owner (e.g. 120 pixels)
                            min_dist = dist
                            closest_person_id = person_id
                            
                state.owner_track_id = closest_person_id
                self.tracked_objects[obj_id] = state
                
                if closest_person_id is not None:
                    logger.info(
                        f"[AbandonedObject] Associated object {obj_id} (class {obj.class_id}) "
                        f"with owner person track {closest_person_id} (distance: {min_dist:.1f}px)"
                    )
            else:
                # 2. Update existing object
                state = self.tracked_objects[obj_id]
                state.update_position(obj.bbox, timestamp)
                
                # Check owner distance
                if state.owner_track_id is not None:
                    owner_track = active_people_tracks.get(state.owner_track_id)
                    
                    if owner_track is None or not owner_track.centroids:
                        # Owner left the camera view
                        owner_dist = float('inf')
                        owner_present = False
                    else:
                        owner_dist = self._calculate_distance(state.centroid, owner_track.centroids[-1])
                        owner_present = True
                        
                    # Check if stationary duration threshold exceeded
                    stationary_duration = timestamp - state.stationary_since
                    
                    # If owner is far away or left the camera view
                    is_far = owner_dist > self.config.owner_max_distance_pixels
                    
                    if is_far and stationary_duration >= self.config.stationary_duration_seconds:
                        if not state.alert_triggered:
                            state.alert_triggered = True
                            
                            alert = {
                                "event_type": "abandoned_object",
                                "object_track_id": obj_id,
                                "class_id": state.class_id,
                                "owner_track_id": state.owner_track_id,
                                "stationary_duration": stationary_duration,
                                "owner_distance": owner_dist if owner_present else -1.0,
                                "owner_present": owner_present,
                                "timestamp": timestamp
                            }
                            alerts.append(alert)
                            logger.warning(
                                f"[Security Alert] Abandoned object {obj_id} detected! "
                                f"Stationary for {stationary_duration:.1f}s. "
                                f"Owner {state.owner_track_id} is {'far (' + str(round(owner_dist)) + 'px)' if owner_present else 'ABSENT'}."
                            )
                else:
                    # Object has no associated owner (left by someone already gone or untracked)
                    stationary_duration = timestamp - state.stationary_since
                    if stationary_duration >= self.config.stationary_duration_seconds:
                        if not state.alert_triggered:
                            state.alert_triggered = True
                            alert = {
                                "event_type": "abandoned_object",
                                "object_track_id": obj_id,
                                "class_id": state.class_id,
                                "owner_track_id": None,
                                "stationary_duration": stationary_duration,
                                "owner_distance": -1.0,
                                "owner_present": False,
                                "timestamp": timestamp
                            }
                            alerts.append(alert)
                            logger.warning(
                                f"[Security Alert] Unattended object {obj_id} detected! "
                                f"Stationary for {stationary_duration:.1f}s with no associated owner."
                            )
                            
        # Clean up objects that disappeared from view
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in current_frame_obj_ids:
                # If it hasn't been seen for 5 seconds, remove it
                if timestamp - self.tracked_objects[obj_id].last_seen > 5.0:
                    self.tracked_objects.pop(obj_id)
                    
        return alerts
