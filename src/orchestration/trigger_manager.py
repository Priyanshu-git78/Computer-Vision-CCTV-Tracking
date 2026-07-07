import logging
from typing import Dict, List, Set, Any, Tuple
from src.tracking.track_manager import TrackHistory
from src.config import EventsConfig

logger = logging.getLogger(__name__)

class TriggerManager:
    """Evaluates rules and metrics to transition the system between dynamic states."""
    
    def __init__(self, events_config: EventsConfig):
        self.config = events_config
        self.state = "IDLE"
        
        # State transition history
        self.state_history: List[Dict[str, Any]] = []
        
        # Frame counters for stability
        self.empty_scene_frames = 0
        self.idle_cooldown_frames = 30  # Number of empty frames before dropping back to IDLE

    def set_state(self, new_state: str, reason: str = ""):
        """Sets a new state, logging the transition."""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            logger.info(f"[State Transition] {old_state} -> {new_state} | Reason: {reason}")
            self.state_history.append({
                "from_state": old_state,
                "to_state": new_state,
                "reason": reason,
                "timestamp": None  # Will be populated by the calling engine
            })

    def evaluate_state(
        self, 
        active_tracks: Dict[int, TrackHistory], 
        active_visits: Dict[Tuple[int, str], Any],
        specialist_detections: Dict[str, List[Any]],
        timestamp: float
    ) -> str:
        """Evaluates rules to determine and set the next system state.
        
        Args:
            active_tracks: Dict of track ID to TrackHistory
            active_visits: Dict of (track_id, zone_name) to active zone visit info
            specialist_detections: Dict mapping specialist model names to their detections
            timestamp: Current frame timestamp
            
        Returns:
            The newly evaluated state string
        """
        # Update transition log with timestamp if not set
        for record in self.state_history:
            if record["timestamp"] is None:
                record["timestamp"] = timestamp
                
        # Rule 1: Empty Scene Check
        if not active_tracks:
            self.empty_scene_frames += 1
            # Check if any specialist detections are active (e.g. fire/smoke detected even without people)
            fire_smoke_active = any(len(dets) > 0 for dets in specialist_detections.values() if isinstance(dets, list))
            
            if fire_smoke_active:
                self.set_state("CRITICAL_EVENT", "Fire/Smoke detected in empty scene")
                self.empty_scene_frames = 0
            elif self.empty_scene_frames >= self.idle_cooldown_frames and self.state != "IDLE":
                self.set_state("IDLE", f"Scene empty for {self.idle_cooldown_frames} frames")
            return self.state
            
        # If there are people, reset the empty scene frame counter
        self.empty_scene_frames = 0
        
        # Rule 2: Check for Critical Events (e.g. Fire/Smoke detection, or fighting verified)
        critical_triggered = False
        critical_reason = ""
        
        # Check if fire/smoke model reports detections
        if "fire_smoke" in specialist_detections and len(specialist_detections["fire_smoke"]) > 0:
            critical_triggered = True
            critical_reason = "Fire/Smoke detected"
            
        # Check if action recognition reports fighting
        if "action_recognition" in specialist_detections and len(specialist_detections["action_recognition"]) > 0:
            critical_triggered = True
            critical_reason = "Aggressive fighting behavior detected"
            
        if critical_triggered:
            self.set_state("CRITICAL_EVENT", critical_reason)
            return self.state
            
        # Rule 3: Check for Suspicious Activity (e.g. restricted zone intrusion, loitering)
        suspicious_triggered = False
        suspicious_reason = ""
        
        # Check for restricted zone intrusion
        restricted_zones = set(self.config.intrusion.restricted_zones)
        for (track_id, zone_name), visit in active_visits.items():
            if zone_name in restricted_zones:
                suspicious_triggered = True
                suspicious_reason = f"Intrusion detected: Track {track_id} entered restricted zone {zone_name}"
                break
                
            # Check if loitering is triggered for any active visit
            if getattr(visit, "loitering_triggered", False):
                suspicious_triggered = True
                suspicious_reason = f"Loitering alert: Track {track_id} in zone {zone_name}"
                break
                
        if suspicious_triggered:
            self.set_state("SUSPICIOUS_ACTIVITY", suspicious_reason)
            return self.state
            
        # Rule 4: Zone Engagement Activity
        # If there are active visits in engagement/service zones (but not restricted/loitering)
        if active_visits:
            self.set_state("ZONE_ACTIVITY", "Customers actively interacting with showroom zones")
            return self.state
            
        # Rule 5: Normal Activity
        # Persons present in camera, but no zone memberships or alerts
        self.set_state("NORMAL_ACTIVITY", "People present in scene")
        return self.state
