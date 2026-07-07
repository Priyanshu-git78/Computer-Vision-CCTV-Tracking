import cv2
import logging
import numpy as np
from typing import List, Tuple, Dict, Set, Any, Optional
from src.tracking.track_manager import TrackHistory
from src.config import ZoneConfig

logger = logging.getLogger(__name__)

class ZoneVisitRecord:
    """Stores data about a completed or ongoing visit by a person to a zone."""
    
    def __init__(self, track_id: int, zone_name: str, entry_time: float):
        self.track_id = track_id
        self.zone_name = zone_name
        self.entry_time = entry_time
        self.exit_time: Optional[float] = None
        self.duration = 0.0
        self.loitering_triggered = False
        self.trajectory: List[Tuple[float, float]] = []

    def update(self, centroid: Tuple[float, float], current_time: float):
        """Updates the ongoing visit with the latest position and updates duration."""
        self.trajectory.append(centroid)
        self.duration = current_time - self.entry_time

    def close(self, exit_time: float):
        """Closes the visit record when the person exits the zone."""
        self.exit_time = exit_time
        self.duration = exit_time - self.entry_time


class ZoneAnalyticsManager:
    """Tracks presence, occupancy, dwell times, and loitering inside polygon zones."""
    
    def __init__(self, zones: List[ZoneConfig], loitering_threshold_seconds: float = 15.0):
        self.zones = zones
        self.loitering_threshold = loitering_threshold_seconds
        
        # Prepare CV2 polygon contours for point-in-polygon checks
        self.contours: Dict[str, np.ndarray] = {}
        for z in zones:
            self.contours[z.name] = np.array(z.points, dtype=np.int32).reshape((-1, 1, 2))
            
        # Ongoing visits. Key: (track_id, zone_name), Value: ZoneVisitRecord
        self.active_visits: Dict[Tuple[int, str], ZoneVisitRecord] = {}
        
        # Completed visits. Key: zone_name, Value: List of ZoneVisitRecord
        self.completed_visits: Dict[str, List[ZoneVisitRecord]] = {z.name: [] for z in zones}
        
        # Track historical zone visits to analyze transitions
        # Key: track_id, Value: List of zone names visited in chronological order
        self.track_zone_history: Dict[int, List[str]] = {}
        
        # Transition counts. Key: (from_zone, to_zone), Value: count
        self.transitions: Dict[Tuple[str, str], int] = {}
        
        # Heatmap coordinates: list of all centroids observed in zones
        self.heatmap_coords: Dict[str, List[Tuple[float, float]]] = {z.name: [] for z in zones}

    def _is_inside_zone(self, point: Tuple[float, float], zone_name: str) -> bool:
        """Helper checking if a point is inside a zone's polygon using OpenCV pointPolygonTest."""
        contour = self.contours.get(zone_name)
        if contour is None:
            return False
        # OpenCV expects float Tuple, returns positive if inside, negative if outside, 0 if on edge
        res = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
        return res >= 0

    def update(self, active_tracks: Dict[int, TrackHistory], timestamp: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Updates zone status for active tracks and returns entry/exit/loitering events.
        
        Args:
            active_tracks: Dict of track ID to TrackHistory
            timestamp: Current frame timestamp in seconds
            
        Returns:
            Tuple of (zone_events, security_events)
        """
        zone_events = []
        security_events = []
        
        # 1. Update active tracks and detect entries / updates
        for track_id, track in active_tracks.items():
            if not track.centroids:
                continue
            
            centroid = track.centroids[-1]
            
            # Check this track against all configured zones
            for zone in self.zones:
                key = (track_id, zone.name)
                is_inside = self._is_inside_zone(centroid, zone.name)
                
                if is_inside:
                    # Accumulate heatmap coordinate
                    self.heatmap_coords[zone.name].append(centroid)
                    
                    if key not in self.active_visits:
                        # Entry Event!
                        visit = ZoneVisitRecord(track_id, zone.name, timestamp)
                        visit.update(centroid, timestamp)
                        self.active_visits[key] = visit
                        
                        # Update zone history
                        if track_id not in self.track_zone_history:
                            self.track_zone_history[track_id] = []
                        
                        # Record transitions if there was a previous zone
                        if self.track_zone_history[track_id]:
                            prev_zone = self.track_zone_history[track_id][-1]
                            if prev_zone != zone.name:
                                transition_key = (prev_zone, zone.name)
                                self.transitions[transition_key] = self.transitions.get(transition_key, 0) + 1
                                logger.info(f"[Zone Transition] Track {track_id} moved from {prev_zone} to {zone.name}")
                                
                        self.track_zone_history[track_id].append(zone.name)
                        
                        entry_event = {
                            "event_type": "zone_entry",
                            "track_id": track_id,
                            "zone_name": zone.name,
                            "timestamp": timestamp
                        }
                        zone_events.append(entry_event)
                        logger.info(f"[Zone Entry] Track {track_id} entered {zone.name} at {timestamp:.2f}s")
                    else:
                        # Update ongoing visit
                        visit = self.active_visits[key]
                        visit.update(centroid, timestamp)
                        
                        # Check for loitering
                        if not visit.loitering_triggered and visit.duration >= self.loitering_threshold:
                            visit.loitering_triggered = True
                            loiter_event = {
                                "event_type": "loitering",
                                "track_id": track_id,
                                "zone_name": zone.name,
                                "duration": visit.duration,
                                "timestamp": timestamp
                            }
                            security_events.append(loiter_event)
                            logger.warning(
                                f"[Loitering Alert] Track {track_id} loitering in {zone.name} "
                                f"for {visit.duration:.1f}s (threshold: {self.loitering_threshold}s)"
                            )
                else:
                    # If it was active in this zone, but is no longer inside, it exited
                    if key in self.active_visits:
                        visit = self.active_visits.pop(key)
                        visit.close(timestamp)
                        self.completed_visits[zone.name].append(visit)
                        
                        exit_event = {
                            "event_type": "zone_exit",
                            "track_id": track_id,
                            "zone_name": zone.name,
                            "entry_time": visit.entry_time,
                            "exit_time": visit.exit_time,
                            "duration": visit.duration,
                            "timestamp": timestamp
                        }
                        zone_events.append(exit_event)
                        logger.info(
                            f"[Zone Exit] Track {track_id} exited {zone.name} after "
                            f"{visit.duration:.2f}s"
                        )
                        
        # 2. Clean up active visits for tracks that disappeared completely from tracking
        for (track_id, zone_name), visit in list(self.active_visits.items()):
            if track_id not in active_tracks:
                # Track went stale/disappeared, close visit at its last seen time
                self.active_visits.pop((track_id, zone_name))
                visit.close(visit.last_seen)
                self.completed_visits[zone_name].append(visit)
                
                exit_event = {
                    "event_type": "zone_exit",
                    "track_id": track_id,
                    "zone_name": zone_name,
                    "entry_time": visit.entry_time,
                    "exit_time": visit.exit_time,
                    "duration": visit.duration,
                    "timestamp": visit.last_seen
                }
                zone_events.append(exit_event)
                logger.info(
                    f"[Zone Exit (Stale)] Track {track_id} exited {zone_name} after "
                    f"{visit.duration:.2f}s due to track loss"
                )
                
        return zone_events, security_events

    def get_zone_metrics(self, zone_name: str) -> Dict[str, Any]:
        """Calculates live stats for a specific zone."""
        active_count = sum(1 for (tid, zname) in self.active_visits if zname == zone_name)
        
        completed = self.completed_visits.get(zone_name, [])
        all_durations = [v.duration for v in completed]
        
        visitor_count = len(completed) + active_count
        avg_dwell = np.mean(all_durations) if all_durations else 0.0
        max_dwell = np.max(all_durations) if all_durations else 0.0
        
        return {
            "zone_name": zone_name,
            "current_occupancy": active_count,
            "visitor_count": visitor_count,
            "average_dwell_time": float(avg_dwell),
            "maximum_dwell_time": float(max_dwell),
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Calculates metrics for all zones, transitions, and overall statistics."""
        zone_metrics = {}
        for z in self.zones:
            zone_metrics[z.name] = self.get_zone_metrics(z.name)
            
        # Format transitions for API consumption
        transitions_formatted = [
            {"from_zone": from_z, "to_zone": to_z, "count": count}
            for (from_z, to_z), count in self.transitions.items()
        ]
        
        return {
            "zones": zone_metrics,
            "transitions": transitions_formatted
        }
