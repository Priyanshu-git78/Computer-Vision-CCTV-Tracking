import logging
from typing import List, Tuple, Dict, Set, Optional, Any
from src.tracking.track_manager import TrackHistory
from src.config import LineConfig

logger = logging.getLogger(__name__)

def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Helper to check if three points A, B, C are in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float], D: Tuple[float, float]) -> bool:
    """Checks if line segment AB intersects line segment CD."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

class LineCrossingDetector:
    """Detects when tracked objects cross a specified line and counts them directionally."""
    
    def __init__(self, config: LineConfig):
        self.name = config.name
        self.start = config.start
        self.end = config.end
        self.in_direction = config.in_direction
        
        # Track counts
        self.in_count = 0
        self.out_count = 0
        
        # Keep track of the last side of the line for each track ID to prevent double counting
        # Key: track_id, Value: side (1 for positive side, -1 for negative side, 0 for on the line)
        self.track_sides: Dict[int, int] = {}
        
        # Set of track IDs that have crossed (maps track_id to direction)
        # We store this to prevent duplicate crossing events in the same direction
        self.crossings_log: List[Dict[str, Any]] = []

    def _get_side(self, point: Tuple[float, float]) -> int:
        """Determines which side of the line the point lies on.
        
        Using cross product sign of vector L1->L2 and L1->Point:
        positive: right/below, negative: left/above
        """
        val = (self.end[0] - self.start[0]) * (point[1] - self.start[1]) - (self.end[1] - self.start[1]) * (point[0] - self.start[0])
        if val > 0:
            return 1
        elif val < 0:
            return -1
        return 0

    def update(self, active_tracks: Dict[int, TrackHistory], timestamp: float) -> List[Dict[str, Any]]:
        """Updates line crossing state for active tracks and returns new crossing events.
        
        Args:
            active_tracks: Dict of track ID to TrackHistory
            timestamp: Current frame timestamp in seconds
            
        Returns:
            List of crossing event dicts
        """
        new_events = []
        
        for track_id, track in active_tracks.items():
            if len(track.centroids) < 2:
                # Need at least two points to establish motion
                continue
                
            p_prev = track.centroids[-2]
            p_curr = track.centroids[-1]
            
            side_prev = self.track_sides.get(track_id)
            side_curr = self._get_side(p_curr)
            
            # If we don't have a record for this track yet, initialize it
            if side_prev is None:
                self.track_sides[track_id] = self._get_side(p_prev)
                side_prev = self.track_sides[track_id]
                
            # Check if crossing occurred
            if side_prev != side_curr and side_prev != 0 and side_curr != 0:
                # Perform segment intersection check to be mathematically certain
                if intersect(p_prev, p_curr, self.start, self.end):
                    direction = ""
                    
                    # Determine crossing direction
                    # If we cross from negative to positive side
                    if side_prev == -1 and side_curr == 1:
                        direction = "in" if self.in_direction == "down" or self.in_direction == "right" else "out"
                    # If we cross from positive to negative side
                    elif side_prev == 1 and side_curr == -1:
                        direction = "out" if self.in_direction == "down" or self.in_direction == "right" else "in"
                        
                    if direction:
                        # Log event
                        event = {
                            "track_id": track_id,
                            "line_name": self.name,
                            "direction": direction,
                            "timestamp": timestamp,
                            "confidence": track.confidences[-1]
                        }
                        
                        # Apply counts
                        if direction == "in":
                            self.in_count += 1
                        else:
                            self.out_count += 1
                            
                        self.crossings_log.append(event)
                        new_events.append(event)
                        logger.info(f"[Line Crossing] Track {track_id} crossed {self.name} {direction.upper()} at {timestamp:.2f}s")
                        
                    # Update side to the new side
                    self.track_sides[track_id] = side_curr
            else:
                # Just update the current side without crossing
                self.track_sides[track_id] = side_curr
                
        # Clean up track_sides for tracks that are no longer active to prevent memory bloat
        inactive_ids = set(self.track_sides.keys()) - set(active_tracks.keys())
        for track_id in inactive_ids:
            self.track_sides.pop(track_id, None)
            
        return new_events
