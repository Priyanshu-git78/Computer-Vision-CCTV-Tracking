import cv2
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.tracking.track_manager import TrackHistory
from src.config import TheftConfig, ZoneConfig

logger = logging.getLogger(__name__)

class TheftInteractionState:
    """Tracks ongoing hand-to-product zone interaction events for a tracked person."""
    
    def __init__(self, track_id: int, timestamp: float):
        self.track_id = track_id
        self.start_time = timestamp
        self.last_seen = timestamp
        
        self.is_interacting = False
        self.interaction_logged = False
        
        # Sequence tracking: has hand moved to product, then back to pocket?
        self.hand_touched_product = False
        self.hand_touched_pocket = False
        self.touch_product_time = 0.0

class TheftDetector:
    """Detects suspicious shoplifting behavior by combining product zones and human keypoints (wrists/hips)."""
    
    def __init__(self, config: TheftConfig, product_zones: List[ZoneConfig]):
        self.config = config
        self.product_zones = product_zones
        
        # Prepare CV2 polygon contours for point-in-polygon checks
        self.contours: Dict[str, np.ndarray] = {}
        for z in product_zones:
            self.contours[z.name] = np.array(z.points, dtype=np.int32).reshape((-1, 1, 2))
            
        # Ongoing interactions. Key: (track_id, zone_name), Value: TheftInteractionState
        self.active_interactions: Dict[Tuple[int, str], TheftInteractionState] = {}
        
        # COCO keypoint indexes:
        # 9: left_wrist, 10: right_wrist
        # 11: left_hip, 12: right_hip
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12

    def _is_inside_zone(self, point: Tuple[float, float], zone_name: str) -> bool:
        contour = self.contours.get(zone_name)
        if contour is None:
            return False
        res = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
        return res >= 0

    def _calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

    def update(
        self, 
        active_tracks: Dict[int, TrackHistory],
        keypoints_by_track: Dict[int, np.ndarray],  # Key: track_id, Value: array of shape (17, 3) (x, y, conf)
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """Updates interaction sequences and reports suspicious behavior events.
        
        Args:
            active_tracks: Dict of track ID to TrackHistory
            keypoints_by_track: Dict of track ID to human pose keypoints
            timestamp: Current timestamp in seconds
            
        Returns:
            List of suspicious theft/shoplifting events
        """
        alerts = []
        
        for track_id, keypoints in keypoints_by_track.items():
            if track_id not in active_tracks:
                continue
                
            # Extract relevant keypoints
            # Verify coordinates have reasonable confidence (e.g. > 0.4)
            left_w = keypoints[self.LEFT_WRIST]
            right_w = keypoints[self.RIGHT_WRIST]
            left_h = keypoints[self.LEFT_HIP]
            right_h = keypoints[self.RIGHT_HIP]
            
            # Check interaction with each product display zone
            for zone in self.product_zones:
                key = (track_id, zone.name)
                
                # Check if hands/wrists are inside the product zone
                left_hand_in = left_w[2] > 0.4 and self._is_inside_zone((left_w[0], left_w[1]), zone.name)
                right_hand_in = right_w[2] > 0.4 and self._is_inside_zone((right_w[0], right_w[1]), zone.name)
                
                hand_in_zone = left_hand_in or right_hand_in
                
                if hand_in_zone:
                    if key not in self.active_interactions:
                        self.active_interactions[key] = TheftInteractionState(track_id, timestamp)
                        
                    state = self.active_interactions[key]
                    state.last_seen = timestamp
                    state.hand_touched_product = True
                    state.touch_product_time = timestamp
                    
                    duration = timestamp - state.start_time
                    if duration >= self.config.product_interaction_duration_seconds:
                        state.is_interacting = True
                        if not state.interaction_logged:
                            state.interaction_logged = True
                            logger.info(
                                f"[TheftDetector] Track {track_id} interacting with products in {zone.name} "
                                f"for {duration:.1f}s"
                            )
                else:
                    # Hands are not currently inside the product display zone
                    if key in self.active_interactions:
                        state = self.active_interactions[key]
                        state.last_seen = timestamp
                        
                        # Temporal sequence: if the hand was recently touching a product (< 3 seconds ago),
                        # check if it is now touching or very close to their pocket/hip area (simulating concealment)
                        time_since_touch = timestamp - state.touch_product_time
                        if state.hand_touched_product and time_since_touch < 3.0:
                            
                            # Calculate distance between wrist and hip keypoints (proximate pocket area)
                            dist_lh = self._calculate_distance((left_w[0], left_w[1]), (left_h[0], left_h[1])) if left_w[2] > 0.4 and left_h[2] > 0.4 else float('inf')
                            dist_rh = self._calculate_distance((right_w[0], right_w[1]), (right_h[0], right_h[1])) if right_w[2] > 0.4 and right_h[2] > 0.4 else float('inf')
                            
                            # If hand moves close to pocket (hip) shortly after product touch
                            pocket_threshold = 45.0  # pixels
                            if dist_lh < pocket_threshold or dist_rh < pocket_threshold:
                                state.hand_touched_pocket = True
                                
                                # Trigger Shoplifting Behavior Sequence Alert
                                alert = {
                                    "event_type": "suspicious_theft_behavior",
                                    "track_id": track_id,
                                    "zone_name": zone.name,
                                    "interaction_duration": timestamp - state.start_time,
                                    "timestamp": timestamp,
                                    "confidence": 0.85
                                }
                                alerts.append(alert)
                                logger.warning(
                                    f"[Security Alert] Suspicious theft sequence! "
                                    f"Track {track_id} touched products in {zone.name} and moved hand to "
                                    f"pocket/hip area within {time_since_touch:.1f}s."
                                )
                                # Clear active interaction to avoid duplicate triggers
                                self.active_interactions.pop(key)
                                
        # Clean up stale interactions
        for key, state in list(self.active_interactions.items()):
            if timestamp - state.last_seen > 4.0:
                self.active_interactions.pop(key)
                
        return alerts
