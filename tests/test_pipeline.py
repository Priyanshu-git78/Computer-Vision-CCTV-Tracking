import os
import sys

# Ensure project root is in PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import LineConfig, ZoneConfig
from src.tracking.tracker import TrackedObject
from src.tracking.track_manager import TrackManager
from src.analytics.line_crossing import LineCrossingDetector
from src.analytics.dwell_time import ZoneAnalyticsManager
from src.analytics.occupancy import OccupancyTracker

def test_tracker_manager_history():
    manager = TrackManager(max_history_points=10)
    
    # Simulate track updates
    track_objs = [
        TrackedObject(track_id=1, bbox=(100, 100, 150, 150), confidence=0.9, class_id=0),
        TrackedObject(track_id=2, bbox=(200, 200, 260, 260), confidence=0.8, class_id=0)
    ]
    
    manager.update(track_objs, timestamp=1.0)
    
    active = manager.get_active_tracks(current_time=1.0, max_age_seconds=2.0)
    assert 1 in active
    assert 2 in active
    assert len(active[1].centroids) == 1
    assert active[1].centroids[-1] == (125.0, 125.0)
    
    # Move track 1 and update again
    track_objs = [
        TrackedObject(track_id=1, bbox=(110, 110, 160, 160), confidence=0.95, class_id=0)
    ]
    manager.update(track_objs, timestamp=2.0)
    
    active = manager.get_active_tracks(current_time=2.0, max_age_seconds=2.0)
    assert 1 in active
    # Track 2 should be inactive/stale at t=2.0 with max_age=0.5
    active_recent = manager.get_active_tracks(current_time=2.0, max_age_seconds=0.5)
    assert 2 not in active_recent
    
    # Check history limit
    for i in range(15):
        manager.update([TrackedObject(track_id=1, bbox=(110, 110, 160, 160), confidence=0.9, class_id=0)], timestamp=3.0+i)
    assert len(manager.get_track(1).centroids) == 10  # clamped to max_history_points

def test_line_crossing_logic():
    line_cfg = LineConfig(
        name="test_line",
        start=(100, 200),
        end=(300, 200),
        in_direction="down"
    )
    detector = LineCrossingDetector(line_cfg)
    
    # Mock Track history moving from (200, 150) to (200, 250) - crossing down
    manager = TrackManager()
    
    # Step 1: Above line
    manager.update([TrackedObject(track_id=1, bbox=(180, 130, 220, 170), confidence=0.9, class_id=0)], timestamp=0.0)
    active = manager.get_active_tracks(0.0)
    events = detector.update(active, timestamp=0.0)
    assert len(events) == 0
    
    # Step 2: Crossing Below line
    manager.update([TrackedObject(track_id=1, bbox=(180, 230, 220, 270), confidence=0.9, class_id=0)], timestamp=1.0)
    active = manager.get_active_tracks(1.0)
    events = detector.update(active, timestamp=1.0)
    
    assert len(events) == 1
    assert events[0]["track_id"] == 1
    assert events[0]["direction"] == "in"
    assert detector.in_count == 1
    assert detector.out_count == 0

def test_zone_analytics_logic():
    zone_cfg = ZoneConfig(
        name="test_zone",
        points=[(100, 100), (200, 100), (200, 200), (100, 200)],
        type="restricted",
        description="test restricted area"
    )
    
    # Dwell threshold = 2 seconds
    manager = ZoneAnalyticsManager(zones=[zone_cfg], loitering_threshold_seconds=2.0)
    
    track_manager = TrackManager()
    
    # Frame 1: Person enters zone
    track_manager.update([TrackedObject(track_id=1, bbox=(120, 120, 160, 160), confidence=0.9, class_id=0)], timestamp=0.0)
    active = track_manager.get_active_tracks(0.0)
    zone_events, security_events = manager.update(active, timestamp=0.0)
    
    assert len(zone_events) == 1
    assert zone_events[0]["event_type"] == "zone_entry"
    assert zone_events[0]["zone_name"] == "test_zone"
    assert len(security_events) == 0
    
    # Frame 2: Person still inside zone, dwell time increases but under loitering threshold
    track_manager.update([TrackedObject(track_id=1, bbox=(130, 130, 170, 170), confidence=0.9, class_id=0)], timestamp=1.0)
    active = track_manager.get_active_tracks(1.0)
    zone_events, security_events = manager.update(active, timestamp=1.0)
    assert len(zone_events) == 0
    assert len(security_events) == 0
    
    # Frame 3: Dwell time reaches 2.5 seconds (exceeds 2.0 threshold) -> triggers loitering
    track_manager.update([TrackedObject(track_id=1, bbox=(130, 130, 170, 170), confidence=0.9, class_id=0)], timestamp=2.5)
    active = track_manager.get_active_tracks(2.5)
    zone_events, security_events = manager.update(active, timestamp=2.5)
    
    assert len(security_events) == 1
    assert security_events[0]["event_type"] == "loitering"
    assert security_events[0]["track_id"] == 1
    
    # Frame 4: Person exits zone
    track_manager.update([TrackedObject(track_id=1, bbox=(250, 250, 290, 290), confidence=0.9, class_id=0)], timestamp=3.0)
    active = track_manager.get_active_tracks(3.0)
    zone_events, security_events = manager.update(active, timestamp=3.0)
    
    assert len(zone_events) == 1
    assert zone_events[0]["event_type"] == "zone_exit"
    assert zone_events[0]["duration"] == 3.0  # From 0.0 to 3.0

if __name__ == "__main__":
    print("Running test_tracker_manager_history...")
    test_tracker_manager_history()
    print("Running test_line_crossing_logic...")
    test_line_crossing_logic()
    print("Running test_zone_analytics_logic...")
    test_zone_analytics_logic()
    print("All core pipeline tests passed successfully!")
