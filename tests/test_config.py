import os
import sys

# Ensure project root and src/ are in PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import SystemConfig

def test_load_configurations():
    # Construct paths relative to the test file
    config_dir = os.path.join(project_root, "configs")
    assert os.path.exists(config_dir), f"Config dir does not exist: {config_dir}"
    
    cfg = SystemConfig(config_dir=config_dir)
    
    # Assert cameras loaded
    assert len(cfg.cameras) > 0, "No cameras loaded"
    assert cfg.get_camera_config("entrance_01") is not None
    
    # Assert models loaded
    assert cfg.models is not None
    assert cfg.models.primary_detector.model_type == "yolo"
    assert cfg.models.primary_detector.device == "cuda"
    
    # Assert camera zones loaded
    assert "entrance_01" in cfg.camera_zones
    entrance_zones = cfg.get_camera_zones("entrance_01")
    assert len(entrance_zones.lines) > 0
    assert len(entrance_zones.zones) > 0
    
    # Assert event rules loaded
    assert cfg.events is not None
    assert cfg.events.loitering.dwell_time_threshold_seconds == 15.0
    
    # Assert benchmark config loaded
    assert cfg.benchmark is not None
    assert cfg.benchmark.duration_seconds == 60.0

if __name__ == "__main__":
    test_load_configurations()
    print("All configuration tests passed successfully!")
