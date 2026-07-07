import os
import yaml
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field

# ==========================================
# 1. Camera Configurations
# ==========================================

class CameraConfig(BaseModel):
    camera_id: str
    source: str
    role: str
    target_fps: int = 10
    enabled: bool = True
    resolution: Tuple[int, int] = (1920, 1080)


# ==========================================
# 2. Model Configurations
# ==========================================

class PrimaryDetectorConfig(BaseModel):
    model_type: str = "yolo"
    path: str = "yolo26n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    classes: List[int] = [0]
    device: str = "cuda"

class TrackerConfig(BaseModel):
    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8

class SpecialistModelConfig(BaseModel):
    model_type: str
    path: str
    conf_threshold: float
    iou_threshold: Optional[float] = 0.45
    classes: Optional[List[int]] = None
    device: str = "cuda"
    enabled: bool = True

class ModelsConfig(BaseModel):
    primary_detector: PrimaryDetectorConfig
    tracker: TrackerConfig
    specialist_models: Dict[str, SpecialistModelConfig]


# ==========================================
# 3. Zone and Line Configurations
# ==========================================

class LineConfig(BaseModel):
    name: str
    start: Tuple[int, int]
    end: Tuple[int, int]
    in_direction: str = "down"

class ZoneConfig(BaseModel):
    name: str
    points: List[Tuple[int, int]]
    type: str
    description: str = ""

class CameraZonesConfig(BaseModel):
    lines: List[LineConfig] = Field(default_factory=list)
    zones: List[ZoneConfig] = Field(default_factory=list)


# ==========================================
# 4. Event Rules Configurations
# ==========================================

class LoiteringConfig(BaseModel):
    dwell_time_threshold_seconds: float = 15.0
    check_interval_seconds: float = 1.0

class IntrusionConfig(BaseModel):
    restricted_zones: List[str]
    min_confidence: float = 0.50

class AbandonedObjectConfig(BaseModel):
    stationary_duration_seconds: float = 30.0
    owner_max_distance_pixels: float = 150.0
    object_classes: List[int] = [24, 26, 28]

class TheftConfig(BaseModel):
    product_interaction_duration_seconds: float = 5.0
    suspicious_pose_angle_threshold: float = 45.0

class FightingConfig(BaseModel):
    consecutive_active_frames: int = 5
    fps_sample_rate: int = 15

class FallConfig(BaseModel):
    aspect_ratio_threshold: float = 0.5
    velocity_threshold: float = 200.0

class EventsConfig(BaseModel):
    loitering: LoiteringConfig
    intrusion: IntrusionConfig
    abandoned_object: AbandonedObjectConfig
    theft: TheftConfig
    fighting: FightingConfig
    fall: FallConfig


# ==========================================
# 5. Benchmark Configurations
# ==========================================

class BenchmarkConfig(BaseModel):
    duration_seconds: float = 60.0
    monitor_interval_seconds: float = 0.5
    warmup_frames: int = 10
    save_annotated_video: bool = True
    output_dir: str = "outputs/reports"
    test_sources: Dict[str, str] = Field(default_factory=dict)


# ==========================================
# Configuration Loader
# ==========================================

class SystemConfig:
    """Consolidated configuration manager that loads configurations from YAML files."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.cameras: List[CameraConfig] = []
        self.models: Optional[ModelsConfig] = None
        self.camera_zones: Dict[str, CameraZonesConfig] = {}
        self.events: Optional[EventsConfig] = None
        self.benchmark: Optional[BenchmarkConfig] = None
        
        self.load_all()

    def _load_yaml(self, filename: str) -> dict:
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or {}

    def load_all(self):
        # 1. Cameras
        cameras_data = self._load_yaml("cameras.yaml")
        self.cameras = [CameraConfig(**c) for c in cameras_data.get("cameras", [])]
        
        # 2. Models
        models_data = self._load_yaml("models.yaml")
        self.models = ModelsConfig(**models_data)
        
        # 3. Zones and Lines
        zones_data = self._load_yaml("zones.yaml")
        cameras_zones_raw = zones_data.get("cameras_zones", {})
        self.camera_zones = {
            cam_id: CameraZonesConfig(**cfg) 
            for cam_id, cfg in cameras_zones_raw.items()
        }
        
        # 4. Events
        events_data = self._load_yaml("events.yaml")
        self.events = EventsConfig(**events_data)
        
        # 5. Benchmark
        benchmark_data = self._load_yaml("benchmark.yaml")
        self.benchmark = BenchmarkConfig(**benchmark_data.get("benchmark", {}))

    def get_camera_config(self, camera_id: str) -> Optional[CameraConfig]:
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                return cam
        return None

    def get_camera_zones(self, camera_id: str) -> CameraZonesConfig:
        return self.camera_zones.get(camera_id, CameraZonesConfig())
