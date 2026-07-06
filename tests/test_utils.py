import csv
from pathlib import Path
import numpy as np
import yaml

from utils import (
    VideoInputConfig,
    CsvLogger,
    VisitorSessionLogger,
    load_config,
    get_video_source,
    resize_frame,
    scale_line_points,
    build_camera_output_paths,
)


def test_video_input_config():
    config = VideoInputConfig(camera_id="cam1", source="video.mp4", display=False)
    assert config.camera_id == "cam1"
    assert config.source == "video.mp4"
    assert not config.display


def test_get_video_source():
    assert get_video_source(0) == 0
    assert get_video_source("0") == 0
    assert get_video_source("rtsp://test") == "rtsp://test"
    assert get_video_source("video.mp4") == "video.mp4"


def test_resize_frame():
    # Create a dummy image frame (height=100, width=200, channels=3)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    resized, scale_x = resize_frame(frame, target_size=128)
    
    assert resized.shape == (128, 128, 3)
    assert scale_x == 128 / 200.0


def test_scale_line_points():
    line_points = ([100, 200], [300, 400])
    original_size = (640, 480)
    target_size = (1280, 960)
    
    scaled = scale_line_points(line_points, original_size, target_size)
    assert scaled == ([200, 400], [600, 800])


def test_build_camera_output_paths(tmp_path):
    base_dir = Path(tmp_path)
    paths = build_camera_output_paths(base_dir, "test_cam")
    
    assert paths["video"] == str(base_dir / "test_cam" / "annotated_output.mp4")
    assert paths["csv"] == str(base_dir / "test_cam" / "tracking_data.csv")
    assert paths["heatmap"] == str(base_dir / "test_cam" / "movement_heatmap.jpg")
    assert (base_dir / "test_cam").exists()


def test_csv_logger(tmp_path):
    csv_file = tmp_path / "test_log.csv"
    logger = CsvLogger(str(csv_file))
    logger.write_row(
        camera_id="cam_01",
        track_id=42,
        frame_index=150,
        timestamp=6.0,
        bbox=[10, 20, 30, 40],
        dwell_time=1.5,
        direction="entry",
    )
    logger.close()
    
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        
    assert len(reader) == 2
    assert reader[0] == ["camera_id", "track_id", "frame_index", "timestamp", "x1", "y1", "x2", "y2", "dwell_time", "direction"]
    assert reader[1] == ["cam_01", "42", "150", "6.0", "10", "20", "30", "40", "1.5", "entry"]


def test_visitor_session_logger(tmp_path):
    csv_file = tmp_path / "visitor_sessions.csv"
    logger = VisitorSessionLogger(str(csv_file))
    logger.write_row(
        visitor_id=1,
        entry_time=12.3456,
        exit_time=15.7891,
        dwell_time=3.4435,
    )
    logger.close()
    
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        
    assert len(reader) == 2
    assert reader[0] == ["id", "entry_time", "exit_time", "dwell_time"]
    assert reader[1] == ["1", "12.346", "15.789", "3.44"]


def test_load_config(tmp_path):
    config_data = {
        "processing": {
            "optimization_profile": "GPU Ultra Realtime",
            "model_path": "yolo11x.onnx",
            "frame_size": 1280,
            "confidence": 0.25,
            "iou_threshold": 0.45,
            "use_fp16": False,
            "device": "cuda:0",
            "augment_inference": False,
            "compile_mode": False,
            "realtime": {
                "target_fps": 18,
                "low_fps_patience": 12,
                "adaptive_frame_sizes": [1280, 1024, 960, 768],
                "drop_stale_frames": True,
            }
        },
        "tracking": {
            "enabled": True,
            "fps": 25,
            "track_activation_threshold": 0.2,
            "lost_track_buffer": 45,
            "minimum_matching_threshold": 0.65,
            "reassociation_window_seconds": 20.0,
            "max_center_distance_ratio": 0.25,
            "min_appearance_similarity": 0.35,
            "min_reassociation_score": 0.45,
        },
        "analytics": {
            "entry_line": [[80, 220], [560, 220]],
            "line_margin": 12,
            "dwell_timeout_seconds": 2.0,
            "trail_length": 30,
        },
        "output": {
            "directory": str(tmp_path / "outputs"),
            "save_video": True,
            "save_csv": True,
            "save_heatmap": True,
        },
        "api": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 8000,
        },
        "ui": {
            "title": "RetailVision CCTV Analytics",
            "refresh_interval_ms": 250,
        },
        "cameras": [
            {
                "camera_id": "front_door",
                "source": 0,
                "display": False
            }
        ]
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)
        
    config = load_config(str(config_file))
    
    assert config["processing"]["optimization_profile"] == "GPU Ultra Realtime"
    assert len(config["cameras"]) == 1
    assert isinstance(config["cameras"][0], VideoInputConfig)
    assert config["cameras"][0].camera_id == "front_door"
    assert config["cameras"][0].source == 0
    assert not config["cameras"][0].display
    assert Path(config["output"]["directory"]).exists()
