from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass
class VideoInputConfig:
    camera_id: str
    source: str | int
    display: bool = True


class CsvLogger:
    def __init__(self, csv_path: str) -> None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["camera_id", "track_id", "frame_index", "timestamp", "x1", "y1", "x2", "y2", "dwell_time", "direction"]
        )

    def write_row(
        self,
        camera_id: str,
        track_id: int,
        frame_index: int,
        timestamp: float,
        bbox: list[int],
        dwell_time: float,
        direction: str,
    ) -> None:
        self.writer.writerow([camera_id, track_id, frame_index, timestamp, *bbox, dwell_time, direction])

    def close(self) -> None:
        self.file.close()


class VisitorSessionLogger:
    def __init__(self, csv_path: str) -> None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self.file = open(csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["id", "entry_time", "exit_time", "dwell_time"])

    def write_row(
        self,
        visitor_id: int,
        entry_time: float,
        exit_time: float | None,
        dwell_time: float,
    ) -> None:
        self.writer.writerow(
            [
                visitor_id,
                round(entry_time, 3),
                round(exit_time, 3) if exit_time is not None else "",
                round(dwell_time, 2),
            ]
        )
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as file:
        if path.suffix.lower() == ".json":
            raw_config = json.load(file)
        else:
            raw_config = yaml.safe_load(file)

    output_dir = Path(raw_config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = [
        VideoInputConfig(
            camera_id=str(camera["camera_id"]),
            source=camera["source"],
            display=bool(camera.get("display", True)),
        )
        for camera in raw_config["cameras"]
    ]

    return {
        "processing": raw_config["processing"],
        "tracking": raw_config["tracking"],
        "analytics": raw_config["analytics"],
        "output": raw_config["output"],
        "api": raw_config["api"],
        "ui": raw_config.get("ui", {"title": "Retail Analytics Hub"}),
        "cameras": cameras,
    }


def get_video_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def save_uploaded_video(uploaded_file: Any, output_dir: str) -> str:
    upload_dir = Path(output_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=upload_dir) as temp_file:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        while True:
            chunk = uploaded_file.read(16 * 1024 * 1024)
            if not chunk:
                break
            temp_file.write(chunk)
        return temp_file.name


def resize_frame(frame: np.ndarray, target_size: int) -> tuple[np.ndarray, float]:
    resized = cv2.resize(frame, (target_size, target_size))
    scale_x = target_size / max(frame.shape[1], 1)
    return resized, scale_x


def scale_line_points(
    line_points: tuple[list[int], list[int]],
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[list[int], list[int]]:
    orig_w, orig_h = original_size
    target_w, target_h = target_size

    scaled_points: list[list[int]] = []
    for x, y in line_points:
        scaled_points.append([int(x * target_w / max(orig_w, 1)), int(y * target_h / max(orig_h, 1))])
    return tuple(scaled_points)  # type: ignore[return-value]


def draw_box_label(
    frame: np.ndarray, bbox: list[int], label: str, color: tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_virtual_line(frame: np.ndarray, line_points: tuple[list[int], list[int]]) -> None:
    p1, p2 = line_points
    cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 255), 2)
    cv2.putText(
        frame,
        "Entry / Exit Line",
        (p1[0], max(30, p1[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_analytics_overlay(frame: np.ndarray, analytics: dict[str, Any], camera_id: str) -> None:
    overlay_lines = [
        f"Camera: {camera_id}",
        f"Total visitors: {analytics['total_visitors']}",
        f"Current people: {analytics['current_people_count']}",
        f"Entries: {analytics['entry_count']}",
        f"Exits: {analytics['exit_count']}",
        f"Avg dwell: {analytics['average_dwell_time_seconds']} sec",
    ]
    if "fps" in analytics:
        overlay_lines.append(f"FPS: {analytics['fps']}")
    if "model_name" in analytics:
        overlay_lines.append(f"Model: {analytics['model_name']}")
    if "input_size" in analytics:
        overlay_lines.append(f"Inference: {analytics['input_size']}")
    if "device" in analytics:
        overlay_lines.append(f"Device: {analytics['device']}")
    if "precision" in analytics:
        overlay_lines.append(f"Precision: {analytics['precision']}")
    x, y = 12, 25
    for line in overlay_lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 25


def create_writer(output_path: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def build_camera_output_paths(base_dir: Path, camera_id: str) -> dict[str, str]:
    camera_dir = base_dir / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    return {
        "video": str(camera_dir / "annotated_output.mp4"),
        "csv": str(camera_dir / "tracking_data.csv"),
        "heatmap": str(camera_dir / "movement_heatmap.jpg"),
    }


def generate_heatmap(
    heatmap_points: list[tuple[int, int]],
    frame_size: tuple[int, int],
    output_path: str,
) -> None:
    width, height = frame_size
    heatmap = np.zeros((height, width), dtype=np.float32)

    for x, y in heatmap_points:
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += 1.0

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, colored_heatmap)
