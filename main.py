from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analytics import CameraAnalytics
from detector import PersonDetector
from realtime import RealtimeTuner, scale_detections_to_output_size
from tracker import PersonTracker
from utils import (
    VideoInputConfig,
    build_camera_output_paths,
    create_writer,
    draw_analytics_overlay,
    draw_box_label,
    draw_virtual_line,
    generate_heatmap,
    get_video_source,
    load_config,
    resize_frame,
    scale_line_points,
)


class RetailAnalyticsPipeline:
    """Runs detection, optional tracking, and analytics for a single input stream."""

    def __init__(self, camera_config: VideoInputConfig, runtime_config: dict[str, Any]) -> None:
        self.camera_config = camera_config
        self.runtime_config = runtime_config
        self.camera_id = camera_config.camera_id
        self.output_paths = build_camera_output_paths(
            Path(runtime_config["output"]["directory"]), self.camera_id
        )

        processing_cfg = runtime_config["processing"]
        tracking_cfg = runtime_config["tracking"]
        analytics_cfg = runtime_config["analytics"]
        self.tracking_config = tracking_cfg
        realtime_cfg = processing_cfg.get("realtime", {})
        self.output_size = processing_cfg["frame_size"]
        self.realtime_tuner = RealtimeTuner(
            output_size=self.output_size,
            adaptive_frame_sizes=list(realtime_cfg.get("adaptive_frame_sizes", [self.output_size])),
            target_fps=float(realtime_cfg.get("target_fps", 0.0)),
            low_fps_patience=int(realtime_cfg.get("low_fps_patience", 10)),
        )
        self.inference_size = self.realtime_tuner.current_size
        self.drop_stale_frames = bool(realtime_cfg.get("drop_stale_frames", False))
        self.original_entry_line = tuple(deepcopy(analytics_cfg["entry_line"]))

        self.detector = PersonDetector(
            model_path=processing_cfg["model_path"],
            input_size=self.inference_size,
            confidence=processing_cfg["confidence"],
            iou_threshold=processing_cfg["iou_threshold"],
            use_half=processing_cfg["use_fp16"],
            device=processing_cfg["device"],
            augment_inference=bool(processing_cfg.get("augment_inference", False)),
            compile_mode=processing_cfg.get("compile_mode", False),
        )
        self.runtime_summary = self.detector.runtime_summary()
        self.tracking_enabled = bool(tracking_cfg["enabled"])
        self.tracker: PersonTracker | None = None
        self.analytics = CameraAnalytics(
            camera_id=self.camera_id,
            entry_line=tuple(analytics_cfg["entry_line"]),
            line_margin=analytics_cfg["line_margin"],
            dwell_timeout_seconds=analytics_cfg["dwell_timeout_seconds"],
            csv_path=self.output_paths["csv"] if runtime_config["output"]["save_csv"] else None,
        )
        self.trail_points: dict[int, deque[tuple[int, int]]] = {}
        self.trail_length = analytics_cfg["trail_length"]
        self.show_heatmap = bool(runtime_config["output"]["save_heatmap"])
        self.heatmap_points: list[tuple[int, int]] = []

        self.capture: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None
        self.frame_size = (self.output_size, self.output_size)
        self.fps = tracking_cfg["fps"]
        self.use_video_timestamps = isinstance(self.camera_config.source, str) and Path(
            str(self.camera_config.source)
        ).exists()
        self.source_frame_size = (self.output_size, self.output_size)
        self.frame_index = 0
        self.initialized = False
        self._last_tick = time.perf_counter()
        self._smoothed_fps = 0.0

    def open(self) -> None:
        source = get_video_source(self.camera_config.source)
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open source for camera '{self.camera_id}': {source}")
        if self.drop_stale_frames and not self.use_video_timestamps:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.output_size
        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.output_size
        self.source_frame_size = (frame_width, frame_height)
        capture_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if capture_fps > 0:
            self.fps = capture_fps

        if self.tracking_enabled:
            reference_fps = max(float(self.tracking_config.get("fps", self.fps)), 1.0)
            scaled_lost_track_buffer = max(
                int(round(self.tracking_config["lost_track_buffer"] * (self.fps / reference_fps))),
                1,
            )
            self.tracker = PersonTracker(
                fps=max(int(round(self.fps)), 1),
                track_activation_threshold=self.tracking_config["track_activation_threshold"],
                lost_track_buffer=scaled_lost_track_buffer,
                minimum_matching_threshold=self.tracking_config["minimum_matching_threshold"],
                reassociation_window_seconds=float(
                    self.tracking_config.get("reassociation_window_seconds", 20.0)
                ),
                max_center_distance_ratio=float(
                    self.tracking_config.get("max_center_distance_ratio", 0.25)
                ),
                min_appearance_similarity=float(
                    self.tracking_config.get("min_appearance_similarity", 0.35)
                ),
                min_reassociation_score=float(
                    self.tracking_config.get("min_reassociation_score", 0.45)
                ),
            )

        self.analytics.entry_line = scale_line_points(
            self.original_entry_line,
            (frame_width, frame_height),
            (self.output_size, self.output_size),
        )

        if self.runtime_config["output"]["save_video"]:
            self.writer = create_writer(
                self.output_paths["video"],
                fps=self.fps,
                frame_size=self.frame_size,
            )

        self.initialized = True

    def process_next_frame(self) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
        if not self.initialized:
            self.open()
        assert self.capture is not None

        ok, frame = self.capture.read()
        if not ok:
            return False, None, self.analytics.current_snapshot()

        self.frame_index += 1
        output_frame, _ = resize_frame(frame, self.output_size)
        frame_timestamp = self._current_timestamp()
        if self.inference_size == self.output_size:
            inference_frame = output_frame
        else:
            inference_frame, _ = resize_frame(frame, self.inference_size)

        detections = self.detector.detect(inference_frame)
        detections = scale_detections_to_output_size(
            detections,
            inference_size=self.inference_size,
            output_size=self.output_size,
        )

        if self.tracker is not None:
            tracked_people = self.tracker.update(detections, output_frame)
            analytics_snapshot = self.analytics.update(tracked_people, timestamp=frame_timestamp)
        else:
            tracked_people = self._create_untracked_people(detections)
            analytics_snapshot = self.analytics.update_without_tracking(
                len(detections),
                timestamp=frame_timestamp,
            )

        annotated_frame = output_frame.copy()
        for person in tracked_people:
            self.heatmap_points.append(person["center"])
            if self.tracker is not None:
                track_id = person["track_id"]
                if track_id not in self.trail_points:
                    self.trail_points[track_id] = deque(maxlen=self.trail_length)
                self.trail_points[track_id].append(person["center"])
                draw_box_label(annotated_frame, person["bbox"], f"ID {track_id}", person["color"])
                self._draw_trail(annotated_frame, list(self.trail_points[track_id]), person["color"])
            else:
                draw_box_label(annotated_frame, person["bbox"], "Person", (0, 200, 255))

        self._update_fps()
        self._maybe_adjust_inference_size()
        analytics_snapshot["fps"] = round(self._smoothed_fps, 2)
        analytics_snapshot["device"] = self.runtime_summary["device"]
        analytics_snapshot["device_name"] = self.runtime_summary["device_name"]
        analytics_snapshot["precision"] = "FP16" if self.runtime_summary["use_half"] else "FP32"
        analytics_snapshot["input_size"] = self.inference_size
        analytics_snapshot["model_name"] = self.runtime_summary["model_name"]
        analytics_snapshot["augment_inference"] = self.runtime_summary["augment_inference"]
        analytics_snapshot["compile_mode"] = self.runtime_summary["compile_mode"]
        analytics_snapshot["optimization_profile"] = self.runtime_config["processing"].get(
            "optimization_profile",
            "Custom",
        )

        draw_virtual_line(annotated_frame, self.analytics.entry_line)
        draw_analytics_overlay(annotated_frame, analytics_snapshot, self.camera_id)

        if self.writer is not None:
            self.writer.write(annotated_frame)

        return True, annotated_frame, analytics_snapshot

    def close(self) -> None:
        if self.show_heatmap and self.heatmap_points:
            generate_heatmap(
                heatmap_points=self.heatmap_points,
                frame_size=self.frame_size,
                output_path=self.output_paths["heatmap"],
            )

        self.analytics.close()

        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    @staticmethod
    def _draw_trail(
        frame: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]
    ) -> None:
        if len(points) < 2:
            return
        for idx in range(1, len(points)):
            cv2.line(frame, points[idx - 1], points[idx], color, 2)

    @staticmethod
    def _create_untracked_people(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        people: list[dict[str, Any]] = []
        for idx, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection["bbox"]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            people.append(
                {
                    "track_id": idx,
                    "bbox": detection["bbox"],
                    "center": center,
                    "confidence": detection["confidence"],
                    "color": (0, 200, 255),
                }
            )
        return people

    def _update_fps(self) -> None:
        now = time.perf_counter()
        delta = max(now - self._last_tick, 1e-6)
        instant_fps = 1.0 / delta
        if self._smoothed_fps == 0.0:
            self._smoothed_fps = instant_fps
        else:
            self._smoothed_fps = (0.85 * self._smoothed_fps) + (0.15 * instant_fps)
        self._last_tick = now

    def _maybe_adjust_inference_size(self) -> None:
        next_size = self.realtime_tuner.observe(self._smoothed_fps)
        if next_size is None or next_size == self.inference_size:
            return

        self.inference_size = next_size
        self.detector.input_size = next_size
        self.runtime_summary = self.detector.runtime_summary()
        print(
            f"[Realtime] Sustained low FPS detected on '{self.camera_id}'. "
            f"Reducing inference size to {next_size}."
        )

    def _current_timestamp(self) -> float:
        if self.capture is not None and self.use_video_timestamps:
            video_seconds = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if video_seconds > 0:
                return video_seconds
            if self.fps > 0:
                return self.frame_index / self.fps
        return time.time()


class PipelineController:
    """Background worker used by the Streamlit UI."""

    def __init__(self, camera_config: VideoInputConfig, runtime_config: dict[str, Any]) -> None:
        self.pipeline = RetailAnalyticsPipeline(camera_config, runtime_config)
        self.latest_frame_bgr: np.ndarray | None = None
        self.latest_snapshot: dict[str, Any] = self.pipeline.analytics.current_snapshot()
        self.error_message: str | None = None
        self.running = False
        self.finished = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=f"pipeline-{camera_config.camera_id}")

    def start(self) -> None:
        self.running = True
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.running = False

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def snapshot(self) -> tuple[np.ndarray | None, dict[str, Any], str | None, bool]:
        with self._lock:
            frame = None if self.latest_frame_bgr is None else self.latest_frame_bgr.copy()
            return frame, deepcopy(self.latest_snapshot), self.error_message, self.finished

    def output_paths(self) -> dict[str, str]:
        return self.pipeline.output_paths

    def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                ok, frame, snapshot = self.pipeline.process_next_frame()
                with self._lock:
                    self.latest_snapshot = snapshot
                    if frame is not None:
                        self.latest_frame_bgr = frame
                if not ok:
                    break
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.error_message = str(exc)
        finally:
            self.pipeline.close()
            with self._lock:
                self.finished = True
                self.running = False


def build_runtime_config(
    config: dict[str, Any],
    confidence: float | None = None,
    tracking_enabled: bool | None = None,
    save_video: bool | None = None,
    save_heatmap: bool | None = None,
    device: str | None = None,
    model_path: str | None = None,
    frame_size: int | None = None,
    use_fp16: bool | None = None,
    augment_inference: bool | None = None,
    compile_mode: str | bool | None = None,
    optimization_profile: str | None = None,
) -> dict[str, Any]:
    runtime_config = deepcopy(config)
    if confidence is not None:
        runtime_config["processing"]["confidence"] = confidence
    if tracking_enabled is not None:
        runtime_config["tracking"]["enabled"] = tracking_enabled
    if save_video is not None:
        runtime_config["output"]["save_video"] = save_video
    if save_heatmap is not None:
        runtime_config["output"]["save_heatmap"] = save_heatmap
    if device is not None:
        runtime_config["processing"]["device"] = device
    if model_path is not None:
        runtime_config["processing"]["model_path"] = model_path
    if frame_size is not None:
        runtime_config["processing"]["frame_size"] = frame_size
    if use_fp16 is not None:
        runtime_config["processing"]["use_fp16"] = use_fp16
    if augment_inference is not None:
        runtime_config["processing"]["augment_inference"] = augment_inference
    if compile_mode is not None:
        runtime_config["processing"]["compile_mode"] = compile_mode
    if optimization_profile is not None:
        runtime_config["processing"]["optimization_profile"] = optimization_profile
    return runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCTV-based customer tracking and shop analytics system")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML or JSON config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime_config = build_runtime_config(config)

    controllers: list[PipelineController] = []
    for camera_cfg in runtime_config["cameras"]:
        controller = PipelineController(camera_cfg, runtime_config)
        controllers.append(controller)
        controller.start()

    try:
        while any(not controller.finished for controller in controllers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        for controller in controllers:
            controller.stop()


if __name__ == "__main__":
    main()
