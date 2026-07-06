from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import supervision as sv


@dataclass
class StableTrackState:
    stable_id: int
    last_tracker_id: int
    last_bbox: list[int]
    last_center: tuple[int, int]
    last_seen_frame: int
    appearance: np.ndarray | None
    observations: int = 1


class PersonTracker:
    """ByteTrack wrapper that converts raw detections into stable display IDs."""

    def __init__(
        self,
        fps: int = 25,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        reassociation_window_frames: int | None = None,
        reassociation_window_seconds: float = 20.0,
        max_center_distance_ratio: float = 0.25,
        min_appearance_similarity: float = 0.35,
        min_reassociation_score: float = 0.45,
    ) -> None:
        self.tracker = sv.ByteTrack(
            frame_rate=fps,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        self.frame_index = 0
        self.next_stable_id = 1
        self.raw_to_stable: dict[int, int] = {}
        self.raw_last_seen_frame: dict[int, int] = {}
        self.stable_tracks: dict[int, StableTrackState] = {}
        default_reassociation_window = max(
            int(round(fps * reassociation_window_seconds)),
            lost_track_buffer,
        )
        self.reassociation_window_frames = reassociation_window_frames or default_reassociation_window
        self.max_center_distance_ratio = max_center_distance_ratio
        self.min_appearance_similarity = min_appearance_similarity
        self.min_reassociation_score = min_reassociation_score

    def update(self, detections: list[dict[str, Any]], frame: np.ndarray) -> list[dict[str, Any]]:
        self.frame_index += 1

        if detections:
            xyxy = np.array([det["bbox"] for det in detections], dtype=np.float32)
            confidence = np.array([det["confidence"] for det in detections], dtype=np.float32)
            class_id = np.array([det["class_id"] for det in detections], dtype=np.int32)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
            class_id = np.empty((0,), dtype=np.int32)

        supervision_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        tracked = self.tracker.update_with_detections(supervision_detections)

        tracked_people: list[dict[str, Any]] = []
        active_raw_ids: set[int] = set()
        active_stable_ids: set[int] = set()
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else np.empty((0,), dtype=np.int32)
        for bbox, track_id, score in zip(tracked.xyxy, tracker_ids, tracked.confidence):
            x1, y1, x2, y2 = map(int, bbox.tolist())
            bbox_coords = [x1, y1, x2, y2]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            raw_track_id = int(track_id)
            stable_track_id = self._resolve_stable_id(
                raw_track_id=raw_track_id,
                bbox=bbox_coords,
                center=center,
                appearance=self._appearance_signature(frame, bbox_coords),
                frame_shape=frame.shape[:2],
                active_stable_ids=active_stable_ids,
            )
            active_raw_ids.add(raw_track_id)
            active_stable_ids.add(stable_track_id)
            tracked_people.append(
                {
                    "track_id": stable_track_id,
                    "source_track_id": raw_track_id,
                    "bbox": bbox_coords,
                    "center": center,
                    "confidence": float(score),
                    "color": self._color_from_id(stable_track_id),
                }
            )

        self._cleanup_state(active_raw_ids)
        return tracked_people

    @staticmethod
    def _color_from_id(track_id: int) -> tuple[int, int, int]:
        return (
            (37 * track_id) % 255,
            (17 * track_id) % 255,
            (29 * track_id) % 255,
        )

    def _resolve_stable_id(
        self,
        raw_track_id: int,
        bbox: list[int],
        center: tuple[int, int],
        appearance: np.ndarray | None,
        frame_shape: tuple[int, int],
        active_stable_ids: set[int],
    ) -> int:
        stable_id = self.raw_to_stable.get(raw_track_id)
        if stable_id is None:
            stable_id = self._find_reassociated_track(
                bbox=bbox,
                center=center,
                appearance=appearance,
                frame_shape=frame_shape,
                active_stable_ids=active_stable_ids,
            )
            if stable_id is None:
                stable_id = self.next_stable_id
                self.next_stable_id += 1
            self.raw_to_stable[raw_track_id] = stable_id

        self.raw_last_seen_frame[raw_track_id] = self.frame_index
        previous_state = self.stable_tracks.get(stable_id)
        self.stable_tracks[stable_id] = StableTrackState(
            stable_id=stable_id,
            last_tracker_id=raw_track_id,
            last_bbox=bbox,
            last_center=center,
            last_seen_frame=self.frame_index,
            appearance=self._blend_appearance(
                previous_state.appearance if previous_state is not None else None,
                appearance,
            ),
            observations=(previous_state.observations + 1) if previous_state is not None else 1,
        )
        return stable_id

    def _find_reassociated_track(
        self,
        bbox: list[int],
        center: tuple[int, int],
        appearance: np.ndarray | None,
        frame_shape: tuple[int, int],
        active_stable_ids: set[int],
    ) -> int | None:
        frame_height, frame_width = frame_shape
        frame_diagonal = max((frame_height**2 + frame_width**2) ** 0.5, 1.0)
        max_center_distance = frame_diagonal * self.max_center_distance_ratio
        best_match_id: int | None = None
        best_match_score = 0.0

        for stable_id, state in self.stable_tracks.items():
            if stable_id in active_stable_ids:
                continue

            age_frames = self.frame_index - state.last_seen_frame
            if age_frames <= 0 or age_frames > self.reassociation_window_frames:
                continue

            center_distance = float(np.linalg.norm(np.array(center) - np.array(state.last_center)))
            if center_distance > max_center_distance:
                continue

            appearance_similarity = self._appearance_similarity(state.appearance, appearance)
            iou = self._bbox_iou(state.last_bbox, bbox)
            motion_score = max(0.0, 1.0 - (center_distance / max(max_center_distance, 1.0)))
            iou_score = min(iou / 0.25, 1.0)
            freshness_score = max(0.0, 1.0 - (age_frames / self.reassociation_window_frames))
            match_score = (
                0.55 * appearance_similarity
                + 0.2 * motion_score
                + 0.15 * iou_score
                + 0.1 * freshness_score
            )

            if appearance_similarity < self.min_appearance_similarity and iou < 0.15:
                continue
            if match_score > best_match_score:
                best_match_id = stable_id
                best_match_score = match_score

        if best_match_score >= self.min_reassociation_score:
            return best_match_id
        return None

    def _cleanup_state(self, active_raw_ids: set[int]) -> None:
        stale_raw_ids = [
            raw_track_id
            for raw_track_id, last_seen_frame in self.raw_last_seen_frame.items()
            if raw_track_id not in active_raw_ids
            and self.frame_index - last_seen_frame > self.reassociation_window_frames
        ]
        for raw_track_id in stale_raw_ids:
            self.raw_last_seen_frame.pop(raw_track_id, None)
            self.raw_to_stable.pop(raw_track_id, None)

        stale_stable_ids = [
            stable_id
            for stable_id, state in self.stable_tracks.items()
            if self.frame_index - state.last_seen_frame > self.reassociation_window_frames
        ]
        for stable_id in stale_stable_ids:
            self.stable_tracks.pop(stable_id, None)

    @staticmethod
    def _appearance_signature(frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv_crop], [0, 1], None, [12, 12], [0, 180, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten().astype(np.float32)
        return histogram

    @staticmethod
    def _appearance_similarity(
        reference: np.ndarray | None,
        candidate: np.ndarray | None,
    ) -> float:
        if reference is None or candidate is None:
            return 0.0
        distance = cv2.compareHist(reference, candidate, cv2.HISTCMP_BHATTACHARYYA)
        return max(0.0, 1.0 - float(distance))

    @staticmethod
    def _blend_appearance(
        reference: np.ndarray | None,
        candidate: np.ndarray | None,
    ) -> np.ndarray | None:
        if reference is None:
            return candidate
        if candidate is None:
            return reference

        blended = (0.85 * reference) + (0.15 * candidate)
        return cv2.normalize(blended, None).flatten().astype(np.float32)

    @staticmethod
    def _bbox_iou(first_bbox: list[int], second_bbox: list[int]) -> float:
        ax1, ay1, ax2, ay2 = first_bbox
        bx1, by1, bx2, by2 = second_bbox

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height
        if intersection <= 0:
            return 0.0

        first_area = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
        second_area = max(bx2 - bx1, 0) * max(by2 - by1, 0)
        union = max(first_area + second_area - intersection, 1)
        return intersection / union
