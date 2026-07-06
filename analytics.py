from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from utils import VisitorSessionLogger


@dataclass
class TrackState:
    first_seen: float
    last_seen: float
    last_center: tuple[int, int]
    entry_time: float
    exit_time: float | None = None
    direction: str = "unknown"
    counted_entry: bool = False
    counted_exit: bool = False
    crossed_line: bool = False
    history: list[tuple[int, int]] = field(default_factory=list)


class CameraAnalytics:
    def __init__(
        self,
        camera_id: str,
        entry_line: tuple[list[int], list[int]],
        line_margin: int = 12,
        dwell_timeout_seconds: float = 2.0,
        save_csv: bool = False,
        csv_path: str | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.entry_line = entry_line
        self.line_margin = line_margin
        self.dwell_timeout_seconds = dwell_timeout_seconds
        self.save_csv = save_csv
        self.csv_path = csv_path

        self.track_states: dict[int, TrackState] = {}
        self.unique_visitors: set[int] = set()
        self.entry_count = 0
        self.exit_count = 0
        self.completed_dwell_times: list[float] = []
        self.session_logger = VisitorSessionLogger(csv_path) if csv_path else None
        self.last_snapshot = self.current_snapshot()

        if csv_path:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        tracked_people: list[dict[str, Any]],
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if timestamp is None else timestamp
        active_ids: set[int] = set()

        for person in tracked_people:
            track_id = person["track_id"]
            center = person["center"]
            active_ids.add(track_id)
            self.unique_visitors.add(track_id)

            if track_id not in self.track_states:
                self.track_states[track_id] = TrackState(
                    first_seen=now,
                    last_seen=now,
                    last_center=center,
                    entry_time=now,
                    history=[center],
                )
            else:
                state = self.track_states[track_id]
                previous_center = state.last_center
                state.last_seen = now
                state.last_center = center
                state.history.append(center)
                state.history = state.history[-100:]
                self._handle_line_crossing(state, previous_center, center)

            state = self.track_states[track_id]
            person["dwell_time"] = round(now - state.first_seen, 2)
            person["direction"] = state.direction

        self._flush_inactive_tracks(now, active_ids)

        active_dwell_times = [
            now - state.first_seen
            for track_id, state in self.track_states.items()
            if track_id in active_ids
        ]
        avg_dwell = mean(self.completed_dwell_times + active_dwell_times) if (self.completed_dwell_times or active_dwell_times) else 0.0

        self.last_snapshot = {
            "camera_id": self.camera_id,
            "total_visitors": len(self.unique_visitors),
            "current_people_count": len(active_ids),
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "average_dwell_time_seconds": round(avg_dwell, 2),
            "active_track_ids": sorted(active_ids),
            "timestamp": now,
        }
        return self.last_snapshot

    def update_without_tracking(
        self,
        current_people_count: int,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if timestamp is None else timestamp
        self.last_snapshot = {
            "camera_id": self.camera_id,
            "total_visitors": 0,
            "current_people_count": current_people_count,
            "entry_count": 0,
            "exit_count": 0,
            "average_dwell_time_seconds": 0.0,
            "active_track_ids": [],
            "timestamp": now,
        }
        return self.last_snapshot

    def current_snapshot(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "total_visitors": self.last_snapshot.get("total_visitors", 0) if hasattr(self, "last_snapshot") else 0,
            "current_people_count": self.last_snapshot.get("current_people_count", 0) if hasattr(self, "last_snapshot") else 0,
            "entry_count": self.last_snapshot.get("entry_count", 0) if hasattr(self, "last_snapshot") else 0,
            "exit_count": self.last_snapshot.get("exit_count", 0) if hasattr(self, "last_snapshot") else 0,
            "average_dwell_time_seconds": self.last_snapshot.get("average_dwell_time_seconds", 0.0)
            if hasattr(self, "last_snapshot")
            else 0.0,
            "active_track_ids": self.last_snapshot.get("active_track_ids", []) if hasattr(self, "last_snapshot") else [],
            "timestamp": time.time(),
        }

    def _flush_inactive_tracks(self, now: float, active_ids: set[int]) -> None:
        expired_ids: list[int] = []
        for track_id, state in self.track_states.items():
            if track_id in active_ids:
                continue
            if now - state.last_seen > self.dwell_timeout_seconds:
                dwell_time = state.last_seen - state.first_seen
                self.completed_dwell_times.append(dwell_time)
                state.exit_time = state.last_seen
                if self.session_logger is not None:
                    self.session_logger.write_row(
                        visitor_id=track_id,
                        entry_time=state.entry_time,
                        exit_time=state.exit_time,
                        dwell_time=dwell_time,
                    )
                expired_ids.append(track_id)

        for track_id in expired_ids:
            del self.track_states[track_id]

    def _handle_line_crossing(
        self,
        state: TrackState,
        previous_center: tuple[int, int],
        current_center: tuple[int, int],
    ) -> None:
        p1, p2 = self.entry_line
        previous_side = self._point_line_side(previous_center, tuple(p1), tuple(p2))
        current_side = self._point_line_side(current_center, tuple(p1), tuple(p2))
        previous_distance = self._point_line_distance(previous_center, tuple(p1), tuple(p2))
        current_distance = self._point_line_distance(current_center, tuple(p1), tuple(p2))

        if previous_distance <= self.line_margin or current_distance <= self.line_margin:
            return
        if previous_side == 0 or current_side == 0 or previous_side * current_side > 0:
            return

        if current_side < 0 and not state.counted_entry:
            self.entry_count += 1
            state.counted_entry = True
            state.direction = "entry"
        elif current_side > 0 and not state.counted_exit:
            self.exit_count += 1
            state.counted_exit = True
            state.direction = "exit"

        state.crossed_line = True

    @staticmethod
    def _point_line_side(
        point: tuple[int, int],
        line_start: tuple[int, int],
        line_end: tuple[int, int],
    ) -> float:
        return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (
            line_end[1] - line_start[1]
        ) * (point[0] - line_start[0])

    @classmethod
    def _point_line_distance(
        cls,
        point: tuple[int, int],
        line_start: tuple[int, int],
        line_end: tuple[int, int],
    ) -> float:
        numerator = abs(cls._point_line_side(point, line_start, line_end))
        denominator = max(
            ((line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2) ** 0.5,
            1e-6,
        )
        return numerator / denominator

    def close(self) -> None:
        if self.track_states:
            now = self.last_snapshot.get("timestamp", time.time())
            for track_id, state in list(self.track_states.items()):
                dwell_time = now - state.first_seen
                self.completed_dwell_times.append(dwell_time)
                state.exit_time = now
                if self.session_logger is not None:
                    self.session_logger.write_row(
                        visitor_id=track_id,
                        entry_time=state.entry_time,
                        exit_time=state.exit_time,
                        dwell_time=dwell_time,
                    )
            self.track_states.clear()
        if self.session_logger is not None:
            self.session_logger.close()
