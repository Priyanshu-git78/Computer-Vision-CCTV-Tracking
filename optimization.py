from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence


MODEL_PRIORITY = (
    "yolo11x.pt",
    "yolo11l.pt",
    "yolo11m.pt",
    "yolo11s.pt",
    "yolov8n.pt",
)

PROFILE_OPTIONS = ("Max Speed", "Balanced", "Max Accuracy", "Custom")
RESOLUTION_OPTIONS = (416, 512, 640, 768, 896, 1024, 1280, 1536)
COMPILE_MODE_LABELS = {
    False: "Off",
    "reduce-overhead": "Reduce Overhead",
    "default": "Default",
    "max-autotune-no-cudagraphs": "Max Autotune",
}
COMPILE_MODE_VALUES = {label: value for value, label in COMPILE_MODE_LABELS.items()}

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "Max Speed": {
        "preferred_models": ("yolo11l.pt", "yolo11m.pt", "yolo11s.pt", "yolo11x.pt", "yolov8n.pt"),
        "frame_size": 640,
        "use_fp16": True,
        "augment_inference": False,
        "compile_mode": "max-autotune-no-cudagraphs",
        "save_video": False,
        "save_heatmap": False,
        "description": "Prioritizes FPS with GPU compile and no output exports.",
    },
    "Balanced": {
        "preferred_models": ("yolo11x.pt", "yolo11l.pt", "yolo11m.pt", "yolo11s.pt", "yolov8n.pt"),
        "frame_size": 896,
        "use_fp16": True,
        "augment_inference": False,
        "compile_mode": "reduce-overhead",
        "save_video": True,
        "save_heatmap": True,
        "description": "Uses the strongest local YOLO11 model with a higher input size and moderate compile tuning.",
    },
    "Max Accuracy": {
        "preferred_models": ("yolo11x.pt", "yolo11l.pt", "yolo11m.pt", "yolo11s.pt", "yolov8n.pt"),
        "frame_size": 1280,
        "use_fp16": True,
        "augment_inference": True,
        "compile_mode": "default",
        "save_video": True,
        "save_heatmap": True,
        "description": "Pushes the largest available model with high resolution and augmented inference.",
    },
}


def discover_available_models(root_dir: str | Path) -> list[str]:
    model_dir = Path(root_dir)
    discovered = {path.name for path in model_dir.glob("*.pt")}
    ordered = [name for name in MODEL_PRIORITY if name in discovered]
    remaining = sorted(discovered - set(ordered))
    return ordered + remaining


def get_profile_settings(profile_name: str, available_models: Sequence[str]) -> dict[str, Any]:
    if profile_name not in PROFILE_PRESETS:
        raise ValueError(f"Unsupported optimization profile: {profile_name}")

    settings = deepcopy(PROFILE_PRESETS[profile_name])
    settings["model_path"] = _choose_model(settings["preferred_models"], available_models)
    return settings


def get_compile_mode_label(mode: str | bool) -> str:
    return COMPILE_MODE_LABELS.get(mode, "Off")


def get_compile_mode_value(label: str) -> str | bool:
    return COMPILE_MODE_VALUES[label]


def _choose_model(preferred_models: Sequence[str], available_models: Sequence[str]) -> str:
    available_set = set(available_models)
    for model_name in preferred_models:
        if model_name in available_set:
            return model_name
    if available_models:
        return available_models[0]
    return preferred_models[0]
