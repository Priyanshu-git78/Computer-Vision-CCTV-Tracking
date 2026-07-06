"""
FastAPI application server.

Replaces the Streamlit UI entirely.
Serves the HTML dashboard, accepts video uploads, and exposes REST endpoints.

Run:
    python app.py
    # Then open http://localhost:8000
"""

from __future__ import annotations

import shutil
import threading
import time
from urllib.parse import quote
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import uvicorn
from fastapi import BackgroundTasks, FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from main import PipelineController, build_runtime_config
from utils import VideoInputConfig, build_camera_output_paths, load_config

# ─── Globals ──────────────────────────────────────────────────────────────
CONFIG_PATH   = "config.yaml"
UPLOAD_DIR    = Path("outputs/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Active per-camera pipeline controllers
_controllers:  dict[str, PipelineController] = {}
_ctrl_lock     = threading.Lock()

# Rolling snapshot cache (last 300 readings per camera)
_snapshots:    dict[str, deque[dict[str, Any]]] = {}


# ─── Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start live-camera pipelines on boot; clean up on shutdown."""
    config = load_config(CONFIG_PATH)
    runtime = build_runtime_config(config)

    for cam_cfg in runtime["cameras"]:
        _start_pipeline(cam_cfg, runtime)

    yield

    with _ctrl_lock:
        for ctrl in _controllers.values():
            ctrl.stop()
        for ctrl in _controllers.values():
            ctrl.join(timeout=5.0)


app = FastAPI(title="RetailVision CCTV Analytics", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# ─── Helper ───────────────────────────────────────────────────────────────
def _start_pipeline(cam_cfg: VideoInputConfig, runtime_config: dict[str, Any]) -> None:
    ctrl = PipelineController(cam_cfg, runtime_config)
    ctrl.start()
    with _ctrl_lock:
        _controllers[cam_cfg.camera_id] = ctrl
        _snapshots[cam_cfg.camera_id] = deque(maxlen=300)


def _collect_snapshots() -> None:
    """Background thread: drain pipeline snapshots into history deque."""
    while True:
        time.sleep(0.3)
        with _ctrl_lock:
            for cam_id, ctrl in _controllers.items():
                _, snapshot, _, _ = ctrl.snapshot()
                if snapshot:
                    _snapshots[cam_id].append(snapshot)


def _camera_output_manifest(camera_id: str) -> dict[str, dict[str, str | bool]]:
    with _ctrl_lock:
        ctrl = _controllers.get(camera_id)

    output_paths = ctrl.output_paths() if ctrl is not None else build_camera_output_paths(Path("outputs"), camera_id)
    encoded_camera_id = quote(camera_id, safe="")
    manifest: dict[str, dict[str, str | bool]] = {}
    for key, filename in (
        ("video", "annotated_output.mp4"),
        ("csv", "tracking_data.csv"),
        ("heatmap", "movement_heatmap.jpg"),
    ):
        manifest[key] = {
            "exists": Path(output_paths[key]).exists(),
            "url": f"/outputs/{encoded_camera_id}/{filename}",
        }
    return manifest


threading.Thread(target=_collect_snapshots, daemon=True, name="snapshot-collector").start()


# ─── Routes ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/api/cameras")
def list_cameras() -> JSONResponse:
    with _ctrl_lock:
        cam_ids = list(_controllers.keys())
    return JSONResponse({"cameras": cam_ids})


@app.get("/api/stats")
def all_stats() -> JSONResponse:
    """Latest snapshot from every active camera."""
    result: dict[str, Any] = {}
    with _ctrl_lock:
        for cam_id, history in _snapshots.items():
            result[cam_id] = history[-1] if history else {}
    return JSONResponse(result)


@app.get("/api/stats/{camera_id}")
def camera_stats(camera_id: str) -> JSONResponse:
    with _ctrl_lock:
        history = _snapshots.get(camera_id)
    if history is None:
        return JSONResponse({"error": f"Camera '{camera_id}' not found"}, status_code=404)
    return JSONResponse(history[-1] if history else {})


@app.get("/api/history/{camera_id}")
def camera_history(camera_id: str) -> JSONResponse:
    """Return the last 300 snapshots for time-series charts."""
    with _ctrl_lock:
        history = _snapshots.get(camera_id)
    if history is None:
        return JSONResponse({"error": f"Camera '{camera_id}' not found"}, status_code=404)
    return JSONResponse({"camera_id": camera_id, "history": list(history)})


@app.get("/api/frame/{camera_id}")
def camera_frame(camera_id: str) -> Response:
    with _ctrl_lock:
        ctrl = _controllers.get(camera_id)
    if ctrl is None:
        return JSONResponse({"error": f"Camera '{camera_id}' not found"}, status_code=404)

    frame, _, _, _ = ctrl.snapshot()
    if frame is None:
        return Response(status_code=204)

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return Response(status_code=500)

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/outputs/{camera_id}")
def camera_outputs(camera_id: str) -> JSONResponse:
    return JSONResponse(
        {
            "camera_id": camera_id,
            **_camera_output_manifest(camera_id),
        }
    )


@app.post("/api/upload")
async def upload_video(bg: BackgroundTasks, file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload a video file for analysis.
    The file is saved to disk then processed in a background task.
    Uses a 16 MB C-level buffer — fast & memory-efficient for large files.
    """
    dest = UPLOAD_DIR / file.filename  # type: ignore[arg-type]
    with dest.open("wb") as buf:
        shutil.copyfileobj(file.file, buf, length=16 * 1024 * 1024)

    bg.add_task(_process_uploaded_video, str(dest))
    return JSONResponse({"status": "queued", "file": file.filename})


def _process_uploaded_video(video_path: str) -> None:
    """Background task: create a one-shot pipeline for the uploaded file."""
    try:
        config  = load_config(CONFIG_PATH)
        runtime = build_runtime_config(config)

        cam_id  = f"upload_{Path(video_path).stem}"
        cam_cfg = VideoInputConfig(camera_id=cam_id, source=video_path, display=False)

        _start_pipeline(cam_cfg, runtime)
    except Exception as exc:
        print(f"[Upload] Pipeline error: {exc}")


@app.delete("/api/stop/{camera_id}")
def stop_camera(camera_id: str) -> JSONResponse:
    with _ctrl_lock:
        ctrl = _controllers.pop(camera_id, None)
        _snapshots.pop(camera_id, None)
    if ctrl is None:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    ctrl.stop()
    ctrl.join(timeout=5.0)
    return JSONResponse({"status": "stopped", "camera_id": camera_id})


# ─── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        config = load_config(CONFIG_PATH)
        api_cfg = config.get("api", {})
        host = api_cfg.get("host", "0.0.0.0")
        port = int(api_cfg.get("port", 8000))
    except Exception:
        host, port = "0.0.0.0", 8000

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        timeout_keep_alive=600,
    )
