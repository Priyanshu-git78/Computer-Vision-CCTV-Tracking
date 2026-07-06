# CCTV Shop Analytics System

A lightweight, deployable retail analytics project that combines Ultralytics YOLO11 X for person detection, ByteTrack for identity tracking, OpenCV for video processing, and Streamlit for an interactive shop-owner dashboard.

## Features

- Webcam or CCTV video file input
- RTSP stream support
- Multi-camera ready pipeline design driven by `config.yaml`
- YOLO11 X person-only detection
- ByteTrack-based unique person IDs
- Live visitor count and total unique visitors
- Entry and exit counting using a virtual line
- Per-customer dwell time
- Path trail drawing and optional heatmap export
- CSV visitor session logs with `id, entry_time, exit_time, dwell_time`
- Streamlit UI with start and stop controls, source selector, confidence slider, and live dashboard
- Accuracy-first defaults with optional FP16 on supported hardware

## Project Structure

```text
cctv-shop-analytics/
|-- app.py
|-- analytics.py
|-- config.yaml
|-- detector.py
|-- main.py
|-- README.md
|-- requirements.txt
|-- tracker.py
`-- utils.py
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Streamlit UI

```bash
streamlit run app.py
```

## Optional CLI Run

```bash
python main.py --config config.yaml
```

## UI Features

- Start and stop buttons
- Input source selector for webcam, uploaded file, or RTSP stream
- Confidence threshold slider
- Tracking on and off toggle
- Save video toggle
- Heatmap toggle
- Live metrics for total visitors, current people count, entries, exits, and average dwell time
- CSV download after a run finishes or while it is running

## Configuration

The system is configured from `config.yaml`.

- `processing.frame_size`: Resize each frame to this square size for faster inference
- `processing.device`: Use `"cpu"`, `"cuda:0"`, or `"auto"`
- `tracking.enabled`: Default tracking state for the UI and CLI
- `cameras`: Default camera blocks for CLI execution
- `analytics.entry_line`: Virtual line used for entry and exit detection
- `output.save_csv`: Enables per-track CSV export
- `output.save_heatmap`: Saves a movement heatmap after processing ends
- `ui.refresh_interval_ms`: Streamlit refresh cadence for live updates

## Example Camera Configurations

```yaml
cameras:
  - camera_id: "front_door"
    source: 0
    display: true

  - camera_id: "cash_counter"
    source: "rtsp://username:password@192.168.1.20:554/stream"
    display: false

  - camera_id: "aisle_cam"
    source: "videos/shop_floor.mp4"
    display: true
```

## Outputs

The system writes outputs to `outputs/<camera_id>/`:

- `annotated_output.mp4`
- `tracking_data.csv`
- `movement_heatmap.jpg`

## Development & CI/CD

### Local Testing and Linting
To keep the codebase clean and verify logic, we use `pytest` for unit testing and `ruff` for code style verification:

1. **Install dev dependencies**:
   ```bash
   pip install pytest ruff
   ```

2. **Run Linting**:
   ```bash
   ruff check .
   ```

3. **Run Unit Tests**:
   ```bash
   pytest tests/
   ```

### CI/CD Pipeline
Every push and pull request to the `main` branch triggers a GitHub Actions workflow (`.github/workflows/ci.yml`) that:
- Sets up Python 3.10.
- Installs necessary system libraries (e.g. `libGL` for OpenCV).
- Installs Python dependencies.
- Runs `ruff` checks to enforce code quality.
- Runs `pytest` to ensure all tests pass.

## Notes

- ByteTrack is strong at maintaining IDs during short occlusions and low-confidence detection gaps, but long full exits from view can still produce a new ID when the person returns.
- For best entry and exit accuracy, place the virtual line near the doorway and tune `line_margin`.
- On CPU, reduce `processing.frame_size` to `416` or `512` if the accuracy-first defaults are too heavy.
- If tracking is disabled from the UI, the app still shows live detections and current count, but unique visitor counting and dwell analytics are intentionally limited.
