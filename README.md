# AI-Powered Intelligent Security Ecosystem for Experience Centers

This project converts CCTV/video streams in experience centers into an intelligent security and customer-behavior analytics system. It includes two parallel computer vision pipelines:
1. **Baseline System**: Fixed FPS frame sampling with continuous inference of all models (detection, tracking, and specialist models).
2. **Adaptive System**: Scene activity-based adaptive sampling, lightweight detector, and contextual specialist activation triggers to optimize GPU resources.

---

## 1. Project Directory Layout

```
project/
├── README.md               # This documentation file
├── pyproject.toml         # Python package dependencies & metadata
├── .env.example            # Environment variables template
├── .gitignore              # Files/folders to exclude from git
│
├── configs/                # Configuration-driven design files
│   ├── cameras.yaml        # Video sources, FPS, target zones
│   ├── models.yaml         # Model filepaths, thresholds, parameters
│   ├── zones.yaml          # Geometry of zones & crossing lines per camera
│   ├── events.yaml         # Rules & durations for security events
│   └── benchmark.yaml      # Benchmarking setup parameters
│
├── src/                    # Source code package
│   ├── ingestion/          # Video loading and streaming
│   ├── detection/          # Object detection wrappers (YOLO)
│   ├── tracking/           # Multi-object trackers (ByteTrack/BoT-SORT)
│   ├── analytics/          # Business logic (dwell time, count, trajectories)
│   ├── events/             # Contextual events (intrusion, loitering, theft)
│   ├── orchestration/      # Engines running baseline & adaptive systems
│   ├── monitoring/         # Resource monitoring (CPU, RAM, GPU/VRAM)
│   ├── database/           # PostgreSQL connection & schemas
│   └── api/                # FastAPI webserver and websockets
│
├── scripts/                # Launch scripts for baseline & adaptive models
├── tests/                  # Unit and integration tests
└── outputs/                # Local storage for reports, video clips, and logs
```

---

## 2. Installation & Setup

Ensure you are using the virtual environment `.venv` located at the root of the Ultralytics repository:

```bash
# Verify virtual environment python and pip
./.venv/bin/python --version
```

Verify your GPU environment:
```bash
nvidia-smi
```

Install the project in editable mode:
```bash
./.venv/bin/pip install -e .
```

---

## 3. Configuration Loading System

Configurations are fully decoupled from application logic and located under the `configs/` directory.

You can load and validate configurations programmatically in Python:
```python
from src.config import SystemConfig

# Initialize and load all yaml settings
cfg = SystemConfig(config_dir="configs")

# Get configuration for a specific camera
entrance_cfg = cfg.get_camera_config("entrance_01")
print(f"Role: {entrance_cfg.role}, Source: {entrance_cfg.source}")

# Get crossing lines and polygon zones
zones_cfg = cfg.get_camera_zones("entrance_01")
print(f"Lines defined: {len(zones_cfg.lines)}")
```

---

## 4. Running the Research and Benchmarking Pipelines

To run the complete MBA comparative research pipeline under identical hardware and environment constraints:

### Step 1: Download and Validate Datasets (Phase 2 & 3)
Downloads standard tracking videos, generates synthetic fallback frames/videos (with moving targets and ground grids) if offline, standardizes splits, and creates a validation QC report:
```bash
./.venv/bin/python project/scripts/download_and_validate.py
```
*Report output: `project/reports/dataset_validation_report.md`*

### Step 2: Run Multi-Strategy Model Training (Phase 4 & 5)
Executes 7 controlled training experiments (Pretrained base, Scratch yolov8n/yolov8s, Transfer learning, Augmentation trials, Hyperparameter tuning, and Domain fine-tuning) on the CPU:
```bash
./.venv/bin/python project/scripts/run_model_experiments.py
```
*Registry output: `project/results/experiment_registry.csv`*  
*Report output: `project/reports/model_comparison_report.md`*

### Step 3: Run Tracker Comparisons (Phase 6)
Executes a head-to-head empirical comparison of ByteTrack vs. BoT-SORT on the standardized tracking stream, recording latency, CPU loads, and track counts:
```bash
./.venv/bin/python project/scripts/run_tracking_experiments.py
```
*Report output: `project/reports/tracking_comparison_report.md`*

### Step 4: Run Baseline vs. Adaptive Benchmark (Phase 11)
Runs the comparative system benchmark comparing the standard continuous-inference engine (Baseline) and the event-driven state-machine engine (Adaptive) on a 500-frame sequence:
```bash
./.venv/bin/python project/scripts/benchmark.py --camera entrance_01 --max-frames 500
```
*Output output: `project/outputs/reports/report_entrance_01_*.json`*

### Step 5: Compile Final Reports & MBA Thesis (Phase 12 & 13)
Parses the benchmark results, calculates percentage changes, and generates the complete final MBA research thesis structured chapters:
```bash
./.venv/bin/python project/scripts/compile_final_reports.py
```
*Report outputs:*
- `project/reports/security_event_report.md`
- `project/reports/customer_analytics_report.md`
- `project/reports/baseline_vs_adaptive_report.md`
- `project/reports/final_mba_project_report.md` (complete academic report)

