import os
import sys
import time
import logging
import threading
import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Ensure project root is in PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import SystemConfig
from src.database.connection import init_db, get_db, SessionLocal
from src.database import repository
from src.api.websocket import ConnectionManager
from src.orchestration.baseline_engine import BaselineEngine
from src.orchestration.adaptive_engine import AdaptiveEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="AI-Powered Intelligent Security Ecosystem",
    description="REST API and WebSocket server for experience center security and analytics.",
    version="0.1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration and DB initialization
config_path = os.path.join(project_root, "configs")
sys_config = SystemConfig(config_dir=config_path)

# Initialize DB on startup
@app.on_event("startup")
def startup_event():
    init_db()

# WebSocket Connection Manager
ws_manager = ConnectionManager()

# Background stream jobs storage
# Key: camera_id, Value: {"thread": Thread, "engine": Engine, "mode": str}
running_streams: Dict[str, Dict[str, Any]] = {}


# ==========================================
# API Models
# ==========================================

class StreamStartRequest(BaseModel):
    camera_id: str
    mode: str = "adaptive"  # baseline or adaptive
    max_frames: Optional[int] = None


# ==========================================
# REST Endpoints
# ==========================================

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Intelligent Security & Behavior Analytics Ecosystem</title>
        <!-- Tailwind CSS CDN -->
        <script src="https://cdn.tailwindcss.com"></script>
        <!-- Chart.js CDN -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <!-- Font -->
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; }
        </style>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col">
        <!-- Top Nav -->
        <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="h-4 w-4 rounded-full bg-indigo-500 animate-pulse"></div>
                <h1 class="text-xl font-bold tracking-tight bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent">
                    AI Security & Behavior Analytics
                </h1>
            </div>
            <div class="flex items-center space-x-4">
                <span class="text-xs text-slate-400">Host System Status:</span>
                <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    Active
                </span>
            </div>
        </header>

        <!-- Main Body -->
        <main class="flex-grow p-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
            <!-- Left Side Control Panel -->
            <div class="lg:col-span-1 space-y-6">
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
                    <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Stream Controls</h2>
                    
                    <div class="space-y-3">
                        <div>
                            <label class="block text-xs text-slate-500 mb-1">Camera Feed</label>
                            <select id="camera-select" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-3.5 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                                <option value="entrance_01">Entrance (entrance_01)</option>
                                <option value="main_hall_01">Main Showroom (main_hall_01)</option>
                                <option value="restricted_corridor_01">Restricted Area (restricted_corridor_01)</option>
                            </select>
                        </div>
                        
                        <div>
                            <label class="block text-xs text-slate-500 mb-1">Engine Mode</label>
                            <div class="grid grid-cols-2 gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800">
                                <button id="mode-adaptive" onclick="setMode('adaptive')" class="py-1.5 text-xs font-medium rounded-lg bg-indigo-600 text-slate-100 transition-all">
                                    Adaptive
                                </button>
                                <button id="mode-baseline" onclick="setMode('baseline')" class="py-1.5 text-xs font-medium rounded-lg text-slate-400 hover:text-slate-200 transition-all">
                                    Baseline
                                </button>
                            </div>
                        </div>
                    </div>

                    <div class="pt-2 flex flex-col space-y-2">
                        <button id="btn-start" onclick="startStream()" class="w-full py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 hover:from-indigo-600 hover:to-violet-700 text-sm font-semibold text-white shadow-lg shadow-indigo-500/20 transition-all">
                            Start Stream
                        </button>
                        <button id="btn-stop" onclick="stopStream()" class="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm font-semibold text-slate-300 transition-all hidden">
                            Stop Stream
                        </button>
                    </div>
                </div>

                <!-- Engine State Monitor -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
                    <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Ecosystem State</h2>
                    <div id="state-container" class="flex flex-col items-center justify-center py-6 bg-slate-950 rounded-xl border border-slate-900">
                        <span id="state-badge" class="px-4 py-2 rounded-full text-lg font-bold bg-slate-800 text-slate-400 border border-slate-700">
                            OFFLINE
                        </span>
                        <p id="state-desc" class="text-xs text-slate-500 text-center mt-3 px-4">
                            No camera stream processing active.
                        </p>
                    </div>
                </div>
            </div>

            <!-- Center Panel: Live Stats and Charts -->
            <div class="lg:col-span-3 space-y-6">
                <!-- Stats Summary Cards -->
                <div class="grid grid-cols-1 sm:grid-cols-4 gap-6">
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                        <p class="text-xs text-slate-500 font-medium">Live Occupancy</p>
                        <p id="stat-occupancy" class="text-3xl font-extrabold mt-1 text-indigo-400">0</p>
                    </div>
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                        <p class="text-xs text-slate-500 font-medium">Total Entries</p>
                        <p id="stat-entries" class="text-3xl font-extrabold mt-1 text-emerald-400">0</p>
                    </div>
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                        <p class="text-xs text-slate-500 font-medium">Total Exits</p>
                        <p id="stat-exits" class="text-3xl font-extrabold mt-1 text-slate-400">0</p>
                    </div>
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                        <p class="text-xs text-slate-500 font-medium">Active Alerts</p>
                        <p id="stat-alerts" class="text-3xl font-extrabold mt-1 text-rose-500">0</p>
                    </div>
                </div>

                <!-- Two-Column Chart Layout -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Performance Chart -->
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                        <h3 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">Resource Utilization</h3>
                        <div class="h-64">
                            <canvas id="perfChart"></canvas>
                        </div>
                    </div>
                    
                    <!-- Alert list panel -->
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col">
                        <h3 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">Live Incident alerts</h3>
                        <div id="alerts-log" class="flex-grow space-y-3 overflow-y-auto max-h-64 pr-2 text-xs">
                            <p class="text-slate-500 text-center py-10">No alerts triggered in this session.</p>
                        </div>
                    </div>
                </div>

                <!-- Zone Occupancy Grid -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                    <h3 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">Zone Engagement & Dwell analytics</h3>
                    <div id="zones-grid" class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="bg-slate-950 p-4 rounded-xl border border-slate-900">
                            <p class="text-xs font-semibold text-slate-400">Lobby Zone</p>
                            <div class="flex justify-between items-end mt-2">
                                <div>
                                    <span class="text-xs text-slate-500">Dwell:</span>
                                    <span id="zone-lobby-dwell" class="text-sm font-bold text-slate-200">0.0s</span>
                                </div>
                                <div class="text-right">
                                    <span class="text-xs text-slate-500">People:</span>
                                    <span id="zone-lobby-occ" class="text-sm font-bold text-indigo-400">0</span>
                                </div>
                            </div>
                        </div>
                        <div class="bg-slate-950 p-4 rounded-xl border border-slate-900">
                            <p class="text-xs font-semibold text-slate-400">Showroom A</p>
                            <div class="flex justify-between items-end mt-2">
                                <div>
                                    <span class="text-xs text-slate-500">Dwell:</span>
                                    <span id="zone-showroom-dwell" class="text-sm font-bold text-slate-200">0.0s</span>
                                </div>
                                <div class="text-right">
                                    <span class="text-xs text-slate-500">People:</span>
                                    <span id="zone-showroom-occ" class="text-sm font-bold text-indigo-400">0</span>
                                </div>
                            </div>
                        </div>
                        <div class="bg-slate-950 p-4 rounded-xl border border-slate-900">
                            <p class="text-xs font-semibold text-slate-400">Server Room Entrance</p>
                            <div class="flex justify-between items-end mt-2">
                                <div>
                                    <span class="text-xs text-slate-500">Dwell:</span>
                                    <span id="zone-server-dwell" class="text-sm font-bold text-slate-200">0.0s</span>
                                </div>
                                <div class="text-right">
                                    <span class="text-xs text-slate-500">People:</span>
                                    <span id="zone-server-occ" class="text-sm font-bold text-rose-500">0</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <script>
            let currentMode = 'adaptive';
            let ws = null;
            let chart = null;
            let performanceHistory = { cpu: [], gpu: [], vram: [], labels: [] };
            
            function setMode(mode) {
                currentMode = mode;
                const btnAdaptive = document.getElementById('mode-adaptive');
                const btnBaseline = document.getElementById('mode-baseline');
                
                if (mode === 'adaptive') {
                    btnAdaptive.className = 'py-1.5 text-xs font-medium rounded-lg bg-indigo-600 text-slate-100 transition-all';
                    btnBaseline.className = 'py-1.5 text-xs font-medium rounded-lg text-slate-400 hover:text-slate-200 transition-all';
                } else {
                    btnBaseline.className = 'py-1.5 text-xs font-medium rounded-lg bg-indigo-600 text-slate-100 transition-all';
                    btnAdaptive.className = 'py-1.5 text-xs font-medium rounded-lg text-slate-400 hover:text-slate-200 transition-all';
                }
            }

            function initChart() {
                const ctx = document.getElementById('perfChart').getContext('2d');
                chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            {
                                label: 'CPU %',
                                data: [],
                                borderColor: 'rgb(99, 102, 241)',
                                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                                tension: 0.2
                            },
                            {
                                label: 'GPU %',
                                data: [],
                                borderColor: 'rgb(244, 63, 94)',
                                backgroundColor: 'rgba(244, 63, 94, 0.1)',
                                tension: 0.2
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { min: 0, max: 100, grid: { color: 'rgba(51, 65, 85, 0.2)' } },
                            x: { grid: { display: false } }
                        },
                        plugins: {
                            legend: { labels: { color: '#94a3b8' } }
                        }
                    }
                });
            }

            function updatePerformanceChart(cpu, gpu) {
                if (!chart) return;
                
                const timeString = new Date().toLocaleTimeString();
                chart.data.labels.push(timeString);
                chart.data.datasets[0].data.push(cpu);
                chart.data.datasets[1].data.push(gpu);
                
                if (chart.data.labels.length > 20) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                    chart.data.datasets[1].data.shift();
                }
                chart.update();
            }

            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'status_update') {
                        document.getElementById('stat-occupancy').innerText = data.occupancy;
                        document.getElementById('stat-entries').innerText = data.total_in;
                        document.getElementById('stat-exits').innerText = data.total_out;
                        
                        const stateBadge = document.getElementById('state-badge');
                        stateBadge.innerText = data.state;
                        
                        // Style state badge dynamically
                        if (data.state === 'CRITICAL_EVENT') {
                            stateBadge.className = 'px-4 py-2 rounded-full text-lg font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30 animate-pulse';
                            document.getElementById('state-desc').innerText = 'CRITICAL ALERT! Specialist security models verifying details.';
                        } else if (data.state === 'SUSPICIOUS_ACTIVITY') {
                            stateBadge.className = 'px-4 py-2 rounded-full text-lg font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30';
                            document.getElementById('state-desc').innerText = 'Unusual behavior or restricted zone entry detected. Verifying.';
                        } else if (data.state === 'ZONE_ACTIVITY') {
                            stateBadge.className = 'px-4 py-2 rounded-full text-lg font-bold bg-indigo-500/20 text-indigo-400 border border-indigo-500/30';
                            document.getElementById('state-desc').innerText = 'Customer interaction zone active.';
                        } else if (data.state === 'NORMAL_ACTIVITY') {
                            stateBadge.className = 'px-4 py-2 rounded-full text-lg font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
                            document.getElementById('state-desc').innerText = 'System functioning normally. People in scene.';
                        } else if (data.state === 'IDLE') {
                            stateBadge.className = 'px-4 py-2 rounded-full text-lg font-bold bg-slate-800 text-slate-400 border border-slate-700';
                            document.getElementById('state-desc').innerText = 'No activity detected. Frame rate adapted to low power.';
                        }
                    }
                };
                
                ws.onclose = function() {
                    setTimeout(connectWebSocket, 3000);
                };
            }

            async function startStream() {
                const camera = document.getElementById('camera-select').value;
                const response = await fetch('/streams/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ camera_id: camera, mode: currentMode })
                });
                
                if (response.ok) {
                    document.getElementById('btn-start').classList.add('hidden');
                    document.getElementById('btn-stop').classList.remove('hidden');
                    
                    // Poll CPU and GPU metrics
                    startPollingMetrics(camera);
                } else {
                    const err = await response.json();
                    alert("Failed to start stream: " + err.detail);
                }
            }

            async function stopStream() {
                const camera = document.getElementById('camera-select').value;
                const response = await fetch(`/streams/stop?camera_id=${camera}`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    document.getElementById('btn-stop').classList.add('hidden');
                    document.getElementById('btn-start').classList.remove('hidden');
                    stopPollingMetrics();
                    
                    document.getElementById('state-badge').innerText = 'OFFLINE';
                    document.getElementById('state-badge').className = 'px-4 py-2 rounded-full text-lg font-bold bg-slate-800 text-slate-400 border border-slate-700';
                    document.getElementById('state-desc').innerText = 'No camera stream processing active.';
                }
            }

            let metricsInterval = null;
            function startPollingMetrics(camera) {
                if (metricsInterval) clearInterval(metricsInterval);
                metricsInterval = setInterval(async () => {
                    try {
                        const response = await fetch(`/performance?camera_id=${camera}`);
                        if (response.ok) {
                            const data = await response.json();
                            if (data.length > 0) {
                                const latest = data[0];
                                updatePerformanceChart(latest.cpu_utilization_pct, latest.gpu_utilization_pct);
                            }
                        }
                        
                        // Also poll zone details
                        const zoneResponse = await fetch('/analytics/zones');
                        if (zoneResponse.ok) {
                            const zoneData = await zoneResponse.json();
                            const camZones = zoneData[camera];
                            if (camZones && camZones.zones) {
                                if (camZones.zones.lobby_engagement_zone) {
                                    document.getElementById('zone-lobby-dwell').innerText = camZones.zones.lobby_engagement_zone.average_dwell_time.toFixed(1) + 's';
                                    document.getElementById('zone-lobby-occ').innerText = camZones.zones.lobby_engagement_zone.current_occupancy;
                                }
                                if (camZones.zones.product_display_zone_a) {
                                    document.getElementById('zone-showroom-dwell').innerText = camZones.zones.product_display_zone_a.average_dwell_time.toFixed(1) + 's';
                                    document.getElementById('zone-showroom-occ').innerText = camZones.zones.product_display_zone_a.current_occupancy;
                                }
                                if (camZones.zones.server_room_entrance) {
                                    document.getElementById('zone-server-dwell').innerText = camZones.zones.server_room_entrance.average_dwell_time.toFixed(1) + 's';
                                    document.getElementById('zone-server-occ').innerText = camZones.zones.server_room_entrance.current_occupancy;
                                }
                            }
                        }
                    } catch (e) {
                        console.error("Error polling metrics: ", e);
                    }
                }, 3000);
            }

            function stopPollingMetrics() {
                if (metricsInterval) {
                    clearInterval(metricsInterval);
                    metricsInterval = null;
                }
            }

            window.onload = function() {
                initChart();
                connectWebSocket();
            };
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_streams": list(running_streams.keys())
    }

@app.get("/cameras")
def get_cameras(db: Session = Depends(get_db)):
    # Synchronize configured cameras to database
    for cam in sys_config.cameras:
        repository.create_camera(
            db, 
            camera_id=cam.camera_id, 
            source=cam.source, 
            role=cam.role, 
            target_fps=cam.target_fps, 
            enabled=cam.enabled
        )
    return repository.get_cameras(db)

@app.get("/events")
def get_events(db: Session = Depends(get_db), limit: int = 100):
    db_events = repository.get_security_events(db, limit=limit)
    return [
        {
            "id": e.id,
            "camera_id": e.camera_id,
            "event_type": e.event_type,
            "track_id": e.track_id,
            "description": e.description,
            "timestamp": e.timestamp.isoformat(),
            "confidence": e.confidence,
            "evidence_clip_path": e.evidence_clip_path
        }
        for e in db_events
    ]

@app.get("/analytics/footfall")
def get_footfall(camera_id: Optional[str] = None, db: Session = Depends(get_db)):
    events = repository.get_counting_events(db, camera_id=camera_id, limit=500)
    
    total_in = sum(1 for e in events if e.direction == "in")
    total_out = sum(1 for e in events if e.direction == "out")
    
    return {
        "total_in": total_in,
        "total_out": total_out,
        "current_occupancy": max(0, total_in - total_out),
        "recent_crossings": [
            {
                "camera_id": e.camera_id,
                "line_name": e.line_name,
                "track_id": e.track_id,
                "direction": e.direction,
                "timestamp": e.timestamp.isoformat()
            }
            for e in events[:50]
        ]
    }

@app.get("/analytics/occupancy")
def get_occupancy(db: Session = Depends(get_db)):
    # Get all active streams current occupancy
    occupancies = {}
    for cid, job in running_streams.items():
        engine = job["engine"]
        occupancies[cid] = engine.occupancy_tracker.get_occupancy()
    return occupancies

@app.get("/analytics/zones")
def get_zones(db: Session = Depends(get_db)):
    zone_metrics = {}
    for cid, job in running_streams.items():
        engine = job["engine"]
        zone_metrics[cid] = engine.zone_analytics.get_all_metrics()
    return zone_metrics

@app.get("/analytics/dwell-time")
def get_dwell_time(zone_name: Optional[str] = None, db: Session = Depends(get_db)):
    visits = repository.get_zone_visits(db, zone_name=zone_name, limit=100)
    return [
        {
            "id": v.id,
            "camera_id": v.camera_id,
            "zone_name": v.zone_name,
            "track_id": v.track_id,
            "entry_time": v.entry_time.isoformat(),
            "exit_time": v.exit_time.isoformat() if v.exit_time else None,
            "dwell_duration": v.dwell_duration,
            "loitering_triggered": v.loitering_triggered
        }
        for v in visits
    ]

@app.get("/performance")
def get_performance(camera_id: str, db: Session = Depends(get_db)):
    metrics = repository.get_performance_metrics(db, camera_id=camera_id, limit=50)
    return [
        {
            "timestamp": m.timestamp.isoformat(),
            "cpu_utilization_pct": m.cpu_utilization_pct,
            "ram_utilization_pct": m.ram_utilization_pct,
            "gpu_utilization_pct": m.gpu_utilization_pct,
            "vram_allocated_mb": m.vram_allocated_bytes / (1024 * 1024),
            "processed_fps": m.processed_fps,
            "inference_latency_ms": m.inference_latency_ms,
            "e2e_latency_ms": m.e2e_latency_ms
        }
        for m in metrics
    ]

@app.get("/benchmarks")
def get_benchmarks(db: Session = Depends(get_db)):
    runs = repository.get_benchmark_runs(db, limit=20)
    return [
        {
            "id": r.id,
            "camera_id": r.camera_id,
            "timestamp": r.timestamp.isoformat(),
            "duration_seconds": r.duration_seconds,
            "baseline_metrics": r.baseline_metrics,
            "adaptive_metrics": r.adaptive_metrics,
            "report_path": r.report_path
        }
        for r in runs
    ]


# ==========================================
# Stream Control
# ==========================================

def run_engine_thread(engine: Any, camera_id: str, mode: str, max_frames: Optional[int]):
    """Wrapper function to run the engine in a background thread and handle database writes."""
    db: Session = SessionLocal()
    try:
        # Override save_results callback inside engine or run loop
        # We periodically capture metrics and save to DB
        last_save_time = time.time()
        
        # We subclass or adapt the run loop callback
        original_read = engine.video_source.read
        
        def intercepted_read():
            # Standard frame read
            success, frame, metadata = original_read()
            if success and metadata:
                nonlocal last_save_time
                current_time = time.time()
                
                # Check for new events and write to DB
                # Broadcaster integration: send websocket updates
                # We can broadcast occupancy changes
                async_loop_run(ws_manager.broadcast({
                    "type": "status_update",
                    "camera_id": camera_id,
                    "occupancy": engine.occupancy_tracker.get_occupancy(),
                    "total_in": engine.occupancy_tracker.total_entries,
                    "total_out": engine.occupancy_tracker.total_exits,
                    "state": getattr(engine, "trigger_manager", None).state if mode == "adaptive" else "BASELINE"
                }))
                
                # Periodically write performance metrics to DB (every 5 seconds)
                if current_time - last_save_time >= 5.0:
                    summary = engine.perf_monitor.update()
                    repository.create_performance_metric(
                        db,
                        camera_id=camera_id,
                        engine_mode=mode,
                        cpu_utilization_pct=summary["cpu_utilization_pct"],
                        ram_utilization_pct=summary["ram_utilization_pct"],
                        gpu_utilization_pct=summary["gpu_utilization_pct"],
                        vram_allocated_bytes=summary["vram_allocated_bytes"],
                        processed_fps=summary["processed_fps"],
                        inference_latency_ms=summary["average_inference_latency_ms"],
                        e2e_latency_ms=summary["average_e2e_latency_ms"]
                    )
                    last_save_time = current_time
                    
            return success, frame, metadata

        # Bind the intercepted read
        engine.video_source.read = intercepted_read
        engine.run(max_frames=max_frames)
        
        # Save final metrics summary
        summary = engine.get_summary()
        repository.create_performance_metric(
            db,
            camera_id=camera_id,
            engine_mode=mode,
            cpu_utilization_pct=summary["average_cpu_pct"],
            ram_utilization_pct=summary["average_ram_pct"],
            gpu_utilization_pct=summary["average_gpu_pct"],
            vram_allocated_bytes=summary["peak_vram_bytes"],
            processed_fps=summary["processed_fps"],
            inference_latency_ms=summary["average_inference_latency_ms"],
            e2e_latency_ms=summary["average_e2e_latency_ms"]
        )
    except Exception as e:
        logger.error(f"Error in running engine thread for camera {camera_id}: {e}", exc_info=True)
    finally:
        db.close()
        running_streams.pop(camera_id, None)
        logger.info(f"Stream thread for camera {camera_id} terminated.")

def async_loop_run(coro):
    """Helper to run async websocket broadcast from sync engine thread."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            loop.run_until_complete(coro)
    except Exception as e:
        # Fail silently if loop shutdown in progress
        pass

@app.post("/streams/start")
def start_stream(req: StreamStartRequest):
    if req.camera_id in running_streams:
        raise HTTPException(status_code=400, detail=f"Stream for camera {req.camera_id} is already running.")
        
    mode = req.mode.lower()
    if mode not in ["baseline", "adaptive"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'baseline' or 'adaptive'.")
        
    # Check if video source exists
    cam_cfg = sys_config.get_camera_config(req.camera_id)
    if not cam_cfg:
        raise HTTPException(status_code=404, detail=f"Camera {req.camera_id} not found in configurations.")
        
    # Instantiate engine
    if mode == "baseline":
        engine = BaselineEngine(system_config=sys_config, camera_id=req.camera_id)
    else:
        engine = AdaptiveEngine(system_config=sys_config, camera_id=req.camera_id)
        
    # Spawn background thread
    t = threading.Thread(
        target=run_engine_thread,
        args=(engine, req.camera_id, mode, req.max_frames),
        daemon=True
    )
    running_streams[req.camera_id] = {
        "thread": t,
        "engine": engine,
        "mode": mode
    }
    t.start()
    
    logger.info(f"Started background engine thread for {req.camera_id} in {mode.upper()} mode.")
    return {"status": "started", "camera_id": req.camera_id, "mode": mode}

@app.post("/streams/stop")
def stop_stream(camera_id: str):
    if camera_id not in running_streams:
        raise HTTPException(status_code=404, detail=f"No active stream running for camera {camera_id}.")
        
    job = running_streams[camera_id]
    engine = job["engine"]
    
    # Signalling stop
    engine.video_source.stop()
    
    # Wait briefly for thread exit
    job["thread"].join(timeout=2.0)
    
    return {"status": "stopped", "camera_id": camera_id}


# ==========================================
# WebSocket Endpoint
# ==========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive by receiving heartbeats or client messages
            data = await websocket.receive_text()
            # Echo back or process commands if any
            await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
