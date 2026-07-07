import os
import time
import json
import logging
import psutil
import pandas as pd
from typing import Dict, Any, List

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors system resource utilization (CPU, RAM, GPU, VRAM) and pipeline latencies."""
    
    def __init__(self, camera_id: str, monitor_interval_seconds: float = 1.0):
        self.camera_id = camera_id
        self.monitor_interval = monitor_interval_seconds
        
        # System Resource History
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Pipeline Counters
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_skipped = 0
        self.detector_executions = 0
        self.tracker_updates = 0
        self.specialist_executions: Dict[str, int] = {}
        self.events_detected = 0
        self.false_alerts = 0
        self.missed_events = 0
        
        # Latency trackers
        self.inference_latencies: List[float] = []
        self.e2e_latencies: List[float] = []
        
        # Timestamps
        self.start_time = time.time()
        self.last_monitor_time = self.start_time
        
        # Initialize NVML
        self.nvml_initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("NVML successfully initialized for GPU monitoring.")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}. GPU monitoring disabled.")
        else:
            logger.info("pynvml package not available. GPU monitoring disabled.")

    def record_frame_received(self):
        self.frames_received += 1

    def record_frame_processed(self):
        self.frames_processed += 1

    def record_frame_skipped(self):
        self.frames_skipped += 1

    def record_detector_execution(self):
        self.detector_executions += 1

    def record_tracker_update(self):
        self.tracker_updates += 1

    def record_specialist_execution(self, model_name: str):
        self.specialist_executions[model_name] = self.specialist_executions.get(model_name, 0) + 1

    def record_event_detected(self):
        self.events_detected += 1

    def record_latency(self, inference_time_sec: float, e2e_time_sec: float):
        """Records inference and end-to-end processing latency for a frame."""
        self.inference_latencies.append(inference_time_sec)
        self.e2e_latencies.append(e2e_time_sec)
        
        # Keep list size bounded to last 1000 items
        if len(self.inference_latencies) > 1000:
            self.inference_latencies.pop(0)
            self.e2e_latencies.pop(0)

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Queries GPU, VRAM, and temperature metrics via NVML."""
        metrics = {
            "gpu_utilization_pct": 0.0,
            "vram_allocated_bytes": 0.0,
            "vram_reserved_bytes": 0.0,
            "vram_total_bytes": 0.0,
            "gpu_temperature_c": 0.0
        }
        
        if not self.nvml_initialized:
            return metrics
            
        try:
            # Query index 0 device (RTX 5060 Ti in this workspace)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # 1. GPU Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["gpu_utilization_pct"] = float(util.gpu)
            
            # 2. VRAM Allocation (Memory info)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["vram_allocated_bytes"] = float(mem_info.used)
            metrics["vram_total_bytes"] = float(mem_info.total)
            metrics["vram_reserved_bytes"] = float(mem_info.total - mem_info.free)
            
            # 3. GPU Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics["gpu_temperature_c"] = float(temp)
            
        except Exception as e:
            logger.debug(f"Failed to fetch GPU metrics from NVML: {e}")
            
        return metrics

    def update(self) -> Dict[str, Any]:
        """Queries resource utilization and appends it to history if interval exceeded.
        
        Returns:
            Dict containing the latest resource metrics
        """
        current_time = time.time()
        
        # Get system-wide metrics
        cpu_pct = psutil.cpu_percent()
        ram_info = psutil.virtual_memory()
        
        # Get GPU metrics
        gpu_metrics = self.get_gpu_metrics()
        
        # Compute latencies
        avg_inf = sum(self.inference_latencies) / len(self.inference_latencies) if self.inference_latencies else 0.0
        avg_e2e = sum(self.e2e_latencies) / len(self.e2e_latencies) if self.e2e_latencies else 0.0
        
        # Calculate processed FPS
        elapsed = current_time - self.start_time
        processed_fps = self.frames_processed / elapsed if elapsed > 0 else 0.0
        
        metrics = {
            "timestamp": current_time,
            "elapsed_seconds": elapsed,
            "cpu_utilization_pct": cpu_pct,
            "ram_utilization_pct": ram_info.percent,
            "ram_used_bytes": ram_info.used,
            "ram_total_bytes": ram_info.total,
            "processed_fps": processed_fps,
            "average_inference_latency_ms": avg_inf * 1000.0,
            "average_e2e_latency_ms": avg_e2e * 1000.0,
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "detector_executions": self.detector_executions,
            "tracker_updates": self.tracker_updates,
            "events_detected": self.events_detected,
            **gpu_metrics
        }
        
        # Append to history at interval
        if current_time - self.last_monitor_time >= self.monitor_interval:
            self.metrics_history.append(metrics)
            self.last_monitor_time = current_time
            
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Calculates mean, peak, and final metrics for the monitored session."""
        if not self.metrics_history:
            return {}
            
        df = pd.DataFrame(self.metrics_history)
        
        avg_gpu = float(df["gpu_utilization_pct"].mean()) if "gpu_utilization_pct" in df else 0.0
        peak_vram = float(df["vram_allocated_bytes"].max()) if "vram_allocated_bytes" in df else 0.0
        avg_cpu = float(df["cpu_utilization_pct"].mean()) if "cpu_utilization_pct" in df else 0.0
        avg_ram = float(df["ram_utilization_pct"].mean()) if "ram_utilization_pct" in df else 0.0
        
        avg_inf = sum(self.inference_latencies) / len(self.inference_latencies) if self.inference_latencies else 0.0
        avg_e2e = sum(self.e2e_latencies) / len(self.e2e_latencies) if self.e2e_latencies else 0.0
        
        elapsed = time.time() - self.start_time
        processed_fps = self.frames_processed / elapsed if elapsed > 0 else 0.0
        
        # Calculate event metrics (precision, recall, etc.)
        # F1 = 2 * P * R / (P + R)
        total_eval = self.events_detected + self.missed_events
        precision = self.events_detected / (self.events_detected + self.false_alerts) if (self.events_detected + self.false_alerts) > 0 else 0.0
        recall = self.events_detected / total_eval if total_eval > 0 else 0.0
        f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "camera_id": self.camera_id,
            "duration_seconds": elapsed,
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "detector_executions": self.detector_executions,
            "tracker_updates": self.tracker_updates,
            "specialist_executions": self.specialist_executions,
            "events_detected": self.events_detected,
            "false_alerts": self.false_alerts,
            "missed_events": self.missed_events,
            "event_precision": precision,
            "event_recall": recall,
            "f1_score": f1_score,
            "average_cpu_pct": avg_cpu,
            "average_ram_pct": avg_ram,
            "average_gpu_pct": avg_gpu,
            "peak_vram_bytes": peak_vram,
            "average_inference_latency_ms": avg_inf * 1000.0,
            "average_e2e_latency_ms": avg_e2e * 1000.0,
            "processed_fps": processed_fps
        }

    def save_results(self, output_dir: str, prefix: str = "benchmark"):
        """Saves metrics history as CSV and summary as JSON to the output directory."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed history to CSV
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            csv_path = os.path.join(output_dir, f"{prefix}_{self.camera_id}_{timestamp_str}_history.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Performance history saved to {csv_path}")
            
        # Save summary to JSON
        summary = self.get_summary()
        json_path = os.path.join(output_dir, f"{prefix}_{self.camera_id}_{timestamp_str}_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Performance summary saved to {json_path}")

    def __del__(self):
        """Cleanup NVML on destruction."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml_initialized = False
