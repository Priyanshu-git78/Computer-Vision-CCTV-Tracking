import os
import sys
import argparse
import logging
import time
import json
import pandas as pd
from typing import Optional

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import SystemConfig
from src.orchestration.baseline_engine import BaselineEngine
from src.orchestration.adaptive_engine import AdaptiveEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_comparison(config_dir: str, camera: str, max_frames: Optional[int], output_dir: str):
    logger.info(f"===== Initializing Benchmark for Camera: {camera} =====")
    cfg = SystemConfig(config_dir=config_dir)
    
    # Ensure outputs directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Run Baseline Engine
    logger.info("----- Starting Baseline Engine Run -----")
    # Clean cache/GC before run to ensure fair VRAM measurement
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    baseline = BaselineEngine(system_config=cfg, camera_id=camera)
    baseline.run(max_frames=max_frames)
    baseline_summary = baseline.get_summary()
    baseline.save_results(output_dir)
    
    # 2. Run Adaptive Engine
    logger.info("----- Starting Adaptive Engine Run -----")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    adaptive = AdaptiveEngine(system_config=cfg, camera_id=camera)
    adaptive.run(max_frames=max_frames)
    adaptive_summary = adaptive.get_summary()
    adaptive.save_results(output_dir)
    
    # 3. Compare Results
    logger.info("===== Processing Benchmark Results =====")
    metrics_to_compare = [
        ("average_cpu_pct", "Average CPU Utilization (%)", "{:.2f}%"),
        ("average_ram_pct", "Average RAM Utilization (%)", "{:.2f}%"),
        ("average_gpu_pct", "Average GPU Utilization (%)", "{:.2f}%"),
        ("peak_vram_bytes", "Peak VRAM Allocation (MB)", lambda x: f"{x / (1024 * 1024):.2f} MB"),
        ("average_inference_latency_ms", "Average Inference Latency (ms)", "{:.2f} ms"),
        ("average_e2e_latency_ms", "Average End-to-End Latency (ms)", "{:.2f} ms"),
        ("processed_fps", "Processed FPS", "{:.2f}"),
        ("frames_processed", "Total Frames Processed", "{:d}"),
        ("detector_executions", "Total Detector Executions", "{:d}"),
        ("events_detected", "Total Events Detected", "{:d}"),
        ("false_alerts", "False Alerts", "{:d}"),
        ("missed_events", "Missed Events", "{:d}"),
        ("event_precision", "Event Precision", "{:.2f}"),
        ("event_recall", "Event Recall", "{:.2f}"),
        ("f1_score", "F1-Score", "{:.2f}"),
    ]
    
    report_rows = []
    for key, label, formatter in metrics_to_compare:
        val_b = baseline_summary.get(key, 0.0)
        val_a = adaptive_summary.get(key, 0.0)
        
        # Calculate difference (Adaptive - Baseline)
        diff = val_a - val_b
        
        # Formatting values
        if callable(formatter):
            str_b = formatter(val_b)
            str_a = formatter(val_a)
            str_diff = formatter(diff)
        else:
            str_b = formatter.format(val_b)
            str_a = formatter.format(val_a)
            str_diff = formatter.format(diff)
            
        report_rows.append({
            "Metric": label,
            "Baseline": str_b,
            "Adaptive": str_a,
            "Difference": str_diff
        })
        
    df_report = pd.DataFrame(report_rows)
    
    # Generate Markdown Table
    md_table = df_report.to_markdown(index=False)
    print("\n" + "="*80 + "\nBENCHMARK REPORT:\n" + "="*80 + f"\n{md_table}\n" + "="*80)
    
    # Save Report
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"report_{camera}_{timestamp_str}.md")
    with open(report_path, "w") as f:
        f.write(f"# Benchmark Comparison Report: Baseline vs Adaptive\n")
        f.write(f"- Camera: {camera}\n")
        f.write(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(md_table)
        f.write("\n\n*Note: Specialist model executions are activated on-demand in the Adaptive Engine, reducing peak compute overhead.*")
        
    logger.info(f"Comparison report written successfully to {report_path}")
    
    # Also save comparison as JSON
    comparison_data = {
        "camera_id": camera,
        "timestamp": time.time(),
        "baseline": baseline_summary,
        "adaptive": adaptive_summary
    }
    json_path = os.path.join(output_dir, f"report_{camera}_{timestamp_str}.json")
    with open(json_path, "w") as f:
        json.dump(comparison_data, f, indent=4)
        
    return report_path

def main():
    default_config_dir = os.path.join(project_root, "configs")
    default_output_dir = os.path.join(project_root, "outputs/reports")
    
    parser = argparse.ArgumentParser(description="Run Benchmark comparing Baseline vs Adaptive Engines")
    parser.add_argument("--config-dir", default=default_config_dir, help="Path to config directory")
    parser.add_argument("--camera", default="entrance_01", help="Camera ID to run")
    parser.add_argument("--max-frames", type=int, default=100, help="Number of frames to process for benchmark")
    parser.add_argument("--output-dir", default=default_output_dir, help="Output directory for reports")
    args = parser.parse_args()

    # Verify input file path before starting benchmark
    cfg = SystemConfig(config_dir=args.config_dir)
    cam_cfg = cfg.get_camera_config(args.camera)
    if cam_cfg and not os.path.exists(cam_cfg.source):
        logger.warning(
            f"Source file '{cam_cfg.source}' for camera '{args.camera}' does not exist. "
            f"The ingestion system will automatically fall back to generating synthetic frames from "
            f"ultralytics/assets/bus.jpg to run the comparative benchmark."
        )
        
    run_comparison(
        config_dir=args.config_dir,
        camera=args.camera,
        max_frames=args.max_frames,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
