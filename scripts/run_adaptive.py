import os
import sys
import argparse
import logging

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import SystemConfig
from src.orchestration.adaptive_engine import AdaptiveEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    default_config_dir = os.path.join(project_root, "configs")
    default_output_dir = os.path.join(project_root, "outputs")
    
    parser = argparse.ArgumentParser(description="Run the Adaptive Event-Driven Inference Pipeline")
    parser.add_argument("--config-dir", default=default_config_dir, help="Path to config directory")
    parser.add_argument("--camera", default="entrance_01", help="Camera ID to run")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--output-dir", default=default_output_dir, help="Output directory for reports")
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config_dir}")
    cfg = SystemConfig(config_dir=args.config_dir)
    
    # Check if video file exists, otherwise warn
    cam_cfg = cfg.get_camera_config(args.camera)
    if cam_cfg and not os.path.exists(cam_cfg.source):
        logger.warning(
            f"Video source '{cam_cfg.source}' does not exist. "
            f"Please verify the file exists or copy a sample video to that location."
        )
        
    engine = AdaptiveEngine(system_config=cfg, camera_id=args.camera)
    engine.run(max_frames=args.max_frames)
    
    # Save results
    engine.save_results(args.output_dir)
    logger.info(f"Adaptive run completed. Metrics saved to {args.output_dir}")

if __name__ == "__main__":
    main()
