import os
import logging
import time
from typing import Dict, Any, Optional, List
from ultralytics import YOLO
from src.config import SystemConfig

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Manages lazy loading, activation, and logging of specialist AI models to optimize VRAM."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Loaded models storage. Key: model_name, Value: YOLO instance
        self.models: Dict[str, YOLO] = {}
        
        # Track active status and last activation timestamps
        self.activation_history: List[Dict[str, Any]] = []
        self.active_models: Dict[str, float] = {}

    def get_model(self, model_name: str) -> Optional[YOLO]:
        """Retrieves or lazily loads the requested specialist model."""
        if model_name in self.models:
            return self.models[model_name]
            
        spec_cfg = self.config.models.specialist_models.get(model_name)
        if not spec_cfg:
            logger.error(f"[ModelRegistry] Model '{model_name}' not defined in config.")
            return None
            
        if not spec_cfg.enabled:
            logger.warning(f"[ModelRegistry] Model '{model_name}' is disabled in configuration.")
            return None
            
        # Log model activation
        t_start = time.time()
        path_to_load = spec_cfg.path
        
        # Fallback if path doesn't exist (simulates workload using base detector)
        if not os.path.exists(path_to_load):
            logger.warning(
                f"[ModelRegistry] Specialist weights for '{model_name}' not found at {path_to_load}. "
                f"Falling back to primary detector {self.config.models.primary_detector.path} for simulation."
            )
            path_to_load = self.config.models.primary_detector.path
            
        logger.info(f"[ModelRegistry] DYNAMIC ACTIVATION: Loading model '{model_name}' from {path_to_load}")
        try:
            model = YOLO(path_to_load)
            self.models[model_name] = model
            load_time = time.time() - t_start
            
            # Record activation log
            activation_event = {
                "model_name": model_name,
                "action": "load_and_activate",
                "timestamp": t_start,
                "load_time_seconds": load_time,
                "weights_path": path_to_load
            }
            self.activation_history.append(activation_event)
            self.active_models[model_name] = t_start
            
            return model
        except Exception as e:
            logger.error(f"[ModelRegistry] Failed to dynamically load specialist '{model_name}': {e}")
            return None

    def release_model(self, model_name: str):
        """Unloads a model from memory to free up VRAM when no longer needed."""
        if model_name in self.models:
            logger.info(f"[ModelRegistry] DEACTIVATION: Unloading model '{model_name}' to free VRAM.")
            self.models.pop(model_name)
            activation_start = self.active_models.pop(model_name, time.time())
            active_duration = time.time() - activation_start
            
            deactivation_event = {
                "model_name": model_name,
                "action": "unload_deactivate",
                "timestamp": time.time(),
                "active_duration_seconds": active_duration
            }
            self.activation_history.append(deactivation_event)
            
            # Request Python garbage collection and empty CUDA cache
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_activation_logs(self) -> List[Dict[str, Any]]:
        """Returns the history of model activation and deactivation events."""
        return self.activation_history
