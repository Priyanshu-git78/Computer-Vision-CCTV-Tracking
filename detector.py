"""
ONNX Runtime person detector.

Zero ultralytics imports at runtime. AutoBackend is never instantiated.
Uses pure NumPy for all pre/post-processing and NMS.

Prerequisites:
    Run export_model.py ONCE to convert your .pt model:
        python export_model.py --model yolo11x.pt --imgsz 1280
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError(
        "onnxruntime is required.\n"
        "  CPU : pip install onnxruntime\n"
        "  GPU : pip install onnxruntime-gpu"
    ) from exc


def _prepare_onnxruntime_cuda() -> None:
    """Load CUDA/cuDNN DLLs from the local PyTorch install when available."""
    try:
        import torch
    except ImportError:
        return

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if hasattr(os, "add_dll_directory") and torch_lib_dir.exists():
        os.add_dll_directory(str(torch_lib_dir))
    try:
        ort.preload_dlls()
    except Exception:
        pass


# ────────────────────────── NMS (pure NumPy) ──────────────────────────────

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Non-maximum suppression with no external dependencies."""
    if boxes.size == 0:
        return np.empty((0,), dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou  = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thresh]

    return np.array(keep, dtype=int)


# ────────────────────────── Detector ──────────────────────────────────────

class PersonDetector:
    """
    Drop-in replacement for the old ultralytics-based PersonDetector.

    The public interface is identical:
        detections = detector.detect(frame)   -> list[dict[str, Any]]
        summary    = detector.runtime_summary()
    """

    PERSON_CLASS_ID = 0   # COCO index for 'person'

    def __init__(
        self,
        model_path: str = "yolov8n.onnx",
        input_size: int = 640,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        # Legacy kwargs accepted but ignored (half/compile are ONNX-irrelevant)
        use_half: bool = False,
        augment_inference: bool = False,
        compile_mode: str | bool = False,
    ) -> None:
        _prepare_onnxruntime_cuda()
        self.input_size   = input_size
        self.confidence   = confidence
        self.iou_threshold = iou_threshold
        self.model_path   = model_path
        requested_device = device.lower()

        # ── Resolve ONNX file path (auto-find .onnx if .pt was given) ──
        onnx_path = self._resolve_onnx_path(model_path)

        # ── Pick best execution provider ──
        available = ort.get_available_providers()
        providers: list[str] = []
        if requested_device == "auto" or "cuda" in requested_device:
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 0   # use all available cores

        self._session   = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)
        self._inp_name  = self._session.get_inputs()[0].name
        self._provider  = self._session.get_providers()[0]
        if "cuda" in requested_device and "CUDA" not in self._provider:
            raise RuntimeError(
                "CUDA inference was requested, but ONNX Runtime could not activate the "
                "CUDAExecutionProvider. Ensure the GPU runtime dependencies are installed."
            )

        self._device_label = "GPU (CUDA)" if "CUDA" in self._provider else "CPU"
        self._warmup()

        print(
            f"[Detector] Ready | {self._device_label} | ONNX Runtime "
            f"| conf={confidence} iou={iou_threshold} imgsz={input_size}"
        )

    # ──────────────────────── Public API ──────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run inference on a BGR frame. Returns list[{bbox, confidence, class_id}]."""
        tensor, r, pad_t, pad_l, h0, w0 = self._preprocess(frame)
        raw = self._session.run(None, {self._inp_name: tensor})[0]   # (1, 84, 8400)
        return self._postprocess(raw, r, pad_t, pad_l, h0, w0)

    def runtime_summary(self) -> dict[str, Any]:
        """Mimics the original interface expected by main.py and app UI."""
        return {
            "device":            self._device_label,
            "device_name":       self._device_label,
            "use_half":          False,
            "input_size":        self.input_size,
            "model_name":        Path(self.model_path).name,
            "augment_inference": False,
            "compile_mode":      False,
        }

    # ──────────────────────── Preprocessing ───────────────────────────────

    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, int, int, int, int]:
        """Letterbox resize -> BGR->RGB -> float32 NCHW [0,1]."""
        h0, w0 = frame.shape[:2]
        r = self.input_size / max(h0, w0)
        new_w, new_h = int(w0 * r), int(h0 * r)

        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

        dw, dh = self.input_size - new_w, self.input_size - new_h
        pad_t, pad_l = dh // 2, dw // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_t, dh - pad_t, pad_l, dw - pad_l,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        img = padded[:, :, ::-1].transpose(2, 0, 1)          # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[np.newaxis], r, pad_t, pad_l, h0, w0       # (1,3,H,W)

    # ──────────────────────── Postprocessing ──────────────────────────────

    def _postprocess(
        self,
        raw: np.ndarray,
        r: float,
        pad_t: int,
        pad_l: int,
        h0: int,
        w0: int,
    ) -> list[dict[str, Any]]:
        """
        Parse raw ONNX output (1, 84, 8400) -> list of detection dicts.
        Compatible with the dict schema expected by the rest of the pipeline.
        """
        preds = raw[0].T                          # (8400, 84)
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        person_scores   = preds[:, 4 + self.PERSON_CLASS_ID]

        mask = person_scores > self.confidence
        if not mask.any():
            return []

        person_scores = person_scores[mask]
        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]

        # cx/cy/bw/bh -> x1/y1/x2/y2 (in letterboxed space)
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep = _nms(boxes, person_scores, self.iou_threshold)
        if keep.size == 0:
            return []

        boxes  = boxes[keep][:100]
        scores = person_scores[keep][:100]

        # Undo letterbox: remove padding, scale back to original coords
        boxes[:, [0, 2]] -= pad_l
        boxes[:, [1, 3]] -= pad_t
        boxes /= r
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)
        boxes = boxes.astype(int)

        detections: list[dict[str, Any]] = []
        for box, score in zip(boxes.tolist(), scores.tolist()):
            detections.append({
                "bbox":       box,
                "confidence": float(score),
                "class_id":   self.PERSON_CLASS_ID,
            })
        return detections

    # ──────────────────────── Helpers ─────────────────────────────────────

    @staticmethod
    def _resolve_onnx_path(model_path: str) -> Path:
        """
        If a .pt path is given, look for the corresponding .onnx file.
        Raises FileNotFoundError with a helpful message if not found.
        """
        p = Path(model_path)
        if p.suffix.lower() == ".onnx":
            if not p.exists():
                raise FileNotFoundError(
                    f"ONNX model not found: {p}\n"
                    "Run export_model.py first:  python export_model.py"
                )
            return p

        # .pt supplied -> look for .onnx sibling
        onnx_candidate = p.with_suffix(".onnx")
        if onnx_candidate.exists() and onnx_candidate.stat().st_size > 1_000_000:
            print(f"[Detector] Auto-resolved .pt -> {onnx_candidate}")
            return onnx_candidate

        raise FileNotFoundError(
            f"No valid ONNX model found for '{model_path}'.\n"
            f"Run:  python export_model.py --model {model_path}"
        )

    def _warmup(self) -> None:
        """One silent forward pass to JIT the graph before real inference."""
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        try:
            self._session.run(None, {self._inp_name: dummy})
        except Exception:
            pass   # warmup is optional
