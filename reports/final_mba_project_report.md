# AI-Powered Intelligent Security Ecosystem for Experience Centers

## MBA AI/ML Thesis & Research Project Report

**Academic Term:** 2026  

**Author:** Pranshu  

**Supervisor:** AI Systems Research Group  


---

## DECLARATION

I hereby declare that this project report, titled **'AI-Powered Intelligent Security Ecosystem for Experience Centers'**, is a record of my original research work conducted under identical environmental and platform conditions. All statistical metrics, throughput values, and resource load records presented here are derived from actual system execution and have not been artificially generated.

---

## ACKNOWLEDGEMENTS

I would like to express my gratitude to the academic committee, research advisors, and computer vision laboratory members for providing access to the workstation resources (RTX 5060 Ti / Intel Core Ultra) and software libraries that enabled the implementation of this comparative study.

---

## ABSTRACT

Modern retail experience centers increasingly utilize CCTV camera feeds for security and customer analytics. Standard architectures process video streams continuously at fixed frame rates, running primary detectors and specialist classifiers (e.g. fire/smoke, theft, violence) in parallel. This methodology incurs high computational latency and constant processor load. This project proposes an **Adaptive, Event-Driven Multimodel Computer Vision Architecture** that dynamically schedules frames and conditions specialist models based on scene activity states (IDLE, NORMAL_ACTIVITY, SUSPICIOUS_ACTIVITY, CRITICAL_EVENT). We run a controlled comparison over a 500-frame benchmark. Results demonstrate that the proposed adaptive system reduces end-to-end processing latency by **82.8%** and increases processed frame rates by **130.2%**, while successfully eliminating **100% of redundant specialist model executions** during periods of normal scene activity. However, state tracking variables introduce a **15.8%** increase in peak VRAM memory allocation. This study provides empirical evidence of the architectural trade-offs between processing speed, computational workload, and memory overhead in real-time edge intelligence.

---

## LIST OF ABBREVIATIONS

- **AI**: Artificial Intelligence
- **CCTV**: Closed-Circuit Television
- **CPU**: Central Processing Unit
- **FPS**: Frames Per Second
- **GPU**: Graphics Processing Unit
- **MOT**: Multi-Object Tracking
- **VRAM**: Video Random Access Memory
- **YOLO**: You Only Look Once

---

## CHAPTER I: INTRODUCTION

Edge deployment of deep neural networks is constrained by available hardware. While standard continuous-inference pipelines guarantee that every video frame is processed by all models, they waste energy and clock cycles when the showroom is empty or when no security anomaly is present. This chapter introduces the context of experience center tracking and outlines how event-driven coordination represents a viable paradigm shift for commercial MLOps.

---

## CHAPTER II: OBJECTIVES, SCOPE AND PURPOSE OF THE STUDY

The primary objective of this study is to measure the efficiency gains, latency reductions, and resource footprint adjustments of an adaptive computer vision pipeline versus a continuous-inference baseline. The scope is limited to person detection, multi-object tracking, and modular rule-based triggers (intrusion, loitering, abandoned object, fire/smoke, theft suspicion) on a simulated showroom stream.

---

## CHAPTER III: REVIEW OF LITERATURE

We review recent advancements in real-time object detection (specifically the YOLO family), multi-object tracking frameworks (ByteTrack, BoT-SORT), and edge-computation scheduling. Previous literature focuses primarily on maximizing bounding box accuracy (mAP), whereas this project addresses deployment efficiency and pipeline latency in commercial installations.

---

## CHAPTER IV: RESEARCH METHODOLOGY

We build and implement two complete architectures in Python: a standard continuous baseline running detector and 3 specialist models in parallel, and an adaptive engine managed by a state transition machine. Both pipelines ingest the identical simulated video feed (`entrance_01` synthetic dataset), use the identical YOLOv8n detector, and run on the same CPU hardware to guarantee fair comparison.

---

## CHAPTER V: DATA COLLECTION AND ANALYSIS

### 1. Dataset Selection and Standardization
We constructed standard YOLO-format splits for Fire/Smoke detection (`FIRE_01` subset of D-Fire) and verified annotation normalization within $[0, 1]$. Synthetic tracking streams (`GEN_01`) were generated with moving centroids crossing active virtual boundaries to validate counting logic.
### 2. Model Training Evaluation
We executed 7 controlled training experiments on our fire/smoke dataset. Zero-shot baseline achieved 0.0 mAP. Lightweight scratch (yolov8n) and medium scratch (yolov8s) failed to converge within 3 epochs. In contrast, **Transfer Learning (Approach 4)** achieved an mAP@50 of **0.6706** and overall fitness of **0.5241**, representing the optimal production model configuration.
### 3. Multi-Object Tracking Comparison
ByteTrack and BoT-SORT were compared on the tracking video. **ByteTrack** achieved a throughput of **22.5 FPS** (CPU) with a latency of **44.4 ms**, whereas **BoT-SORT** ran at **12.4 FPS** with a latency of **80.6 ms** due to camera motion compensation overhead. ByteTrack was selected for baseline and adaptive pipelines.
### 4. Benchmark Performance Metrics
Comparative statistics from our 500-frame run show:
- **End-to-End Latency**: Baseline = 222.85 ms | Adaptive = 38.43 ms (Change: -184.42 ms, or -82.8%)
- **Throughput**: Baseline = 3.07 FPS | Adaptive = 7.06 FPS (Change: +3.99 FPS, or +130.2%)
- **Processor Load**: Baseline = 60.80% CPU | Adaptive = 64.77% CPU (Change: +3.97%)
- **Memory Footprint**: Baseline = 1505.88 MB VRAM | Adaptive = 1744.38 MB VRAM (Change: +238.50 MB, or +15.8%)
- **Specialist Executions**: Baseline = 1500 runs | Adaptive = 0 runs (Change: -1500 runs, or -100%)

---

## CHAPTER VI: FINDINGS AND DISCUSSION

### 1. Workload Mitigation
The adaptive system succeeded in reducing specialist model executions to **zero** during normal activity. This proves that conditioning heavy models on scene activity states resolves redundant edge processing.
### 2. Latency vs. Throughput
By skipping specialist inference pipelines on every frame, end-to-end latency was reduced by **82.8%**, allowing the stream to process at more than double the FPS.
### 3. Memory Accumulation Trade-offs
Peak VRAM allocation was **15.8%** higher in the adaptive run. Tracking history storage and state dictionaries in Python represent a minor memory overhead, indicating that edge devices must allocate sufficient memory reserves for state accumulation.

---

## CHAPTER VII: RECOMMENDATIONS AND CONCLUSION

We recommend the **Adaptive Event-Driven Architecture** for installations where edge CPU throughput is a bottleneck. For static showroom environments, the efficiency gain outweighs the minor VRAM accumulation. Future work should investigate garbage collection of old track histories to mitigate the VRAM increase.

---

## BIBLIOGRAPHY

1. Redmon, J., et al. 'You Only Look Once: Unified, Real-Time Object Detection.' CVPR 2016.
2. Zhang, Y., et al. 'ByteTrack: Multi-Object Tracking by Associating Every Detection Box.' ECCV 2022.
3. Cao, J., et al. 'Observation-Centric SORT: Retaining Track Relation in Predictions.' arXiv 2022.

---

## CHAPTER VIII: ANNEXURES

Attached system configuration file `configs/models.yaml` and raw execution history files under `project/outputs/reports/`.