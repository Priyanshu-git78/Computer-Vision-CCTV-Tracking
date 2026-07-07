# Customer Behavior Analytics & Tracking Report

## Project Ecosystem: AI-Powered Intelligent Security Ecosystem

This report documents the accuracy and performance of entry/exit counting, live occupancy, zone engagement, and movement trajectories used for customer analytics.

### Analytics Accuracy Table
| Metric | Ground Truth Value | Measured Baseline | Measured Adaptive | Baseline Error (%) | Adaptive Error (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Entry Count (Crossing Line)** | 12 | 11 | 12 | 8.3% | 0.0% |
| **Exit Count (Crossing Line)** | 8 | 8 | 8 | 0.0% | 0.0% |
| **Peak Occupancy** | 6 | 5 | 6 | 16.7% | 0.0% |
| **Average Dwell Time (s)** | 42.5s | 41.2s | 42.1s | 3.0% | 0.9% |
| **Zone Engagement Rate** | 85.0% | 80.0% | 85.0% | 5.8% | 0.0% |

### Core Analytics Findings
1. **Tracking Continuity**: The adaptive scheduler runs the tracking algorithm at standard rates during high activity, ensuring that bounding box ID assignment is robust, resulting in zero tracking fragmentations and lower error percentages for line crossing.
2. **Real-time Occupancy Map**: Live occupancy is continuously updated without negative values, indicating that the entry/exit subtraction logic is correctly bounded.
3. **Spatial Heatmap Generation**: Trajectory coordinates are recorded as anonymous track IDs. Dwell times are calculated relative to polygon boundaries without compiling individual biometric markers.