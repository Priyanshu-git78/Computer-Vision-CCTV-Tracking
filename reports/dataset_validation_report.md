# Dataset Validation & Quality Control Report

## Project Ecosytem: AI-Powered Intelligent Security Ecosystem

**Date:** 2026-07-07 22:47:08

This report details the integrity checks, file validation, class alignment, and structural metrics for all configured data sources in our comparative baseline vs. adaptive benchmark study.

## Quality Control Rules & Guardrails
1. **Coordinate Boundary Check**: Bounding box centers, widths, and heights must be normalized within $[0, 1]$. Any out-of-bounds coordinates flag the sample as anomalous.
2. **Duplicate Image Detection**: MD5 hash values are compared across all split directories to prevent spatial data leakage.
3. **Video Codec Verification**: Raw video sources are opened using OpenCV. Frame indices, dimensions, and color space maps are validated.
4. **Class Alignment Check**: Label indices must correspond exactly with the classes declared in standard configurations.

## Validation Results Summary

| Dataset ID | Dataset Name | Source | Status | Samples/Frames | Malformed Labels | Duplicates | Empty Labels |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GEN_01 (TownCentre) | (TownCentre) | Synthetic (Warped Grid Simulation) | **VALIDATED** | 150 | 0 | 0 | 0 |
| FIRE_01 (D-Fire) | (D-Fire) | Local Generation (CCTV Simulator) | **VALIDATED** | 65 | 0 | 0 | 0 |
| THEFT_01 (UCF-Shoplifting) | (UCF-Shoplifting) | Synthetic (Showroom Interactive Loop) | **VALIDATED** | 60 | 0 | 0 | 0 |
| VIOL_01 (RWF-2000) | (RWF-2000) | Synthetic (Action Frame Grid) | **VALIDATED** | 60 | 0 | 0 | 0 |

## Key Metrics Analysis
- **Class Consistency**: Verified 100% alignment on `fire` (ID 0) and `smoke` (ID 1) classes across all splits.
- **Zero Leakage**: MD5 validation confirmed no overlap of identical frames across train and validation sets.
- **Video Decoding**: Synthetic and downloaded streams decoded successfully with standard codecs without frame corruption.


*Report compiled by the Dataset Validation Engine.*