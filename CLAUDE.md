# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Instance-level object detection comparing two approaches:
1. **SIFT** — classical CV: RootSIFT + FLANN + RANSAC homography
2. **SuperPoint+LightGlue** — deep learning: learned features + matcher + RANSAC

## Running the Pipeline

```bash
# Activate virtual environment
source .venv/bin/activate

# SIFT method
python cli.py --ref data_example/ref.jpg --scenes data_example/ --output output/ --method sift

# SuperPoint method
python cli.py --ref data_example/ref.jpg --scenes data_example/ --output output_SP/ --method superpoint --max-keypoints 4096
```

**CLI arguments:**
- `--ref`: reference image path (required)
- `--scenes`: image or directory path (required; expects `ref/` + `scenes/` subdirs when directory)
- `--output`: output directory (default: `./output`)
- `--method`: `sift` or `superpoint` (default: `sift`)
- `--ratio`: Lowe's ratio test threshold, SIFT only (default: 0.80)
- `--min-matches`: minimum matches for homography (default: 6)
- `--max-keypoints`: max keypoints, SuperPoint only (default: 2048)

## Architecture

```
cli.py                      # Click CLI entry point
detector/
  __init__.py               # Re-exports detect, detect_batch, detect_sp, detect_batch_sp
  pipeline.py               # SIFT pipeline: multi-scale extraction, batch support
  pipeline_sp.py            # SuperPoint pipeline: single-scale, batch support
  features.py               # SIFT extraction + RootSIFT normalization
  matching.py               # FLANN KD-tree matcher + Lowe's ratio test
  superpoint.py             # SuperPointMatcher class wrapping SuperPoint + LightGlue
  geometry.py               # Shared RANSAC verification, bbox computation, visualization
  utils/timer.py            # banchmark_time() timing decorator
```

### Data Flow

Both pipelines follow the same pattern:
1. Extract features from reference and scene images
2. Match descriptors (FLANN ratio test for SIFT; LightGlue for SuperPoint)
3. Geometric verification via RANSAC homography (`geometry._find_object_core()`)
4. Compute bounding box (homography projection if ≥15 inliers, else `minAreaRect`)
5. Return `DetectionResult` (found, confidence, bbox, homography, num_inliers, num_matches)

### Key Design Decisions

- **Multi-scale SIFT**: reference processed at 3 scales (0.25×, 0.5×, 1.0×); best inlier count wins
- **RootSIFT**: L1-norm + sqrt applied before FLANN matching for better Hellinger distance
- **Detection threshold**: ≥5 inliers AND inlier_ratio ≥ 0.15 AND bbox < 50% of scene area
- **SuperPoint**: GPU auto-detection (CUDA if available, else CPU); no multi-scale needed

## Dependencies

Install into virtualenv:
```bash
pip install -r requirements.txt
```

Key packages: `opencv-python`, `numpy`, `torch`, `torchvision`, `click`, `lightglue` (from GitHub).
