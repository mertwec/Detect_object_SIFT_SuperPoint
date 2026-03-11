# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (LightGlue installs from GitHub)
pip install -r requirements.txt

# Run SIFT pipeline (CPU-friendly)
python cli.py sift data_example/ --output output/ --ratio 0.80 --min-matches 6

# Run SuperPoint+LightGlue pipeline (GPU recommended)
python cli.py superpoint data_example/ --output output_SP/ --max-keypoints 2048 --min-matches 6

# Single image detection (pass specific scene path)
python cli.py sift data_example/ --output output/
```

## Architecture

Instance-level object detection: given a reference image, find and localize the object in scene images. Two independent pipelines share geometry logic.

**Data layout expected by CLI**: `<data_dir>/ref/ref.jpg` and `<data_dir>/scenes/*.jpg`

### Module responsibilities

- `detector/features.py` — SIFT extraction with RootSIFT normalization (L1 normalize → sqrt). Returns `Features` dataclass (keypoints, 128-dim descriptors, image_shape).
- `detector/matching.py` — FLANN KD-tree matching with Lowe's ratio test. Returns `cv2.DMatch` list.
- `detector/geometry.py` — Shared geometric verification for both pipelines. `find_object()` (SIFT API) and `find_object_from_points()` (SuperPoint API) both call `_find_object_core()` which runs RANSAC homography, computes inlier ratio, and determines the bounding box. Detection threshold: ≥5 inliers AND inlier_ratio ≥ 0.15. Bbox uses homography projection if ≥15 inliers, else falls back to `minAreaRect` of inlier points. Rejects bbox if it covers >50% of scene area.
- `detector/pipeline.py` — SIFT pipeline. Extracts reference features at 3 scales (`REF_SCALES = [0.25, 0.5, 1.0]`), picks best result (most inliers) per scene.
- `detector/superpoint.py` — `SuperPointMatcher` class wrapping SuperPoint + LightGlue. Pre-extracts reference features once; auto-selects CUDA if available.
- `detector/pipeline_sp.py` — SuperPoint pipeline. Reuses reference features across all scenes in batch mode.
- `cli.py` — Click CLI with two subcommands: `sift` and `superpoint`. Calls `detect_batch` / `detect_batch_sp` and prints a summary table.

### Key design notes

- Both pipelines converge at `geometry.py` — changes to detection thresholds or bbox logic affect both methods.
- `SuperPointMatcher.extract_features()` returns cacheable feature dicts; `detect_batch_sp` exploits this by extracting reference features once.
- `banchmark_time` decorator in `utils/timer.py` (note: typo in name) prints wall-clock time for any wrapped function.
- LightGlue is installed from GitHub source (`git+https://github.com/cvg/LightGlue.git`), not PyPI.
