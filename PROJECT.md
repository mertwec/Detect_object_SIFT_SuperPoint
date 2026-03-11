# Object Detection via Feature Matching (SIFT & SuperPoint+LightGlue)

Instance-level object detection using two methods:
1. **SIFT** (classical CV) — RootSIFT + FLANN + RANSAC homography
2. **SuperPoint+LightGlue** (deep learning) — learned features + learned matcher + RANSAC

## Task

Given a reference image of a rigid static object (building, statue, bridge, etc.), find and localize it in scene images.

**Output per scene:**
- Detection status (found / not found)
- Confidence score (inlier ratio)
- Bounding box around the detected object
- Annotated image saved to output directory

## Pipelines

### SIFT Pipeline
```
ref.jpg ──→ SIFT + RootSIFT (multi-scale: 0.25x, 0.5x, 1.0x)
scene.jpg ──→ SIFT + RootSIFT
                  ↓
        FLANN matching + Lowe's ratio test (0.80)
                  ↓
        RANSAC homography → bbox → Result
```

### SuperPoint+LightGlue Pipeline
```
ref.jpg ──→ SuperPoint (learned keypoints + descriptors)
scene.jpg ──→ SuperPoint
                  ↓
        LightGlue (learned matcher, no ratio test needed)
                  ↓
        RANSAC homography → bbox → Result
```

## Project Structure

```
├── cli.py                    # CLI entry point (--method sift|superpoint)
├── detector/
│   ├── __init__.py           # re-exports all public APIs
│   ├── features.py           # SIFT extraction + RootSIFT normalization
│   ├── matching.py           # FLANN matcher + Lowe's ratio test
│   ├── geometry.py           # RANSAC homography, bbox, drawing (shared)
│   ├── pipeline.py           # SIFT pipeline (multi-scale, batch)
│   ├── superpoint.py         # SuperPoint extractor + LightGlue matcher
│   └── pipeline_sp.py        # SuperPoint+LightGlue pipeline (batch)
├── data_example/             # test data
│   ├── ref.jpg               # reference (Eiffel Tower)
│   └── 01-10.jpg             # scene images
├── output/                   # SIFT results
├── output_SP/                # SuperPoint results
├── requirements.txt
├── algorythm.md              # original task description (RU)
└── PROJECT.md
```

## Usage

```bash
# SIFT (classical, no GPU needed)
python cli.py --ref data_example/ref.jpg --scenes data_example/ --output output/ --method sift

# SuperPoint+LightGlue (deep learning, better accuracy)
python cli.py --ref data_example/ref.jpg --scenes data_example/ --output output_SP/ --method superpoint

# SuperPoint with more keypoints for distant objects
python cli.py --ref data_example/ref.jpg --scenes data_example/ --output output_SP/ \
    --method superpoint --max-keypoints 4096
```

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--ref` | required | Path to reference image |
| `--scenes` | required | Path to image or directory |
| `--output` | `./output` | Output directory for annotated images |
| `--method` | `sift` | `sift` or `superpoint` |
| `--ratio` | `0.80` | Lowe's ratio test threshold (SIFT only) |
| `--min-matches` | `6` | Minimum matches for homography |
| `--max-keypoints` | `2048` | Max keypoints (SuperPoint only) |

## Comparison on Test Data (Eiffel Tower)

| Scene | Ground Truth | SIFT | SuperPoint+LG | Notes |
|---|---|---|---|---|
| 01.jpg | tower (distant aerial) | No (4 inl) | **Yes** (6 inl) | SP detects at extreme distance |
| 02.jpg | tower (night, colored) | No (5 inl) | **Yes** (18 inl) | SP handles illumination change |
| 03.jpg | no tower (view FROM it) | Yes (FP!) | **No** | SP correctly rejects |
| 04.jpg | tower (frontal) | Yes (7 inl) | **Yes** (109 inl) | SP: 92% inlier ratio |
| 05.jpg | tower (aerial, far) | Yes (7 inl) | Yes (5 inl) | Both detect, wide bbox |
| 06.jpg | tower (sunset) | Yes (6 inl) | **Yes** (74 inl) | SP: 12x more inliers |
| 07.jpg | tower (distant, haze) | Yes (7 inl) | Yes (8 inl) | SP: tighter bbox |
| 08.jpg | tower (top-down aerial) | Yes (6 inl) | No (0 inl) | Extreme viewpoint |
| 09.jpg | tower (aerial) | Yes (8 inl) | Yes (6 inl) | Both detect |
| 10.jpg | tower (street level) | Yes (48 inl) | **Yes** (202 inl) | SP: 4x more inliers |

**Summary:**
- **SIFT**: 8/10 detected (1 false positive on 03.jpg), low inlier counts, imprecise bboxes
- **SuperPoint+LG**: 8/10 detected (0 false positives), high inlier counts, precise bboxes
- SP wins on quality: 109 vs 7 inliers on frontal view (04.jpg), handles night scenes (02.jpg), no false positives

## Dependencies

```
# Core (SIFT only)
opencv-python >= 4.5.0
numpy >= 1.20.0

# SuperPoint+LightGlue (additional)
torch >= 2.0.0
torchvision >= 0.15.0
lightglue  (from github.com/cvg/LightGlue)
```

## Algorithm Details

### SIFT Method (features.py, matching.py)
- **RootSIFT**: L1 normalize + sqrt for Hellinger distance (better than raw SIFT)
- **FLANN** with KD-tree + Lowe's ratio test at 0.80
- **Multi-scale**: reference processed at 0.25x, 0.5x, 1.0x for scale robustness

### SuperPoint+LightGlue Method (superpoint.py)
- **SuperPoint**: learned keypoint detector + descriptor, robust to viewpoint and illumination
- **LightGlue**: learned matcher that replaces ratio test with attention-based matching
- No multi-scale needed (SuperPoint handles scale internally)

### Geometric Verification (geometry.py, shared)
- **RANSAC homography** via `cv2.findHomography`
- **Bbox strategy**: homography projection (15+ inliers) or minAreaRect fallback
- **Area filter**: reject bbox > 50% of scene area
- **Detection criteria**: inliers >= 5, inlier_ratio >= 0.15, valid bbox
