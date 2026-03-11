# Object Detection via Feature Matching

Instance-level object detection: given a reference image of a rigid static object (building, statue, bridge, etc.), find and localize it in scene images.

Two methods are implemented and compared:

| Method | Features | Matcher | GPU |
|---|---|---|---|
| **SIFT** | RootSIFT (multi-scale) | FLANN + Lowe's ratio test | not required |
| **SuperPoint+LightGlue** | learned keypoints | learned matcher | optional (CUDA) |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> LightGlue is installed directly from GitHub — internet access required on first install.

## Usage

```bash
# SIFT — classical CV, no GPU needed
python cli.py --ref data_example/ref.jpg --scenes data_example/ --output output/ --method sift

# SuperPoint+LightGlue — deep learning, higher accuracy
python cli.py --ref data_example/ref.jpg --scenes data_example/ --output output_SP/ --method superpoint

# SuperPoint with more keypoints (better for distant/small objects)
python cli.py --ref data_example/ref.jpg --scenes data_example/ --output output_SP/ \
    --method superpoint --max-keypoints 4096
```

The `--scenes` argument accepts either a single image or a directory. When a directory is given, it should contain `ref/` and `scenes/` subdirectories; the script searches all images in `scenes/`.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--ref` | required | Path to reference image |
| `--scenes` | required | Path to image or directory |
| `--output` | `./output` | Output directory for annotated images |
| `--method` | `sift` | `sift` or `superpoint` |
| `--ratio` | `0.80` | Lowe's ratio test threshold (SIFT only) |
| `--min-matches` | `6` | Minimum matches required for homography |
| `--max-keypoints` | `2048` | Max keypoints to extract (SuperPoint only) |

### Output

For each scene image the script prints a summary table and saves an annotated image to `--output`:
- **found** — detection status
- **confidence** — inlier ratio (0–1)
- **bbox** — bounding box coordinates in the scene image

## Results on Test Data (Eiffel Tower, 10 scenes)

| Scene | SIFT | SuperPoint+LG | Notes |
|---|---|---|---|
| 01 — distant aerial | No (4 inl) | **Yes** (6 inl) | SP detects at extreme distance |
| 02 — night, colored | No (5 inl) | **Yes** (18 inl) | SP handles illumination change |
| 03 — view from the tower | Yes **(FP!)** | No | SP correctly rejects |
| 04 — frontal | Yes (7 inl) | **Yes** (109 inl) | SP: 92% inlier ratio |
| 05 — aerial, far | Yes (7 inl) | Yes (5 inl) | Both detect |
| 06 — sunset | Yes (6 inl) | **Yes** (74 inl) | SP: 12× more inliers |
| 07 — distant, haze | Yes (7 inl) | Yes (8 inl) | Both detect |
| 08 — top-down aerial | Yes (6 inl) | No | Extreme viewpoint |
| 09 — aerial | Yes (8 inl) | Yes (6 inl) | Both detect |
| 10 — street level | Yes (48 inl) | **Yes** (202 inl) | SP: 4× more inliers |

**SIFT**: 8/10 detected, 1 false positive, low inlier counts.
**SuperPoint+LG**: 8/10 detected, 0 false positives, significantly higher inlier counts and tighter bounding boxes.

## How It Works

Both pipelines share the same geometric verification step:

```
Reference image ──→ Feature extraction
Scene image     ──→ Feature extraction
                        ↓
                  Descriptor matching
                        ↓
               RANSAC homography estimation
                        ↓
              Bounding box + confidence score
```

**SIFT pipeline** processes the reference at 3 scales (0.25×, 0.5×, 1.0×) and picks the result with the most inliers. Descriptors are L1-normalized and square-rooted (RootSIFT) before FLANN matching.

**SuperPoint+LightGlue pipeline** uses a single scale (the network handles scale internally). LightGlue replaces the ratio test with attention-based learned matching.

Detection is accepted when: `inliers ≥ 5`, `inlier_ratio ≥ 0.15`, and the bounding box covers less than 50% of the scene.

## Project Structure

```
cli.py              # CLI entry point
detector/
  features.py       # SIFT extraction + RootSIFT normalization
  matching.py       # FLANN KD-tree matcher + Lowe's ratio test
  superpoint.py     # SuperPoint extractor + LightGlue matcher
  geometry.py       # RANSAC verification, bbox, visualization (shared)
  pipeline.py       # SIFT pipeline
  pipeline_sp.py    # SuperPoint+LightGlue pipeline
data_example/
  ref/ref.jpg       # reference object (Eiffel Tower)
  scenes/01-10.jpg  # test scenes
output/             # SIFT annotated results
output_SP/          # SuperPoint annotated results
```
