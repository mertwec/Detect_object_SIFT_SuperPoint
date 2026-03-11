import os
from pathlib import Path
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .features import Features, extract_sift
from .geometry import DetectionResult, draw_detection, find_object
from .matching import match_descriptors


# Scales to resize the reference image for multi-scale matching
REF_SCALES = [0.25, 0.5, 1.0]


def detect(
    ref_path: str,
    scene_path: str,
    output_path: Optional[str] = None,
    ratio: float = 0.80,
    min_matches: int = 6,
) -> DetectionResult:
    """Detect the reference object in a scene image."""
    ref_img = cv2.imread(ref_path)
    scene_img = cv2.imread(scene_path)

    if ref_img is None:
        raise FileNotFoundError(f"Cannot load reference: {ref_path}")
    if scene_img is None:
        raise FileNotFoundError(f"Cannot load scene: {scene_path}")

    ref_features = _extract_multiscale(ref_img)
    scene_feat = extract_sift(scene_img)

    return _detect_with_features(ref_features, scene_img, scene_feat, output_path, ratio, min_matches)


def detect_batch(
    ref_path: str,
    scene_dir: str,
    output_dir: str,
    ratio: float = 0.80,
    min_matches: int = 6,
) -> List[Tuple[str, DetectionResult]]:
    """Detect reference object across all images in a directory."""
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        raise FileNotFoundError(f"Cannot load reference: {ref_path}")

    ref_features = _extract_multiscale(ref_img)

    scene_dir_path = Path(scene_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    scene_files = sorted(
        p for p in scene_dir_path.iterdir()
        if p.suffix.lower() in extensions and p.name != Path(ref_path).name
    )

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for scene_file in scene_files:
        t_start = time.perf_counter()   

        scene_img = cv2.imread(str(scene_file))
        if scene_img is None:
            continue

        scene_feat = extract_sift(scene_img)
        out_path = os.path.join(output_dir, scene_file.name)
        result = _detect_with_features(ref_features, scene_img, scene_feat, out_path, ratio, min_matches)
        results.append((scene_file.name, result))
        
        t_end = time.perf_counter()
        print(f"Processed {scene_file.name} in {t_end - t_start:.2f} seconds")

    return results


def _extract_multiscale(image: np.ndarray) -> List[Features]:
    """Extract SIFT features at multiple scales of the reference image."""
    features = []
    for scale in REF_SCALES:
        if scale == 1.0:
            features.append(extract_sift(image))
        else:
            h, w = image.shape[:2]
            resized = cv2.resize(image, (int(w * scale), int(h * scale)))
            features.append(extract_sift(resized))
    return features


def _detect_with_features(
    ref_features: List[Features],
    scene_img: np.ndarray,
    scene_feat: Features,
    output_path: Optional[str],
    ratio: float,
    min_matches: int,
) -> DetectionResult:
    """Core detection logic: try all ref scales and pick best result."""
    best_result = DetectionResult(
        found=False, confidence=0.0, bbox=None,
        homography=None, num_inliers=0, num_matches=0,
    )

    for ref_feat in ref_features:
        matches = match_descriptors(ref_feat.descriptors, scene_feat.descriptors, ratio)
        result = find_object(
            matches,
            ref_feat.keypoints,
            scene_feat.keypoints,
            ref_feat.image_shape,
            scene_shape=scene_feat.image_shape,
            min_matches=min_matches,
        )

        # Pick result with most inliers
        if result.num_inliers > best_result.num_inliers:
            best_result = result

    if output_path:
        annotated = draw_detection(scene_img, best_result)
        cv2.imwrite(output_path, annotated)

    return best_result
