"""Detection pipeline using SuperPoint + LightGlue."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from .geometry import DetectionResult, draw_detection, find_object_from_points
from .superpoint import SuperPointMatcher


def detect_sp(
    ref_path: str,
    scene_path: str,
    output_path: Optional[str] = None,
    max_keypoints: int = 2048,
) -> DetectionResult:
    """Detect the reference object in a scene using SuperPoint+LightGlue."""
    sp = SuperPointMatcher(max_keypoints=max_keypoints)
    ref_feats = sp.extract_features(ref_path)
    return _detect_single(sp, ref_feats, ref_path, scene_path, output_path)


def detect_batch_sp(
    ref_path: str,
    scene_dir: str,
    output_dir: str,
    max_keypoints: int = 2048,
) -> List[Tuple[str, DetectionResult]]:
    """Detect reference object across all images using SuperPoint+LightGlue."""
    sp = SuperPointMatcher(max_keypoints=max_keypoints)
    ref_feats = sp.extract_features(ref_path)

    scene_dir_path = Path(scene_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    scene_files = sorted(
        p for p in scene_dir_path.iterdir()
        if p.suffix.lower() in extensions and p.name != Path(ref_path).name
    )

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for scene_file in scene_files:
        out_path = os.path.join(output_dir, scene_file.name)
        result = _detect_single(sp, ref_feats, ref_path, str(scene_file), out_path)
        results.append((scene_file.name, result))

    return results


def _detect_single(
    sp: SuperPointMatcher,
    ref_feats,
    ref_path: str,
    scene_path: str,
    output_path: Optional[str],
) -> DetectionResult:
    """Run detection on a single scene."""
    scene_feats = sp.extract_features(scene_path)
    match_result = sp.match_features(ref_feats, scene_feats)

    # Get image dimensions for bbox validation
    ref_img = cv2.imread(ref_path)
    scene_img = cv2.imread(scene_path)
    ref_shape = ref_img.shape[:2]
    scene_shape = scene_img.shape[:2]

    result = find_object_from_points(
        match_result.pts_ref,
        match_result.pts_scene,
        ref_shape=ref_shape,
        scene_shape=scene_shape,
        min_matches=6,
    )

    if output_path:
        annotated = draw_detection(scene_img, result)
        cv2.imwrite(output_path, annotated)

    return result
