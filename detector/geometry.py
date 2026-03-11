from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionResult:
    found: bool
    confidence: float
    bbox: Optional[np.ndarray]  # (4, 2) corners in scene coordinates
    homography: Optional[np.ndarray]
    num_inliers: int
    num_matches: int


def find_object_from_points(
    pts_ref: np.ndarray,
    pts_scene: np.ndarray,
    ref_shape: Tuple[int, int],
    scene_shape: Tuple[int, int] = None,
    min_matches: int = 6,
    ransac_thresh: float = 5.0,
) -> DetectionResult:
    """Verify matches geometrically using pre-matched point arrays.

    Args:
        pts_ref: (N, 2) matched points in reference image
        pts_scene: (N, 2) matched points in scene image
    """
    return _find_object_core(
        pts_ref.reshape(-1, 1, 2).astype(np.float32),
        pts_scene.reshape(-1, 1, 2).astype(np.float32),
        len(pts_ref), ref_shape, scene_shape, min_matches, ransac_thresh,
    )


def find_object(
    matches: List[cv2.DMatch],
    kp_ref: List[cv2.KeyPoint],
    kp_scene: List[cv2.KeyPoint],
    ref_shape: Tuple[int, int],
    scene_shape: Tuple[int, int] = None,
    min_matches: int = 6,
    ransac_thresh: float = 5.0,
) -> DetectionResult:
    """Verify matches geometrically and compute bounding box (SIFT DMatch API)."""
    num = len(matches)
    not_found = DetectionResult(
        found=False, confidence=0.0, bbox=None,
        homography=None, num_inliers=0, num_matches=num,
    )
    if num < min_matches:
        return not_found

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return _find_object_core(src_pts, dst_pts, num, ref_shape, scene_shape, min_matches, ransac_thresh)


def _find_object_core(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    num_matches: int,
    ref_shape: Tuple[int, int],
    scene_shape: Tuple[int, int] = None,
    min_matches: int = 6,
    ransac_thresh: float = 5.0,
) -> DetectionResult:
    """Core geometric verification shared by both APIs."""
    not_found = DetectionResult(
        found=False,
        confidence=0.0,
        bbox=None,
        homography=None,
        num_inliers=0,
        num_matches=num_matches,
    )

    if num_matches < min_matches:
        return not_found

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if H is None or mask is None:
        return not_found

    mask_flat = mask.ravel().astype(bool)
    inliers = int(mask_flat.sum())
    inlier_ratio = inliers / num_matches

    # Inlier points in scene coordinates
    inlier_pts = dst_pts[mask_flat].reshape(-1, 2)

    # --- Compute bbox ---
    # Strategy: use inlier-based minAreaRect (tight around actual matched points).
    # Homography projection is only used when many inliers confirm a reliable transform
    # AND the projected area is reasonable.
    bbox = None

    if len(inlier_pts) >= 3:
        # Inlier-based bbox (always computed)
        rect = cv2.minAreaRect(inlier_pts.astype(np.float32))
        inlier_box = cv2.boxPoints(rect)
        inlier_area = cv2.contourArea(inlier_box.astype(np.int32))

        # Homography-based bbox (only if enough inliers)
        if inliers >= 15:
            h, w = ref_shape
            ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(ref_corners, H).reshape(4, 2)
            proj_int = projected.astype(np.int32)
            proj_area = cv2.contourArea(proj_int)

            # Use homography bbox only if convex and not oversized
            scene_area = scene_shape[0] * scene_shape[1] if scene_shape else float('inf')
            if (cv2.isContourConvex(proj_int)
                    and proj_area > 100
                    and proj_area < 0.5 * scene_area):
                bbox = projected

        # Fallback: always have the inlier-based bbox
        if bbox is None and inlier_area > 100:
            bbox = inlier_box

    # Reject bbox that covers too much of the scene (likely wrong)
    if bbox is not None and scene_shape is not None:
        bbox_area = cv2.contourArea(bbox.astype(np.int32))
        scene_area = scene_shape[0] * scene_shape[1]
        if bbox_area > 0.5 * scene_area:
            bbox = None

    # Detection criteria
    found = inliers >= 5 and inlier_ratio >= 0.15 and bbox is not None

    return DetectionResult(
        found=found,
        confidence=round(inlier_ratio, 3),
        bbox=bbox,
        homography=H,
        num_inliers=inliers,
        num_matches=num_matches,
    )


def draw_detection(
    scene: np.ndarray,
    result: DetectionResult,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw bounding box and confidence on the scene image."""
    out = scene.copy()
    h, w = scene.shape[:2]
    font_scale = max(0.5, min(w, h) / 1000)
    line_thick = max(1, int(font_scale * 2))

    if result.found and result.bbox is not None:
        pts = result.bbox.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

        top_left = result.bbox.min(axis=0).astype(int)
        label = f"conf: {result.confidence:.2f}  inliers: {result.num_inliers}/{result.num_matches}"
        cv2.putText(
            out, label,
            (max(top_left[0], 10), max(top_left[1] - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, line_thick,
        )
    else:
        label = f"NOT FOUND (matches: {result.num_matches}, inliers: {result.num_inliers})"
        cv2.putText(
            out, label,
            (20, int(40 * font_scale)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thick,
        )

    return out
