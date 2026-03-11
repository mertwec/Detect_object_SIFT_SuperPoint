from typing import List

import cv2
import numpy as np


def match_descriptors(
    des_ref: np.ndarray,
    des_scene: np.ndarray,
    ratio: float = 0.75,
) -> List[cv2.DMatch]:
    """Match descriptors using FLANN + Lowe's ratio test."""
    if des_ref.shape[0] < 2 or des_scene.shape[0] < 2:
        return []

    # FLANN with KD-tree (faster than BFMatcher for large descriptor sets)
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    raw_matches = flann.knnMatch(des_ref, des_scene, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good
