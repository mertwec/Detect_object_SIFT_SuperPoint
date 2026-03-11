from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Features:
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    image_shape: Tuple[int, int]  # (h, w)


def extract_sift(image: np.ndarray, max_features: int = 0, rootsift: bool = True) -> Features:
    """Extract SIFT keypoints and descriptors from an image.

    When rootsift=True, applies RootSIFT normalization (L1 norm + sqrt)
    which significantly improves matching quality.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    elif rootsift and descriptors.shape[0] > 0:
        # RootSIFT: L1 normalize, then take sqrt
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        descriptors = np.sqrt(descriptors)

    return Features(
        keypoints=list(keypoints),
        descriptors=descriptors,
        image_shape=gray.shape[:2],
    )
