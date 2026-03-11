"""SuperPoint feature extraction + LightGlue matching."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image


@dataclass
class SPMatchResult:
    pts_ref: np.ndarray    # (N, 2) matched points in reference
    pts_scene: np.ndarray  # (N, 2) matched points in scene
    scores: np.ndarray     # (N,) match confidence scores
    num_matches: int


class SuperPointMatcher:
    """SuperPoint extractor + LightGlue matcher."""

    def __init__(self, max_keypoints: int = 2048, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.extractor = (
            SuperPoint(max_num_keypoints=max_keypoints)
            .eval()
            .to(device)
        )
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    @torch.inference_mode()
    def extract_and_match(
        self, ref_path: str, scene_path: str
    ) -> SPMatchResult:
        """Extract SuperPoint features and match with LightGlue."""
        img0 = load_image(ref_path).to(self.device)
        img1 = load_image(scene_path).to(self.device)

        feats0 = self.extractor.extract(img0)
        feats1 = self.extractor.extract(img1)

        result = self.matcher({"image0": feats0, "image1": feats1})

        matches = result["matches"][0]  # [N, 2]
        scores = result["scores"][0]    # [N]

        kpts0 = feats0["keypoints"][0]
        kpts1 = feats1["keypoints"][0]

        pts_ref = kpts0[matches[:, 0]].cpu().numpy()
        pts_scene = kpts1[matches[:, 1]].cpu().numpy()
        scores_np = scores.cpu().numpy()

        return SPMatchResult(
            pts_ref=pts_ref,
            pts_scene=pts_scene,
            scores=scores_np,
            num_matches=len(pts_ref),
        )

    @torch.inference_mode()
    def extract_features(self, image_path: str):
        """Extract features from a single image (for batch reuse)."""
        img = load_image(image_path).to(self.device)
        return self.extractor.extract(img)

    @torch.inference_mode()
    def match_features(self, feats0, feats1) -> SPMatchResult:
        """Match pre-extracted features."""
        result = self.matcher({"image0": feats0, "image1": feats1})

        matches = result["matches"][0]
        scores = result["scores"][0]

        if len(matches) == 0:
            return SPMatchResult(
                pts_ref=np.empty((0, 2)),
                pts_scene=np.empty((0, 2)),
                scores=np.empty(0),
                num_matches=0,
            )

        kpts0 = feats0["keypoints"][0]
        kpts1 = feats1["keypoints"][0]

        pts_ref = kpts0[matches[:, 0]].cpu().numpy()
        pts_scene = kpts1[matches[:, 1]].cpu().numpy()
        scores_np = scores.cpu().numpy()

        return SPMatchResult(
            pts_ref=pts_ref,
            pts_scene=pts_scene,
            scores=scores_np,
            num_matches=len(pts_ref),
        )
