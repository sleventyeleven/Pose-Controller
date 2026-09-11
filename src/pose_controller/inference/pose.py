from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Landmark(IntEnum):
    """Upper-body-relevant subset, indices match MediaPipe Pose's 33-point schema."""

    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24


@dataclass
class Keypoint:
    x: float  # normalized [0, 1], image space (not corrected for handedness)
    y: float
    visibility: float  # 0..1 confidence, low when occluded/submerged


@dataclass
class PoseResult:
    """One detected person's keypoints for a single frame."""

    keypoints: dict[int, Keypoint]

    def get(self, landmark: Landmark) -> Keypoint | None:
        return self.keypoints.get(int(landmark))


class PoseEstimator(ABC):
    """Backend-agnostic single-person pose estimator.

    Implementations: `backends.cpu.MediaPipePoseEstimator` (dev, laptop CPU)
    and `backends.qnn.QnnPoseEstimator` (on-device, Q6A Hexagon NPU).
    """

    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray) -> PoseResult | None:
        """Return keypoints for the most prominent person in `frame_bgr`, or
        None if no person was detected."""

    def close(self) -> None:
        pass
