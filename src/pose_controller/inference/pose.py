from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Landmark(IntEnum):
    """Indices match MediaPipe Pose's 33-point schema. Covers COCO's full
    17-keypoint set (nose, eyes, ears, shoulders, elbows, wrists, hips,
    knees, ankles) for the overlay's full-body skeleton -- see
    `overlay/renderer.py`'s `_SKELETON_EDGES`.

    Gesture recognition (`gestures/`) only ever reads the upper-body
    subset (shoulders/elbows/wrists/hips); the leg entries exist purely
    for display. Not every backend can actually fill every entry: the
    `qnn` backend's compiled BlazePose landmark model only outputs 25 of
    MediaPipe's 33 landmarks (`_blazepose.NUM_VALID_LANDMARKS`), which
    covers eyes/ears but stops short of knees/ankles -- see
    `docs/pipelines.md`. `yolo26`/`hrnet` cover all of these (COCO's own
    17-point schema, decoded in full by both)."""

    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass
class Keypoint:
    x: float  # normalized [0, 1], image space (not corrected for handedness)
    y: float
    visibility: float  # 0..1 confidence, low when occluded/submerged


# Minimum visibility for a keypoint to count towards a person's bounding
# box -- low-visibility points (occluded/guessed) are excluded so a bbox
# isn't dragged out by unreliable extremities.
_BBOX_MIN_VISIBILITY = 0.3


def bbox_from_keypoints(
    keypoints: dict[int, Keypoint], min_visibility: float = _BBOX_MIN_VISIBILITY
) -> tuple[float, float, float, float] | None:
    """Normalized [0,1] xyxy bounding box enclosing all sufficiently-visible
    keypoints, or None if fewer than 2 such keypoints exist. Used as a
    backend-agnostic stand-in for a detector-provided box (both backends'
    underlying models -- MediaPipe's legacy solutions API and the
    BlazePose two-stage pipeline -- report per-keypoint visibility, so this
    works the same way for either), primarily to feed `tracking.Tracker`."""
    visible = [kp for kp in keypoints.values() if kp.visibility >= min_visibility]
    if len(visible) < 2:
        return None
    xs = [kp.x for kp in visible]
    ys = [kp.y for kp in visible]
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class PoseResult:
    """One detected person's keypoints for a single frame.

    `bbox` (normalized xyxy) is derived from keypoint visibility rather
    than provided by the detector directly, so it means the same thing
    across backends -- see `bbox_from_keypoints`. `track_id` is unset until
    a `tracking.Tracker` assigns one; `PoseEstimator.estimate` never sets it.
    """

    keypoints: dict[int, Keypoint]
    bbox: tuple[float, float, float, float] | None = field(default=None)
    track_id: int | None = field(default=None)

    def __post_init__(self) -> None:
        if self.bbox is None:
            self.bbox = bbox_from_keypoints(self.keypoints)

    def get(self, landmark: Landmark) -> Keypoint | None:
        return self.keypoints.get(int(landmark))


class PoseEstimator(ABC):
    """Backend-agnostic multi-person pose estimator.

    Implementations: `backends.cpu.MediaPipePoseEstimator` (dev, laptop
    CPU -- currently single-person only, see docs/backlog.md) and
    `backends.qnn.QnnPoseEstimator` (on-device, Q6A Hexagon NPU --
    genuinely multi-person via NMS over the detector's anchors).
    """

    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray) -> list[PoseResult]:
        """Return one PoseResult per detected person in `frame_bgr` (empty
        list if none). Order is not guaranteed to be stable across frames
        -- use `tracking.Tracker` for persistent identity."""

    def close(self) -> None:
        pass
