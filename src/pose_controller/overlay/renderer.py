from __future__ import annotations

import cv2
import numpy as np

from pose_controller.inference.pose import PoseResult

# Upper-body-focused skeleton edges (MediaPipe Pose landmark indices).
_SKELETON_EDGES = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),  # hips
]

_VISIBILITY_THRESHOLD = 0.5

# Distinct colors cycled by track_id so multiple people are visually
# separable frame to frame (BGR, OpenCV convention).
_TRACK_COLORS = [
    (0, 200, 255), (255, 120, 0), (0, 255, 0), (255, 0, 255),
    (0, 165, 255), (255, 255, 0), (128, 0, 255), (0, 255, 128),
]


def _color_for_track(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (160, 160, 160)  # neutral gray for an untracked/unconfirmed detection
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


def draw_poses(frame_bgr: np.ndarray, poses: list[PoseResult]) -> np.ndarray:
    """Draw skeleton keypoints/edges for every detected person on top of
    `frame_bgr` in place, returning it for chaining. Each person's color
    is keyed by `track_id` (stable across frames once `tracking.Tracker`
    has assigned one) so identity is visible at a glance, not just
    position."""
    for pose in poses:
        _draw_one(frame_bgr, pose)
    return frame_bgr


def _draw_one(frame_bgr: np.ndarray, pose: PoseResult) -> None:
    h, w = frame_bgr.shape[:2]
    color = _color_for_track(pose.track_id)

    def pixel(idx: int) -> tuple[int, int] | None:
        kp = pose.keypoints.get(idx)
        if kp is None or kp.visibility < _VISIBILITY_THRESHOLD:
            return None
        return int(kp.x * w), int(kp.y * h)

    for a, b in _SKELETON_EDGES:
        pa, pb = pixel(a), pixel(b)
        if pa and pb:
            cv2.line(frame_bgr, pa, pb, color, 2)

    for idx in pose.keypoints:
        p = pixel(idx)
        if p:
            cv2.circle(frame_bgr, p, 4, color, -1)

    label = f"#{pose.track_id}" if pose.track_id is not None else "?"
    origin = pixel(0) or (10, 30)
    cv2.putText(
        frame_bgr, label, (origin[0], max(origin[1] - 20, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )
