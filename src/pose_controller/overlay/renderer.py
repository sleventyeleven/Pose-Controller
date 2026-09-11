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


def draw_pose(frame_bgr: np.ndarray, pose: PoseResult | None, label: str | None = None) -> np.ndarray:
    """Draw skeleton keypoints/edges and an optional entity label on top of
    `frame_bgr` in place, returning it for chaining."""
    if pose is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]

    def pixel(idx: int) -> tuple[int, int] | None:
        kp = pose.keypoints.get(idx)
        if kp is None or kp.visibility < _VISIBILITY_THRESHOLD:
            return None
        return int(kp.x * w), int(kp.y * h)

    for a, b in _SKELETON_EDGES:
        pa, pb = pixel(a), pixel(b)
        if pa and pb:
            cv2.line(frame_bgr, pa, pb, (0, 255, 0), 2)

    for idx in pose.keypoints:
        p = pixel(idx)
        if p:
            cv2.circle(frame_bgr, p, 4, (0, 200, 255), -1)

    if label:
        origin = pixel(0) or (10, 30)
        cv2.putText(
            frame_bgr, label, (origin[0], max(origin[1] - 20, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return frame_bgr
