from __future__ import annotations

import cv2
import numpy as np

from pose_controller.gestures.types import ArmPose
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

_ARM_POSE_LABEL = {
    ArmPose.DOWN: "-",
    ArmPose.RAISED: "UP",
    ArmPose.OUT_TO_SIDE: "OUT",
}


def _color_for_track(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (160, 160, 160)  # neutral gray for an untracked/unconfirmed detection
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


def draw_poses(
    frame_bgr: np.ndarray,
    poses: list[PoseResult],
    arm_states: dict[int, tuple[ArmPose, ArmPose]] | None = None,
) -> np.ndarray:
    """Draw skeleton keypoints/edges for every detected person on top of
    `frame_bgr` in place, returning it for chaining. Each person's color
    is keyed by `track_id` (stable across frames once `tracking.Tracker`
    has assigned one) so identity is visible at a glance, not just
    position.

    `arm_states` (from `gestures.GestureStateMachine.get_arm_states`, keyed
    by track_id) draws each person's current confirmed left/right arm pose
    as text -- this is the "why" half of causality: a viewer sees an arm
    state building up before the action it eventually triggers, not just
    the action appearing with no visible cause.
    """
    for pose in poses:
        state = arm_states.get(pose.track_id) if arm_states and pose.track_id is not None else None
        _draw_one(frame_bgr, pose, state)
    return frame_bgr


def _draw_one(
    frame_bgr: np.ndarray, pose: PoseResult, arm_state: tuple[ArmPose, ArmPose] | None
) -> None:
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
    if arm_state is not None:
        left, right = arm_state
        label += f"  L:{_ARM_POSE_LABEL[left]} R:{_ARM_POSE_LABEL[right]}"
    origin = pixel(0) or (10, 30)
    cv2.putText(
        frame_bgr, label, (origin[0], max(origin[1] - 20, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )


def draw_action_banner(frame_bgr: np.ndarray, text: str | None) -> np.ndarray:
    """Draws a brief "what just happened" banner near the top of the
    frame -- the other half of causality alongside `draw_poses`'
    `arm_states` labels. `text` is typically something like
    "#1: NEXT" (see app.py, which owns how long a banner stays visible
    before clearing `text` back to None); this function itself has no
    concept of fading/timing, it just draws or doesn't."""
    if not text:
        return frame_bgr

    w = frame_bgr.shape[1]
    cv2.rectangle(frame_bgr, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr, text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame_bgr
