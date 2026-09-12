"""Pose normalization for gesture recognition.

Converts raw image-space keypoints into a person-centered, scale-invariant
representation so gesture thresholds (how far "out" counts as "out to the
side", how far up counts as "raised") work the same regardless of how
close the person is to the camera or where they're standing in frame.

Handedness note: MediaPipe's pose landmark schema (which the compiled
BlazePose model shares -- see inference/backends/_blazepose.py) labels
LEFT_*/RIGHT_* landmarks by the subject's own anatomical left/right, not
by which side of the image they appear on -- a person facing the camera
has their anatomical right arm on the image's left side. This module
never needs to know or care which absolute image direction is
"anatomically outward" for a given arm: `ArmGeometry.shoulder_offset_x`
already encodes that (see its docstring), so `static_poses.classify_arm`
can define "outward" purely relative to the person's own body geometry.
If real-camera testing ever shows gestures feel mirrored (e.g. "left arm
out" reliably triggers the "right arm" action), that would point to the
underlying model's L/R labels being flipped from the documented
convention, not a bug in this normalization -- see docs/backlog.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from pose_controller.inference.pose import Keypoint, Landmark, PoseResult

MIN_KEYPOINT_VISIBILITY = 0.3
MIN_SHOULDER_WIDTH = 1e-3  # degenerate-scale guard (shoulders nearly coincident)


@dataclass
class ArmGeometry:
    """Geometry for one arm, in shoulder-width units, relative to that
    arm's own shoulder.

    shoulder_offset_x: this shoulder's x position minus the torso center's
        x position. Its *sign* is what "outward" is measured against (see
        module docstring) -- not meaningful as an absolute direction on
        its own.
    wrist_relative: (wrist.x - shoulder.x, wrist.y - shoulder.y), or None
        if the shoulder or wrist isn't visible enough to trust. Image
        y-convention is preserved (increases downward), so a raised wrist
        has a *negative* relative y.
    """

    shoulder_offset_x: float
    wrist_relative: tuple[float, float] | None


@dataclass
class NormalizedPose:
    scale: float  # shoulder width, in original normalized-image units
    left: ArmGeometry
    right: ArmGeometry


def _visible(kp: Keypoint | None) -> bool:
    return kp is not None and kp.visibility >= MIN_KEYPOINT_VISIBILITY


def normalize_pose(pose: PoseResult) -> NormalizedPose | None:
    """Returns None if both shoulders aren't visible enough to establish a
    scale/center reference -- without that, no gesture can be reliably
    classified regardless of arm visibility."""
    left_shoulder = pose.get(Landmark.LEFT_SHOULDER)
    right_shoulder = pose.get(Landmark.RIGHT_SHOULDER)
    if not (_visible(left_shoulder) and _visible(right_shoulder)):
        return None

    scale = ((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2) ** 0.5
    if scale < MIN_SHOULDER_WIDTH:
        return None

    center_x = (left_shoulder.x + right_shoulder.x) / 2

    def arm_geometry(shoulder: Keypoint, wrist: Keypoint | None) -> ArmGeometry:
        shoulder_offset_x = (shoulder.x - center_x) / scale
        wrist_relative = None
        if _visible(wrist):
            wrist_relative = ((wrist.x - shoulder.x) / scale, (wrist.y - shoulder.y) / scale)
        return ArmGeometry(shoulder_offset_x=shoulder_offset_x, wrist_relative=wrist_relative)

    return NormalizedPose(
        scale=scale,
        left=arm_geometry(left_shoulder, pose.get(Landmark.LEFT_WRIST)),
        right=arm_geometry(right_shoulder, pose.get(Landmark.RIGHT_WRIST)),
    )
