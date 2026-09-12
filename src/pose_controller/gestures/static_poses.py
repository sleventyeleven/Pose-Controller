from __future__ import annotations

from pose_controller.gestures.normalize import ArmGeometry
from pose_controller.gestures.types import ArmPose

# All thresholds are in shoulder-width units (see normalize.py) -- not
# empirically tuned against real camera footage yet, just reasoned
# starting points. Expect to retune once real gesture testing is
# possible; see docs/backlog.md.
RAISE_Y_THRESHOLD = 0.4  # wrist this far above the shoulder (image y decreases upward) -> RAISED
SIDE_X_THRESHOLD = 0.5  # wrist this far outward from the shoulder -> OUT_TO_SIDE
SIDE_Y_TOLERANCE = 0.35  # how close to shoulder height counts as "to the side" rather than raised/lowered


def classify_arm(
    arm: ArmGeometry,
    raise_threshold: float = RAISE_Y_THRESHOLD,
    side_x_threshold: float = SIDE_X_THRESHOLD,
    side_y_tolerance: float = SIDE_Y_TOLERANCE,
) -> ArmPose | None:
    """Classify one arm's static pose from its normalized geometry.

    Returns None (not ArmPose.DOWN) when the wrist isn't visible enough to
    classify -- distinct from a confidently-observed DOWN, so callers
    (gestures.state_machine.GestureStateMachine) can choose to hold the
    last confirmed state through a momentary occlusion rather than
    treating a dropped keypoint as "arm lowered."
    """
    if arm.wrist_relative is None:
        return None

    rel_x, rel_y = arm.wrist_relative

    if rel_y < -raise_threshold:
        return ArmPose.RAISED

    # "Outward" is relative to this arm's own shoulder offset from torso
    # center -- same sign means the wrist has moved further in the
    # direction the shoulder already leans, i.e. away from the body's
    # midline, regardless of which absolute image direction that is for
    # this particular arm (see normalize.py's module docstring).
    is_outward = (rel_x * arm.shoulder_offset_x) > 0
    outward_magnitude = abs(rel_x) if is_outward else 0.0

    if abs(rel_y) < side_y_tolerance and outward_magnitude > side_x_threshold:
        return ArmPose.OUT_TO_SIDE

    return ArmPose.DOWN
