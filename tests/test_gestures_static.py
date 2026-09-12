from pose_controller.gestures.normalize import ArmGeometry
from pose_controller.gestures.static_poses import classify_arm
from pose_controller.gestures.types import ArmPose


def test_classify_arm_none_when_wrist_not_visible():
    arm = ArmGeometry(shoulder_offset_x=-0.5, wrist_relative=None)
    assert classify_arm(arm) is None


def test_classify_arm_raised_when_wrist_well_above_shoulder():
    arm = ArmGeometry(shoulder_offset_x=-0.5, wrist_relative=(0.1, -1.0))
    assert classify_arm(arm) == ArmPose.RAISED


def test_classify_arm_out_to_side_negative_shoulder_offset():
    # Shoulder offset negative (e.g. this is the side of the body closer
    # to image-left); wrist moves further negative (further outward) at
    # roughly shoulder height.
    arm = ArmGeometry(shoulder_offset_x=-0.5, wrist_relative=(-1.0, 0.0))
    assert classify_arm(arm) == ArmPose.OUT_TO_SIDE


def test_classify_arm_out_to_side_positive_shoulder_offset():
    # Mirror of the above -- "outward" is relative to the shoulder's own
    # offset sign, not a hardcoded image direction (see normalize.py).
    arm = ArmGeometry(shoulder_offset_x=0.5, wrist_relative=(1.0, 0.0))
    assert classify_arm(arm) == ArmPose.OUT_TO_SIDE


def test_classify_arm_down_when_wrist_moves_inward_not_outward():
    # Wrist displaced from shoulder, but toward the body's midline (sign
    # mismatch with shoulder_offset_x), not away from it -- not "out".
    arm = ArmGeometry(shoulder_offset_x=-0.5, wrist_relative=(1.0, 0.0))
    assert classify_arm(arm) == ArmPose.DOWN


def test_classify_arm_down_when_wrist_hangs_at_side():
    arm = ArmGeometry(shoulder_offset_x=-0.5, wrist_relative=(0.0, 1.0))
    assert classify_arm(arm) == ArmPose.DOWN


def test_classify_arm_down_when_displacement_below_thresholds():
    arm = ArmGeometry(shoulder_offset_x=-0.5, wrist_relative=(-0.1, 0.05))
    assert classify_arm(arm) == ArmPose.DOWN
