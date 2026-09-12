import pytest

from pose_controller.gestures.normalize import normalize_pose
from pose_controller.inference.pose import Keypoint, Landmark, PoseResult


def _pose(landmarks):
    return PoseResult(keypoints={int(lm): kp for lm, kp in landmarks.items()})


def test_normalize_returns_none_without_both_shoulders():
    pose = _pose({Landmark.LEFT_SHOULDER: Keypoint(0.3, 0.4, 0.9)})
    assert normalize_pose(pose) is None


def test_normalize_returns_none_for_degenerate_shoulder_width():
    pose = _pose(
        {
            Landmark.LEFT_SHOULDER: Keypoint(0.4, 0.4, 0.9),
            Landmark.RIGHT_SHOULDER: Keypoint(0.4, 0.4, 0.9),  # coincident
        }
    )
    assert normalize_pose(pose) is None


def test_normalize_computes_scale_and_shoulder_offsets():
    pose = _pose(
        {
            Landmark.LEFT_SHOULDER: Keypoint(0.3, 0.4, 0.9),
            Landmark.RIGHT_SHOULDER: Keypoint(0.5, 0.4, 0.9),
        }
    )
    result = normalize_pose(pose)

    assert result is not None
    assert result.scale == pytest.approx(0.2)  # shoulder distance
    # center_x = 0.4; left shoulder at 0.3 -> offset -0.5 shoulder-widths
    assert result.left.shoulder_offset_x == pytest.approx(-0.5)
    assert result.right.shoulder_offset_x == pytest.approx(0.5)


def test_normalize_wrist_relative_is_none_when_wrist_not_visible():
    pose = _pose(
        {
            Landmark.LEFT_SHOULDER: Keypoint(0.3, 0.4, 0.9),
            Landmark.RIGHT_SHOULDER: Keypoint(0.5, 0.4, 0.9),
            Landmark.LEFT_WRIST: Keypoint(0.2, 0.2, 0.1),  # below visibility threshold
        }
    )
    result = normalize_pose(pose)

    assert result.left.wrist_relative is None


def test_normalize_wrist_relative_computed_in_shoulder_width_units():
    pose = _pose(
        {
            Landmark.LEFT_SHOULDER: Keypoint(0.3, 0.4, 0.9),
            Landmark.RIGHT_SHOULDER: Keypoint(0.5, 0.4, 0.9),
            Landmark.LEFT_WRIST: Keypoint(0.2, 0.2, 0.9),
        }
    )
    result = normalize_pose(pose)

    # scale = 0.2; rel = ((0.2-0.3)/0.2, (0.2-0.4)/0.2) = (-0.5, -1.0)
    assert result.left.wrist_relative == pytest.approx((-0.5, -1.0))
