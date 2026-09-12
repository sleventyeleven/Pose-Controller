import numpy as np

from pose_controller.inference.pose import Keypoint, PoseResult
from pose_controller.overlay import draw_poses


def test_draw_poses_noop_on_empty_list():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = draw_poses(frame, [])
    assert np.array_equal(result, frame)


def test_draw_poses_draws_visible_keypoints_only():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pose = PoseResult(
        keypoints={
            11: Keypoint(x=0.3, y=0.3, visibility=0.9),
            12: Keypoint(x=0.7, y=0.3, visibility=0.9),
            13: Keypoint(x=0.2, y=0.5, visibility=0.1),  # below visibility threshold
        }
    )

    result = draw_poses(frame, [pose])

    assert result.sum() > 0


def test_draw_poses_handles_multiple_people():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pose_a = PoseResult(
        keypoints={11: Keypoint(x=0.2, y=0.2, visibility=0.9), 12: Keypoint(x=0.3, y=0.2, visibility=0.9)},
        track_id=1,
    )
    pose_b = PoseResult(
        keypoints={11: Keypoint(x=0.7, y=0.7, visibility=0.9), 12: Keypoint(x=0.8, y=0.7, visibility=0.9)},
        track_id=2,
    )

    result = draw_poses(frame, [pose_a, pose_b])

    assert result.sum() > 0
