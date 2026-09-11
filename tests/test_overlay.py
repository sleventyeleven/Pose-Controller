import numpy as np

from pose_controller.inference.pose import Keypoint, PoseResult
from pose_controller.overlay import draw_pose


def test_draw_pose_noop_on_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = draw_pose(frame, None)
    assert np.array_equal(result, frame)


def test_draw_pose_draws_visible_keypoints_only():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pose = PoseResult(
        keypoints={
            11: Keypoint(x=0.3, y=0.3, visibility=0.9),
            12: Keypoint(x=0.7, y=0.3, visibility=0.9),
            13: Keypoint(x=0.2, y=0.5, visibility=0.1),  # below visibility threshold
        }
    )

    result = draw_pose(frame, pose)

    assert result.sum() > 0
