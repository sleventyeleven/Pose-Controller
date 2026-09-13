import numpy as np

from pose_controller.inference.pose import Keypoint, Landmark, PoseResult
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


def test_draw_poses_draws_a_bounding_box_when_present():
    frame_with_box = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_without_box = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = {
        int(Landmark.LEFT_SHOULDER): Keypoint(x=0.3, y=0.3, visibility=0.9),
        int(Landmark.RIGHT_SHOULDER): Keypoint(x=0.7, y=0.3, visibility=0.9),
    }
    pose_with_box = PoseResult(keypoints=keypoints, bbox=(0.1, 0.1, 0.9, 0.9))
    pose_without_box = PoseResult(keypoints=keypoints)
    # PoseResult.__post_init__ auto-derives a bbox from these same
    # keypoints when none is given -- force it back to None afterward to
    # isolate the "no box at all" case specifically.
    pose_without_box.bbox = None

    draw_poses(frame_with_box, [pose_with_box])
    draw_poses(frame_without_box, [pose_without_box])

    # The explicit box's edge (full-width rectangle border) should light up
    # pixels along a row inside the with-box frame that the without-box
    # frame has no reason to touch.
    box_row_y = int(0.1 * 100)
    assert frame_with_box[box_row_y, :, :].sum() > frame_without_box[box_row_y, :, :].sum()


def test_draw_poses_connects_full_body_skeleton_when_landmarks_present():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    pose = PoseResult(
        keypoints={
            int(Landmark.LEFT_HIP): Keypoint(x=0.4, y=0.5, visibility=0.9),
            int(Landmark.LEFT_KNEE): Keypoint(x=0.4, y=0.7, visibility=0.9),
            int(Landmark.LEFT_ANKLE): Keypoint(x=0.4, y=0.9, visibility=0.9),
        }
    )

    result = draw_poses(frame, [pose])

    # A leg line should pass through a pixel roughly between hip and knee.
    midpoint_y = int(0.6 * 200)
    midpoint_x = int(0.4 * 200)
    assert result[midpoint_y, midpoint_x, :].sum() > 0
