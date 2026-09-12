from pose_controller.app import update_held_overlay
from pose_controller.gestures.types import ArmPose
from pose_controller.inference.pose import Keypoint, PoseResult


def _pose(track_id, x=0.5):
    result = PoseResult(keypoints={0: Keypoint(x, 0.5, 0.9), 1: Keypoint(x, 0.6, 0.9)})
    result.track_id = track_id
    return result


def _pose_box(track_id, x0, y0, x1, y1):
    """Like `_pose`, but with a genuine (non-zero-area) bounding box, for
    exercising the overlap-dedup logic -- `_pose`'s degenerate zero-width
    box always has IoU 0 against anything, so it can't."""
    result = PoseResult(keypoints={0: Keypoint(x0, y0, 0.9), 1: Keypoint(x1, y1, 0.9)})
    result.track_id = track_id
    return result


def test_a_fresh_detection_is_drawn_directly_with_no_holding():
    held_overlay = {}
    pose = _pose(1)

    display = update_held_overlay(held_overlay, [pose], {}, max_age=30)

    assert display == [pose]
    assert 1 in held_overlay


def test_a_missed_frame_still_draws_the_last_known_pose():
    held_overlay = {}
    pose = _pose(1)
    update_held_overlay(held_overlay, [pose], {}, max_age=30)

    # This frame: no detection for track 1 at all.
    display = update_held_overlay(held_overlay, [], {}, max_age=30)

    assert display == [pose]


def test_a_fresh_detection_replaces_the_held_pose_not_duplicates_it():
    held_overlay = {}
    pose1 = _pose(1, x=0.5)
    update_held_overlay(held_overlay, [pose1], {}, max_age=30)
    update_held_overlay(held_overlay, [], {}, max_age=30)  # one missed frame

    pose2 = _pose(1, x=0.6)
    display = update_held_overlay(held_overlay, [pose2], {}, max_age=30)

    assert display == [pose2]  # only the fresh one, not both


def test_held_pose_is_dropped_after_max_age_missed_frames():
    held_overlay = {}
    pose = _pose(1)
    update_held_overlay(held_overlay, [pose], {}, max_age=3)

    for _ in range(3):
        display = update_held_overlay(held_overlay, [], {}, max_age=3)
        assert display == [pose]  # still within max_age

    display = update_held_overlay(held_overlay, [], {}, max_age=3)

    assert display == []
    assert 1 not in held_overlay


def test_held_arm_states_are_carried_forward_into_the_output_dict():
    held_overlay = {}
    pose = _pose(1)
    arm_states = {1: (ArmPose.DOWN, ArmPose.OUT_TO_SIDE)}
    update_held_overlay(held_overlay, [pose], arm_states, max_age=30)

    next_arm_states = {}
    update_held_overlay(held_overlay, [], next_arm_states, max_age=30)

    assert next_arm_states[1] == (ArmPose.DOWN, ArmPose.OUT_TO_SIDE)


def test_multiple_tracks_are_held_independently():
    held_overlay = {}
    pose1 = _pose(1)
    pose2 = _pose(2)
    update_held_overlay(held_overlay, [pose1, pose2], {}, max_age=30)

    # Only track 1 detected this frame -- track 2 should still be held.
    display = update_held_overlay(held_overlay, [pose1], {}, max_age=30)

    # PoseResult isn't hashable, so compare membership rather than as a set.
    assert len(display) == 2
    assert pose1 in display
    assert pose2 in display


def test_poses_with_no_track_id_are_neither_held_nor_dropped_early():
    held_overlay = {}
    untracked = PoseResult(keypoints={0: Keypoint(0.5, 0.5, 0.9), 1: Keypoint(0.5, 0.6, 0.9)})
    assert untracked.track_id is None

    display = update_held_overlay(held_overlay, [untracked], {}, max_age=30)

    assert display == [untracked]
    assert held_overlay == {}


def test_fresh_detection_dedupes_an_overlapping_held_ghost():
    # Regression guard for a real observation: a fragmenting tracker
    # (e.g. from a poorly-framed camera feeding inconsistent partial-body
    # crops to IoU matching and re-ID) spawns a new track_id for what's
    # really the same continuously-present person, and without dedup the
    # old track's held ghost keeps drawing on top of the new one.
    held_overlay = {}
    pose1 = _pose_box(1, 0.40, 0.40, 0.60, 0.80)
    update_held_overlay(held_overlay, [pose1], {}, max_age=30)  # track 1 now held after this

    # Track 2 appears in nearly the same spot the very next frame.
    pose2 = _pose_box(2, 0.42, 0.42, 0.62, 0.82)
    display = update_held_overlay(held_overlay, [pose2], {}, max_age=30)

    assert display == [pose2]  # track 1's ghost is suppressed, not stacked underneath


def test_overlapping_held_ghosts_keep_only_the_least_stale_one():
    held_overlay = {}
    pose1 = _pose_box(1, 0.40, 0.40, 0.60, 0.80)
    pose2 = _pose_box(2, 0.42, 0.42, 0.62, 0.82)
    update_held_overlay(held_overlay, [pose1], {}, max_age=30)  # track 1: stale=0 this frame
    update_held_overlay(held_overlay, [pose2], {}, max_age=30)  # track 1: stale=1, track 2: stale=0

    # Neither detected this frame -- both now purely held ghosts, track 1
    # staler (2 missed frames) than track 2 (1 missed frame).
    display = update_held_overlay(held_overlay, [], {}, max_age=30)

    assert display == [pose2]


def test_non_overlapping_poses_are_not_deduped():
    held_overlay = {}
    pose1 = _pose_box(1, 0.0, 0.0, 0.2, 0.3)
    pose2 = _pose_box(2, 0.7, 0.7, 0.9, 0.9)
    update_held_overlay(held_overlay, [pose1, pose2], {}, max_age=30)

    display = update_held_overlay(held_overlay, [pose1], {}, max_age=30)  # pose2 now held

    assert len(display) == 2
    assert pose1 in display
    assert pose2 in display
