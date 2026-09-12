from pose_controller.gestures.state_machine import DEBOUNCE_FRAMES, GestureStateMachine
from pose_controller.gestures.types import ArmPose, ControlAction
from pose_controller.inference.pose import Keypoint, Landmark, PoseResult


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def advance(self, dt):
        self.t += dt
        return self.t

    def __call__(self):
        return self.t


_SHOULDERS = {
    Landmark.LEFT_SHOULDER: Keypoint(0.3, 0.4, 0.9),
    Landmark.RIGHT_SHOULDER: Keypoint(0.5, 0.4, 0.9),
}


def _pose(track_id, extra=None, shoulders=_SHOULDERS):
    keypoints = {int(lm): kp for lm, kp in shoulders.items()}
    if extra:
        keypoints.update({int(lm): kp for lm, kp in extra.items()})
    result = PoseResult(keypoints=keypoints)
    result.track_id = track_id
    return result


def _right_out_pose(track_id):
    # right shoulder at x=0.5, offset_x=+0.5; wrist further positive (outward), shoulder height.
    return _pose(track_id, {Landmark.RIGHT_WRIST: Keypoint(0.8, 0.4, 0.9)})


def _left_out_pose(track_id):
    return _pose(track_id, {Landmark.LEFT_WRIST: Keypoint(0.0, 0.4, 0.9)})


def _both_raised_pose(track_id):
    return _pose(
        track_id,
        {
            Landmark.LEFT_WRIST: Keypoint(0.3, 0.1, 0.9),
            Landmark.RIGHT_WRIST: Keypoint(0.5, 0.1, 0.9),
        },
    )


def _occluded_pose(track_id):
    """No wrist keypoints at all -- exercises classify_arm returning None
    (genuinely unknown this frame), not an observed DOWN."""
    return _pose(track_id, {})


def _down_pose(track_id):
    """Wrists actually observed hanging below the shoulders -- a real
    DOWN classification, distinct from `_occluded_pose`'s "no data"."""
    return _pose(
        track_id,
        {
            Landmark.LEFT_WRIST: Keypoint(0.3, 0.6, 0.9),
            Landmark.RIGHT_WRIST: Keypoint(0.5, 0.6, 0.9),
        },
    )


def test_right_arm_out_triggers_next_once_on_confirmation():
    gsm = GestureStateMachine()
    fired = []

    for _ in range(DEBOUNCE_FRAMES + 2):
        fired.extend(gsm.update_all([_right_out_pose(1)]))

    next_events = [a for _, a in fired if a == ControlAction.NEXT]
    assert len(next_events) == 1  # fires once on confirmation, not repeatedly while held


def test_left_arm_out_triggers_previous():
    gsm = GestureStateMachine()
    fired = []

    for _ in range(DEBOUNCE_FRAMES + 1):
        fired.extend(gsm.update_all([_left_out_pose(1)]))

    assert (1, ControlAction.PREVIOUS) in fired


def test_both_arms_raised_triggers_play_pause_once():
    gsm = GestureStateMachine()
    fired = []

    for _ in range(DEBOUNCE_FRAMES + 2):
        fired.extend(gsm.update_all([_both_raised_pose(1)]))

    play_pause_events = [a for _, a in fired if a == ControlAction.PLAY_PAUSE]
    assert len(play_pause_events) == 1


def test_single_raised_arm_does_not_trigger_play_pause():
    gsm = GestureStateMachine()
    fired = []

    for _ in range(DEBOUNCE_FRAMES + 2):
        pose = _pose(1, {Landmark.RIGHT_WRIST: Keypoint(0.5, 0.1, 0.9)})
        fired.extend(gsm.update_all([pose]))

    assert not any(a == ControlAction.PLAY_PAUSE for _, a in fired)


def test_returning_to_neutral_allows_retrigger():
    gsm = GestureStateMachine()

    for _ in range(DEBOUNCE_FRAMES + 1):
        gsm.update_all([_right_out_pose(1)])
    for _ in range(DEBOUNCE_FRAMES + 1):
        gsm.update_all([_down_pose(1)])

    fired = []
    for _ in range(DEBOUNCE_FRAMES + 1):
        fired.extend(gsm.update_all([_right_out_pose(1)]))

    assert any(a == ControlAction.NEXT for _, a in fired)


def test_brief_occlusion_mid_debounce_does_not_reset_progress():
    gsm = GestureStateMachine()

    # DEBOUNCE_FRAMES - 1 hits, then one frame with no wrist data, then
    # enough further hits to reach DEBOUNCE_FRAMES total -- should still
    # confirm rather than restarting the count from zero.
    for _ in range(DEBOUNCE_FRAMES - 1):
        gsm.update_all([_right_out_pose(1)])

    gsm.update_all([_occluded_pose(1)])  # wrist not visible this frame

    fired = []
    fired.extend(gsm.update_all([_right_out_pose(1)]))

    assert any(a == ControlAction.NEXT for _, a in fired)


def test_sweep_triggers_skip():
    clock = _FakeClock()
    gsm = GestureStateMachine(time_fn=clock)

    fired = []
    for i in range(11):
        clock.advance(0.1)
        # Sweep the right wrist across a wide raised horizontal range.
        rel_progress = -1.0 + i * 0.2
        wrist_x = 0.5 + rel_progress * 0.2  # shoulder x=0.5, scale=0.2 -> rel_x = rel_progress
        pose = _pose(1, {Landmark.RIGHT_WRIST: Keypoint(wrist_x, 0.0, 0.9)})  # rel_y = (0.0-0.4)/0.2 = -2.0, raised
        fired.extend(gsm.update_all([pose]))

    assert any(a == ControlAction.SKIP for _, a in fired)


def test_swipe_up_triggers_volume_up():
    clock = _FakeClock()
    gsm = GestureStateMachine(time_fn=clock)

    fired = []
    for i in range(11):
        clock.advance(0.1)
        # Right wrist rising from below the shoulder to above it, centered
        # (shoulder x=0.5) -- an "up" swipe in front of the body.
        wrist_y = 0.6 - i * 0.04  # 0.6 .. 0.2, i.e. rel_y = (wrist_y-0.4)/0.2 falls 1.0 -> -1.0
        pose = _pose(1, {Landmark.RIGHT_WRIST: Keypoint(0.5, wrist_y, 0.9)})
        fired.extend(gsm.update_all([pose]))

    assert any(a == ControlAction.VOLUME_UP for _, a in fired)


def test_swipe_down_triggers_volume_down():
    clock = _FakeClock()
    gsm = GestureStateMachine(time_fn=clock)

    fired = []
    for i in range(11):
        clock.advance(0.1)
        # Left wrist falling from above the shoulder to below it, centered
        # (shoulder x=0.3) -- a "down" swipe in front of the body.
        wrist_y = 0.2 + i * 0.04  # 0.2 .. 0.6, i.e. rel_y rises -1.0 -> 1.0
        pose = _pose(1, {Landmark.LEFT_WRIST: Keypoint(0.3, wrist_y, 0.9)})
        fired.extend(gsm.update_all([pose]))

    assert any(a == ControlAction.VOLUME_DOWN for _, a in fired)


def test_volume_up_does_not_retrigger_while_raised_is_held():
    # Real-footage regression guard (scripts/qnn_spike.md): a raised arm
    # held for over a second, with ordinary jitter, kept re-firing
    # VOLUME_UP roughly every 0.6s -- jitter alone was enough to redevelop
    # a fresh 0.8-shoulder-width range each time. Once RAISED confirms,
    # evaluation should stop entirely until the arm comes back down.
    clock = _FakeClock()
    gsm = GestureStateMachine(time_fn=clock)

    fired = []
    # Rise from resting to well past raised -- triggers VOLUME_UP once
    # and confirms RAISED along the way.
    for i in range(8):
        clock.advance(0.1)
        rel_y = 1.0 - i * 0.4  # 1.0 .. -1.8
        wrist_y = 0.4 + rel_y * 0.2  # shoulder y=0.4, scale=0.2
        pose = _pose(1, {Landmark.RIGHT_WRIST: Keypoint(0.5, wrist_y, 0.9)})
        fired.extend(gsm.update_all([pose]))

    assert gsm._tracks[1].right.confirmed == ArmPose.RAISED
    assert sum(1 for _, a in fired if a == ControlAction.VOLUME_UP) == 1

    # Hold raised with jitter comfortably exceeding MIN_VERTICAL_SWIPE_RANGE
    # if it were still being evaluated -- must not fire again.
    for i in range(20):
        clock.advance(0.1)
        rel_y = -1.5 if i % 2 == 0 else -2.5  # jitter range 1.0, > 0.8
        wrist_y = 0.4 + rel_y * 0.2
        pose = _pose(1, {Landmark.RIGHT_WRIST: Keypoint(0.5, wrist_y, 0.9)})
        fired.extend(gsm.update_all([pose]))

    assert sum(1 for _, a in fired if a == ControlAction.VOLUME_UP) == 1


def test_stale_track_state_is_pruned_and_requires_fresh_debounce():
    gsm = GestureStateMachine()

    for _ in range(DEBOUNCE_FRAMES + 1):
        gsm.update_all([_right_out_pose(1)])
    assert 1 in gsm._tracks

    # Track 1 absent for longer than STALE_TRACK_FRAMES -- only track 2 updates.
    for _ in range(301):
        gsm.update_all([_occluded_pose(2)])
    assert 1 not in gsm._tracks
