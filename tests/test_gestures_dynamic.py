from pose_controller.gestures.dynamic_gestures import SweepDetector, VerticalSwipeDetector


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def advance(self, dt):
        self.t += dt
        return self.t

    def __call__(self):
        return self.t


def test_sweep_detected_for_wide_raised_horizontal_travel():
    clock = _FakeClock()
    detector = SweepDetector(time_fn=clock)

    # Wrist raised (rel_y well below -RAISE_Y_THRESHOLD) sweeping from
    # rel_x=-1.0 to rel_x=+1.0 over ~1 second (within the 1.2s window).
    results = []
    for i in range(11):
        clock.advance(0.1)
        rel_x = -1.0 + i * 0.2  # -1.0 .. 1.0
        results.append(detector.add_sample(rel_x=rel_x, rel_y=-1.0))

    assert any(results)


def test_no_sweep_when_not_raised():
    clock = _FakeClock()
    detector = SweepDetector(time_fn=clock)

    results = []
    for i in range(11):
        clock.advance(0.1)
        rel_x = -1.0 + i * 0.2
        results.append(detector.add_sample(rel_x=rel_x, rel_y=0.0))  # at shoulder height, not raised

    assert not any(results)


def test_no_sweep_when_range_too_narrow():
    clock = _FakeClock()
    detector = SweepDetector(time_fn=clock)

    results = []
    for i in range(11):
        clock.advance(0.1)
        rel_x = 0.0 + i * 0.01  # tiny drift, well under MIN_SWEEP_RANGE
        results.append(detector.add_sample(rel_x=rel_x, rel_y=-1.0))

    assert not any(results)


def test_no_sweep_when_samples_are_too_bursty():
    # Wide range and raised, but all crammed into a tiny time span --
    # shouldn't count as a deliberate sweep (guards against a couple of
    # noisy outlier frames rather than a real, unfolding motion).
    clock = _FakeClock()
    detector = SweepDetector(time_fn=clock)

    results = []
    for rel_x in (-1.0, 0.0, 1.0):
        clock.advance(0.01)
        results.append(detector.add_sample(rel_x=rel_x, rel_y=-1.0))

    assert not any(results)


def test_history_clears_after_detected_sweep():
    clock = _FakeClock()
    detector = SweepDetector(time_fn=clock)

    for i in range(11):
        clock.advance(0.1)
        rel_x = -1.0 + i * 0.2
        detector.add_sample(rel_x=rel_x, rel_y=-1.0)

    # Immediately after a detected sweep, a single new sample alone
    # shouldn't itself look like a sweep (too few samples).
    clock.advance(0.1)
    assert detector.add_sample(rel_x=0.0, rel_y=-1.0) is False


def test_reset_clears_history():
    clock = _FakeClock()
    detector = SweepDetector(time_fn=clock)

    for i in range(5):
        clock.advance(0.1)
        detector.add_sample(rel_x=-1.0 + i * 0.2, rel_y=-1.0)

    detector.reset()

    clock.advance(0.1)
    assert detector.add_sample(rel_x=0.0, rel_y=-1.0) is False


def test_vertical_swipe_up_detected_for_wrist_rising_in_front_of_body():
    clock = _FakeClock()
    detector = VerticalSwipeDetector(direction="up", time_fn=clock)

    # rel_y falls from 1.0 (low/at-rest) to -1.0 (raised) -- an "up" swipe --
    # while staying centered (rel_x near 0, well under SIDE_X_THRESHOLD).
    results = []
    for i in range(11):
        clock.advance(0.1)
        rel_y = 1.0 - i * 0.2  # 1.0 .. -1.0
        results.append(detector.add_sample(rel_x=0.0, rel_y=rel_y))

    assert any(results)


def test_vertical_swipe_down_detected_for_wrist_falling_in_front_of_body():
    clock = _FakeClock()
    detector = VerticalSwipeDetector(direction="down", time_fn=clock)

    results = []
    for i in range(11):
        clock.advance(0.1)
        rel_y = -1.0 + i * 0.2  # -1.0 .. 1.0
        results.append(detector.add_sample(rel_x=0.0, rel_y=rel_y))

    assert any(results)


def test_vertical_swipe_not_detected_for_wrong_direction():
    clock = _FakeClock()
    up_detector = VerticalSwipeDetector(direction="up", time_fn=clock)

    # Wrist moving DOWN should never satisfy the "up" detector.
    results = []
    for i in range(11):
        clock.advance(0.1)
        rel_y = -1.0 + i * 0.2  # -1.0 .. 1.0 (downward)
        results.append(up_detector.add_sample(rel_x=0.0, rel_y=rel_y))

    assert not any(results)


def test_vertical_swipe_not_detected_when_range_too_narrow():
    clock = _FakeClock()
    detector = VerticalSwipeDetector(direction="up", time_fn=clock)

    results = []
    for i in range(11):
        clock.advance(0.1)
        rel_y = 0.0 - i * 0.01  # tiny drift, well under MIN_VERTICAL_SWIPE_RANGE
        results.append(detector.add_sample(rel_x=0.0, rel_y=rel_y))

    assert not any(results)


def test_vertical_swipe_up_not_detected_when_confined_to_resting_range():
    # Real-footage regression guard (scripts/qnn_spike.md): a sufficiently
    # large relative delta occurring entirely within the arm's normal
    # resting range (never actually reaching raised territory) must not
    # count, even though range and direction alone would be satisfied.
    clock = _FakeClock()
    detector = VerticalSwipeDetector(direction="up", time_fn=clock)

    results = []
    for i in range(11):
        clock.advance(0.1)
        rel_y = 3.0 - i * 0.2  # 3.0 .. 1.0 -- big range, but never below -0.4
        results.append(detector.add_sample(rel_x=0.0, rel_y=rel_y))

    assert not any(results)


def test_vertical_swipe_down_not_detected_when_confined_to_resting_range():
    clock = _FakeClock()
    detector = VerticalSwipeDetector(direction="down", time_fn=clock)

    results = []
    for i in range(11):
        clock.advance(0.1)
        rel_y = 1.0 + i * 0.2  # 1.0 .. 3.0 -- big range, but never below -0.4 to start
        results.append(detector.add_sample(rel_x=0.0, rel_y=rel_y))

    assert not any(results)


def test_vertical_swipe_not_detected_when_strayed_out_to_side():
    clock = _FakeClock()
    detector = VerticalSwipeDetector(direction="up", time_fn=clock)

    # Good vertical range, but rel_x drifts past SIDE_X_THRESHOLD (0.5) --
    # not "in front of the body" anymore, so this shouldn't count even
    # though it would otherwise satisfy an "up" swipe.
    results = []
    for i in range(11):
        clock.advance(0.1)
        rel_y = 1.0 - i * 0.2
        results.append(detector.add_sample(rel_x=0.7, rel_y=rel_y))

    assert not any(results)


def test_vertical_swipe_not_detected_when_samples_are_too_bursty():
    clock = _FakeClock()
    detector = VerticalSwipeDetector(direction="up", time_fn=clock)

    results = []
    for rel_y in (1.0, 0.0, -1.0):
        clock.advance(0.01)
        results.append(detector.add_sample(rel_x=0.0, rel_y=rel_y))

    assert not any(results)


def test_vertical_swipe_history_clears_after_detected_swipe():
    clock = _FakeClock()
    detector = VerticalSwipeDetector(direction="up", time_fn=clock)

    for i in range(11):
        clock.advance(0.1)
        detector.add_sample(rel_x=0.0, rel_y=1.0 - i * 0.2)

    clock.advance(0.1)
    assert detector.add_sample(rel_x=0.0, rel_y=-1.0) is False


def test_vertical_swipe_reset_clears_history():
    clock = _FakeClock()
    detector = VerticalSwipeDetector(direction="up", time_fn=clock)

    for i in range(5):
        clock.advance(0.1)
        detector.add_sample(rel_x=0.0, rel_y=1.0 - i * 0.2)

    detector.reset()

    clock.advance(0.1)
    assert detector.add_sample(rel_x=0.0, rel_y=-1.0) is False


def test_vertical_swipe_rejects_invalid_direction():
    import pytest

    with pytest.raises(ValueError):
        VerticalSwipeDetector(direction="sideways")
