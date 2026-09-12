from pose_controller.gestures.dynamic_gestures import SweepDetector


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
