from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from pose_controller.gestures.static_poses import RAISE_Y_THRESHOLD

# Not empirically tuned against real camera footage yet -- see
# docs/backlog.md and static_poses.py's threshold comment.
WINDOW_SECONDS = 1.2  # how far back "recent" wrist history extends
MIN_SWEEP_RANGE = 1.2  # shoulder-widths of horizontal travel required
MIN_RAISED_FRACTION = 0.8  # fraction of the window the wrist must stay raised
MIN_WINDOW_COVERAGE = 0.5  # samples must span at least this fraction of WINDOW_SECONDS


@dataclass
class _WristSample:
    timestamp: float
    rel_x: float
    rel_y: float


class SweepDetector:
    """Detects a one-armed overhead sweep: the wrist stays raised for most
    of a rolling time window while covering a substantial horizontal
    range.

    This is a first-pass heuristic (range + raised-fraction over a
    window), not a full trajectory/direction analysis -- it doesn't
    distinguish a clean single sweep from, say, vigorously waving a raised
    arm back and forth. Good enough to validate the pipeline end-to-end;
    refining it needs real gesture footage to tune against (see
    docs/backlog.md).

    One instance tracks one wrist for one track_id -- see
    `state_machine.GestureStateMachine`, which owns one `SweepDetector`
    per (track_id, arm).
    """

    def __init__(
        self,
        window_seconds: float = WINDOW_SECONDS,
        min_range: float = MIN_SWEEP_RANGE,
        min_raised_fraction: float = MIN_RAISED_FRACTION,
        raise_threshold: float = RAISE_Y_THRESHOLD,
        time_fn=time.monotonic,
    ):
        self._window_seconds = window_seconds
        self._min_range = min_range
        self._min_raised_fraction = min_raised_fraction
        self._raise_threshold = raise_threshold
        self._time_fn = time_fn
        self._samples: deque[_WristSample] = deque()

    def add_sample(self, rel_x: float, rel_y: float) -> bool:
        """Record one frame's (rel_x, rel_y) wrist-relative-to-shoulder
        position (see normalize.ArmGeometry). Returns True the moment a
        sweep is detected, and clears history so the same physical sweep
        can't re-trigger on the very next frame."""
        now = self._time_fn()
        self._samples.append(_WristSample(now, rel_x, rel_y))
        self._prune(now)

        if self._detect():
            self._samples.clear()
            return True
        return False

    def reset(self) -> None:
        """Clear history -- call when the arm is no longer raised at all,
        so an old partial sweep doesn't contribute to a later one."""
        self._samples.clear()

    def _prune(self, now: float) -> None:
        while self._samples and now - self._samples[0].timestamp > self._window_seconds:
            self._samples.popleft()

    def _detect(self) -> bool:
        if len(self._samples) < 3:
            return False

        duration = self._samples[-1].timestamp - self._samples[0].timestamp
        if duration < self._window_seconds * MIN_WINDOW_COVERAGE:
            return False

        raised_count = sum(1 for s in self._samples if s.rel_y < -self._raise_threshold)
        if raised_count / len(self._samples) < self._min_raised_fraction:
            return False

        xs = [s.rel_x for s in self._samples]
        return (max(xs) - min(xs)) >= self._min_range
