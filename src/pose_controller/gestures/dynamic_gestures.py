from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from pose_controller.gestures.static_poses import RAISE_Y_THRESHOLD, SIDE_X_THRESHOLD

# Not empirically tuned against real camera footage yet -- see
# docs/backlog.md and static_poses.py's threshold comment.
WINDOW_SECONDS = 1.2  # how far back "recent" wrist history extends
MIN_SWEEP_RANGE = 1.2  # shoulder-widths of horizontal travel required
MIN_RAISED_FRACTION = 0.8  # fraction of the window the wrist must stay raised
MIN_WINDOW_COVERAGE = 0.5  # samples must span at least this fraction of WINDOW_SECONDS
MIN_VERTICAL_SWIPE_RANGE = 0.8  # shoulder-widths of vertical travel required for a volume swipe
# Reuses SIDE_X_THRESHOLD (the same boundary static_poses.classify_arm uses
# for "out to side") so "in front of the body" means exactly "wouldn't
# also register as OUT_TO_SIDE", rather than an independently-guessed value.
MAX_VOLUME_SWIPE_HORIZONTAL_DRIFT = SIDE_X_THRESHOLD


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


class VerticalSwipeDetector:
    """Detects a vertical wrist swipe (up or down) roughly in front of the
    body: the wrist covers a substantial vertical range in the requested
    direction within a rolling time window, ending (for "up") or starting
    (for "down") in genuinely raised territory -- not just anywhere a
    sufficiently large relative delta happens to occur -- while staying
    close enough to centered horizontally that it wouldn't also read as
    an OUT_TO_SIDE motion.

    Unlike `SweepDetector` (which requires staying raised throughout the
    window, since the overhead sweep *is* a held-up motion), this detector
    is defined by the wrist *changing* height -- there's no "stay in one
    state" requirement, and direction is checked with a simple first-vs-
    last-sample comparison rather than a full trajectory analysis. This is
    a first-pass heuristic like `SweepDetector`, not fully validated
    against real gesture footage.

    The raised-endpoint requirement (added after a real-footage re-run,
    see scripts/qnn_spike.md) fixes one concrete failure mode found there:
    without it, ordinary pose-estimation jitter or a slow drift confined
    entirely within the arm's normal resting range could satisfy the
    range+direction check on its own, since nothing required the motion
    to actually reach raised territory. It does *not* by itself stop a
    held raised pose's jitter from re-satisfying the check on its own
    (the endpoint stays "raised" throughout a hold) -- that's what
    `state_machine.GestureStateMachine` resetting this detector on the
    corresponding static RAISED confirmation is for.

    One instance tracks one wrist for one track_id and one direction --
    see `state_machine.GestureStateMachine`, which owns one per
    (track_id, arm, direction) pairing actually in use (right/up for
    volume-up, left/down for volume-down).
    """

    def __init__(
        self,
        direction: str,
        window_seconds: float = WINDOW_SECONDS,
        min_range: float = MIN_VERTICAL_SWIPE_RANGE,
        max_horizontal_drift: float = MAX_VOLUME_SWIPE_HORIZONTAL_DRIFT,
        raise_threshold: float = RAISE_Y_THRESHOLD,
        time_fn=time.monotonic,
    ):
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
        self._direction = direction
        self._window_seconds = window_seconds
        self._min_range = min_range
        self._max_horizontal_drift = max_horizontal_drift
        self._raise_threshold = raise_threshold
        self._time_fn = time_fn
        self._samples: deque[_WristSample] = deque()

    def add_sample(self, rel_x: float, rel_y: float) -> bool:
        """Record one frame's (rel_x, rel_y) wrist-relative-to-shoulder
        position (see normalize.ArmGeometry). Returns True the moment a
        swipe is detected, and clears history so the same physical swipe
        can't re-trigger on the very next frame."""
        now = self._time_fn()
        self._samples.append(_WristSample(now, rel_x, rel_y))
        self._prune(now)

        if self._detect():
            self._samples.clear()
            return True
        return False

    def reset(self) -> None:
        """Clear history -- call when the arm returns to a clearly
        unrelated position, so an old partial swipe doesn't contribute to
        a later one."""
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

        if max(abs(s.rel_x) for s in self._samples) > self._max_horizontal_drift:
            return False  # strayed out to the side -- not "in front of the body"

        first_y = self._samples[0].rel_y
        last_y = self._samples[-1].rel_y
        # Image y increases downward, so "up" means rel_y decreasing.
        # Beyond just covering enough range in the right direction, the
        # relevant endpoint must actually reach raised territory (mirrors
        # static_poses.classify_arm's own RAISED boundary) -- otherwise a
        # sufficiently large relative delta confined entirely within the
        # arm's normal resting range would pass, which real footage showed
        # does happen from ordinary pose-estimation jitter/drift alone
        # (see scripts/qnn_spike.md).
        if self._direction == "up":
            if last_y >= -self._raise_threshold:
                return False
            return (first_y - last_y) >= self._min_range
        if first_y >= -self._raise_threshold:
            return False
        return (last_y - first_y) >= self._min_range
