from __future__ import annotations

import time
from dataclasses import dataclass, field

from pose_controller.gestures.dynamic_gestures import SweepDetector, VerticalSwipeDetector
from pose_controller.gestures.normalize import normalize_pose
from pose_controller.gestures.static_poses import classify_arm
from pose_controller.gestures.types import ArmPose, ControlAction
from pose_controller.inference.pose import PoseResult

DEBOUNCE_FRAMES = 4  # consecutive frames a new raw classification must hold before it's confirmed
STALE_TRACK_FRAMES = 300  # frames a track can go unseen before its gesture state is forgotten


@dataclass
class _ArmDebouncer:
    """Hysteresis for one arm's static pose: a new raw classification must
    hold for DEBOUNCE_FRAMES consecutive frames before it's confirmed,
    filtering out single-frame jitter/noise. A raw classification of None
    (arm not visible this frame -- see static_poses.classify_arm) neither
    advances nor resets pending progress, tolerating brief occlusion mid
    debounce."""

    confirmed: ArmPose = ArmPose.DOWN
    _pending: ArmPose | None = None
    _pending_count: int = 0

    def update(self, raw: ArmPose | None) -> bool:
        """Returns True if `confirmed` changed as a result of this call."""
        if raw is None:
            return False  # hold pending progress through a momentary occlusion

        if raw == self.confirmed:
            self._pending = None
            self._pending_count = 0
            return False

        if raw == self._pending:
            self._pending_count += 1
        else:
            self._pending = raw
            self._pending_count = 1

        if self._pending_count >= DEBOUNCE_FRAMES:
            self.confirmed = raw
            self._pending = None
            self._pending_count = 0
            return True
        return False


@dataclass
class _TrackGestureState:
    left: _ArmDebouncer = field(default_factory=_ArmDebouncer)
    right: _ArmDebouncer = field(default_factory=_ArmDebouncer)
    left_sweep: SweepDetector = field(default_factory=SweepDetector)
    right_sweep: SweepDetector = field(default_factory=SweepDetector)
    # Only the two directions the project actually maps to an action --
    # right arm up for volume-up, left arm down for volume-down (see
    # docs/backlog.md; not implemented symmetrically for the other two
    # arm/direction combinations since nothing requested them).
    right_volume_swipe: VerticalSwipeDetector = field(
        default_factory=lambda: VerticalSwipeDetector(direction="up")
    )
    left_volume_swipe: VerticalSwipeDetector = field(
        default_factory=lambda: VerticalSwipeDetector(direction="down")
    )
    last_seen_frame: int = 0

    @classmethod
    def with_clock(cls, time_fn) -> "_TrackGestureState":
        return cls(
            left_sweep=SweepDetector(time_fn=time_fn),
            right_sweep=SweepDetector(time_fn=time_fn),
            right_volume_swipe=VerticalSwipeDetector(direction="up", time_fn=time_fn),
            left_volume_swipe=VerticalSwipeDetector(direction="down", time_fn=time_fn),
        )


class GestureStateMachine:
    """Per-track pose-state machine: turns normalized arm geometry into
    edge-triggered `ControlAction`s.

    Mapping (per the project's stated requirements -- see docs/backlog.md
    for what's *not* explicitly specified, like single-arm-raised):
      - right arm confirmed OUT_TO_SIDE -> NEXT
      - left arm confirmed OUT_TO_SIDE -> PREVIOUS
      - both arms confirmed RAISED (simultaneously) -> PLAY_PAUSE
      - one-armed overhead sweep (either arm) -> SKIP
      - right arm swipe up in front of the body -> VOLUME_UP
      - left arm swipe down in front of the body -> VOLUME_DOWN

    All triggers are edge-triggered (fire once on the transition into the
    triggering state), not level-triggered -- holding a pose doesn't
    repeat-fire its action; the arm must return to a different state and
    re-enter the trigger state to fire again.

    Call `update_all` once per frame with that frame's tracked poses (from
    `tracking.Tracker.update`); it also prunes gesture state for track_ids
    that haven't appeared in `STALE_TRACK_FRAMES` frames, so a long-running
    session doesn't accumulate state for people who left permanently.
    """

    def __init__(self, time_fn=time.monotonic) -> None:
        self._tracks: dict[int, _TrackGestureState] = {}
        self._frame_index = 0
        self._time_fn = time_fn

    def update_all(self, tracked_poses: list[PoseResult]) -> list[tuple[int, ControlAction]]:
        self._frame_index += 1
        triggered: list[tuple[int, ControlAction]] = []

        for pose in tracked_poses:
            if pose.track_id is None:
                continue
            triggered.extend(
                (pose.track_id, action) for action in self._update_track(pose.track_id, pose)
            )

        self._prune_stale_tracks()
        return triggered

    def _update_track(self, track_id: int, pose: PoseResult) -> list[ControlAction]:
        if track_id not in self._tracks:
            self._tracks[track_id] = _TrackGestureState.with_clock(self._time_fn)
        state = self._tracks[track_id]
        state.last_seen_frame = self._frame_index

        normalized = normalize_pose(pose)
        if normalized is None:
            return []

        actions: list[ControlAction] = []

        # right_volume_swipe is only evaluated while the arm ISN'T already
        # confirmed RAISED. This is what actually stops repeat-firing from
        # jitter during a long hold: real footage showed a raised arm held
        # for over a second re-triggering VOLUME_UP roughly every 0.6s
        # purely from jitter, because each trigger's own self-clear (see
        # add_sample) let a fresh 0.8-shoulder-width range redevelop from
        # noise alone well within the hold. Gating is safe here because
        # the legitimate up-swipe's own trigger fires *before* RAISED
        # confirms (debounce lags the raw crossing by a few frames), so
        # cutting off evaluation once confirmed doesn't cost real
        # detections -- only the post-confirmation jitter that shouldn't
        # count anyway. Once the arm leaves RAISED, `_prune`'s own
        # age-based cleanup discards anything more than WINDOW_SECONDS old
        # by the time evaluation resumes -- no separate reset needed.
        #
        # left_volume_swipe deliberately does NOT get the mirror-image
        # gate (evaluate only while confirmed RAISED): tried it, and it
        # broke real swipe-down detection -- by the time RAISED actually
        # confirms (again, lagging a few frames), the arm has often
        # already started descending below the raised threshold, so
        # gating on "still confirmed raised" cuts off evaluation right as
        # the descent itself needs to be measured. There's also no real
        # evidence this gate is needed for "down": VOLUME_DOWN never fired
        # once across every real-footage test run, gated or not. Fixing
        # a problem that was never observed, at the cost of breaking one
        # that was working, isn't a trade worth making -- see
        # scripts/qnn_spike.md for the full trace of both findings.
        if normalized.right.wrist_relative is not None:
            rel_x, rel_y = normalized.right.wrist_relative
            if state.right_sweep.add_sample(rel_x, rel_y):
                actions.append(ControlAction.SKIP)
            if state.right.confirmed != ArmPose.RAISED and state.right_volume_swipe.add_sample(rel_x, rel_y):
                actions.append(ControlAction.VOLUME_UP)
        if normalized.left.wrist_relative is not None:
            rel_x, rel_y = normalized.left.wrist_relative
            if state.left_sweep.add_sample(rel_x, rel_y):
                actions.append(ControlAction.SKIP)
            if state.left_volume_swipe.add_sample(rel_x, rel_y):
                actions.append(ControlAction.VOLUME_DOWN)

        right_changed = state.right.update(classify_arm(normalized.right))
        left_changed = state.left.update(classify_arm(normalized.left))

        if right_changed and state.right.confirmed == ArmPose.OUT_TO_SIDE:
            actions.append(ControlAction.NEXT)
        if left_changed and state.left.confirmed == ArmPose.OUT_TO_SIDE:
            actions.append(ControlAction.PREVIOUS)
        if (
            (left_changed or right_changed)
            and state.left.confirmed == ArmPose.RAISED
            and state.right.confirmed == ArmPose.RAISED
        ):
            actions.append(ControlAction.PLAY_PAUSE)

        return actions

    def get_arm_states(self, track_id: int) -> tuple[ArmPose, ArmPose] | None:
        """Current confirmed (left, right) arm states for `track_id`, or
        None if this track has no gesture state yet. For overlay/causality
        display -- lets a viewer see *why* an action is about to fire, not
        just that one did."""
        state = self._tracks.get(track_id)
        return (state.left.confirmed, state.right.confirmed) if state else None

    def _prune_stale_tracks(self) -> None:
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if self._frame_index - state.last_seen_frame <= STALE_TRACK_FRAMES
        }
