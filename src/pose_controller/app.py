from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import cv2

from pose_controller.capture import Camera, assess_exposure
from pose_controller.config import AppConfig, InferenceConfig, OverlayConfig, TrackingConfig
from pose_controller.control import build_media_controller
from pose_controller.gestures import ArmPose, GestureStateMachine
from pose_controller.inference.backends import build_pose_estimator
from pose_controller.inference.pose import PoseResult
from pose_controller.overlay import draw_action_banner, draw_detections, draw_poses
from pose_controller.tracking import ReidEmbedder, Tracker
from pose_controller.web import DashboardServer, DashboardState

# How long a triggered action stays shown in the overlay banner, in
# frames -- not real time, since fps varies by backend/hardware. ~1.5s at
# a typical 30fps dev-loop rate.
ACTION_BANNER_FRAMES = 45


@dataclass
class _HeldOverlay:
    """A track's last-known pose/arm-states, kept around purely for
    drawing on frames where that track has no fresh detection. Real
    footage regularly misses detection on a large fraction of frames
    (scripts/qnn_spike.md) -- `Tracker.update()` only ever returns *this
    frame's* real detections (by design: it never synthesizes a result
    for a coasting track, since `gestures.state_machine` depends on
    knowing "no real observation this frame" as distinct from a
    confirmed DOWN classification -- see `_ArmDebouncer`). Drawing
    nothing at all on a miss produces a distracting pop-in/pop-out
    flicker even though the person hasn't moved or left, so the overlay
    (only the overlay -- gesture recognition is untouched) holds the last
    real frame's pose/arm-states and keeps drawing them until either a
    fresh detection replaces them or `max_age` frames pass with no real
    detection, at which point the tracker itself would no longer
    consider this the same physical track anyway."""

    pose: PoseResult
    arm_states: tuple[ArmPose, ArmPose] | None
    frames_stale: int = 0


# IoU above which two displayed poses are treated as "the same physical
# spot" and deduplicated to just one -- deliberately looser (requires
# more overlap) than Tracker's own iou_threshold (0.3, for identity
# matching), since this is a purely cosmetic display decision: the goal
# is only to stop multiple ghosted overlays from stacking on top of each
# other when identity tracking itself is fragmenting (e.g. a poorly-
# framed camera producing inconsistent partial-body crops -- see
# docs/backlog.md), not to merge two genuinely different nearby people.
OVERLAY_DEDUP_IOU = 0.5


def _bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def update_held_overlay(
    held_overlay: dict[int, _HeldOverlay],
    tracked_poses: list[PoseResult],
    arm_states: dict[int, tuple[ArmPose, ArmPose]],
    max_age: int,
) -> list[PoseResult]:
    """Merge this frame's real detections into `held_overlay` (mutated in
    place) and return the poses to actually draw -- this frame's real
    detections plus any track that has no detection this frame but is
    still within `max_age` frames of its last one. Also fills in
    `arm_states` (mutated in place) for held tracks from their last-known
    state, so the overlay's arm-state labels don't flicker either. See
    `_HeldOverlay` for why this exists.

    Before returning, poses that heavily overlap another already-kept one
    (see `OVERLAY_DEDUP_IOU`) are dropped in favor of the fresher/less-
    stale one -- without this, a fragmenting tracker (frequent new
    track_ids for what's really the same continuously-present person, one
    real cause: an inconsistent partial-body crop from bad camera framing
    feeding both IoU matching and re-ID appearance embeddings poor input)
    would pile up several held ghosts directly on top of each other,
    which looks far messier than the flicker this whole mechanism exists
    to prevent."""
    seen_this_frame = set()
    candidates: list[tuple[PoseResult, int]] = []  # (pose, frames_stale) -- 0 for this frame's real detections

    for pose in tracked_poses:
        if pose.track_id is not None:
            seen_this_frame.add(pose.track_id)
            held_overlay[pose.track_id] = _HeldOverlay(pose=pose, arm_states=arm_states.get(pose.track_id))
        candidates.append((pose, 0))

    for track_id, held in list(held_overlay.items()):
        if track_id in seen_this_frame:
            continue
        held.frames_stale += 1
        if held.frames_stale > max_age:
            del held_overlay[track_id]
            continue
        candidates.append((held.pose, held.frames_stale))
        if held.arm_states is not None:
            arm_states[track_id] = held.arm_states

    candidates.sort(key=lambda c: c[1])  # freshest (lowest frames_stale) first

    kept: list[PoseResult] = []
    kept_boxes: list[tuple[float, float, float, float]] = []
    for pose, _stale in candidates:
        if pose.bbox is not None and any(
            _bbox_iou(pose.bbox, kept_box) >= OVERLAY_DEDUP_IOU for kept_box in kept_boxes
        ):
            continue  # a fresher/less-stale pose already occupies this spot
        kept.append(pose)
        if pose.bbox is not None:
            kept_boxes.append(pose.bbox)

    return kept


def build_tracker(config: TrackingConfig) -> Tracker:
    embedder = None
    if config.reid_enabled:
        try:
            embedder = ReidEmbedder(config.reid_model_path)
        except Exception as exc:  # noqa: BLE001 -- report and fall back, don't crash the app
            print(f"Re-ID disabled: could not load {config.reid_model_path!r} ({exc})")

    return Tracker(
        iou_threshold=config.iou_threshold,
        max_age=config.max_age,
        min_hits=config.min_hits,
        embedder=embedder,
        reid_similarity_threshold=config.reid_similarity_threshold,
        reid_gallery_ttl=config.reid_gallery_ttl,
    )


def build_dashboard(config: OverlayConfig) -> tuple[DashboardState, DashboardServer] | tuple[None, None]:
    if not config.web_enabled:
        return None, None
    state = DashboardState()
    server = DashboardServer(state, port=config.web_port)
    server.start()
    print(f"[web] dashboard running at http://localhost:{server.port}/")
    return state, server


def build_detect_overlay(overlay_config: OverlayConfig, inference_config: InferenceConfig):
    """Optional general-object-detection overlay (YOLO26-Detection, all
    80 COCO classes) drawn alongside the pose skeleton -- purely a
    visualization/diagnostic layer, not part of the pose/gesture
    pipeline. See `inference/backends/_yolo26_detect.py`'s module
    docstring and `docs/backlog.md`. Imported lazily so the CPU dev-loop
    backend (no `onnxruntime-qnn` installed) never needs this module at
    all unless explicitly enabled."""
    if not overlay_config.detect_overlay_enabled:
        return None
    from pose_controller.inference.backends.yolo26_detect_qnn import Yolo26Detector

    return Yolo26Detector(model_dir=inference_config.yolo26_det_model_dir)


def run(config: AppConfig) -> None:
    camera = Camera(config.capture)
    pose_estimator = build_pose_estimator(config.inference)
    tracker = build_tracker(config.tracking)
    gestures = GestureStateMachine()
    dashboard_state, dashboard_server = build_dashboard(config.overlay)
    detect_overlay = build_detect_overlay(config.overlay, config.inference)
    media_controller = build_media_controller(config.control)

    banner_text: str | None = None
    banner_frames_remaining = 0
    last_exposure_issues: list[str] = []
    held_overlay: dict[int, _HeldOverlay] = {}

    try:
        while True:
            frame = camera.read()

            exposure = assess_exposure(frame)
            if exposure.issues != last_exposure_issues:
                if exposure.issues:
                    print(f"[exposure] lighting issue detected: {', '.join(exposure.issues)} "
                          f"(brightness={exposure.mean_brightness:.0f}, saturation={exposure.mean_saturation:.0f}) "
                          "-- pose detection may be unreliable, see docs/backlog.md")
                else:
                    print("[exposure] lighting OK")
                last_exposure_issues = exposure.issues

            poses = pose_estimator.estimate(frame)
            tracked_poses = tracker.update(poses, frame_bgr=frame)
            triggered = gestures.update_all(tracked_poses)

            for track_id, action in triggered:
                print(f"[gesture] #{track_id}: {action.name}")
                banner_text = f"#{track_id}: {action.name}"
                banner_frames_remaining = ACTION_BANNER_FRAMES
                media_controller.dispatch(action)
                if dashboard_state is not None:
                    dashboard_state.add_event(track_id, action.name)

            arm_states = {}
            for pose in tracked_poses:
                if pose.track_id is None:
                    continue
                state = gestures.get_arm_states(pose.track_id)
                if state is not None:
                    arm_states[pose.track_id] = state

            display_poses = update_held_overlay(held_overlay, tracked_poses, arm_states, config.tracking.max_age)
            draw_poses(frame, display_poses, arm_states=arm_states)

            if detect_overlay is not None:
                draw_detections(frame, detect_overlay.detect(frame))

            if exposure.issues:
                h = frame.shape[0]
                cv2.rectangle(frame, (0, h - 30), (frame.shape[1], h), (0, 0, 150), -1)
                cv2.putText(
                    frame, f"Lighting: {', '.join(exposure.issues)}", (10, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
                )

            if banner_frames_remaining > 0:
                draw_action_banner(frame, banner_text)
                banner_frames_remaining -= 1
                if banner_frames_remaining == 0:
                    banner_text = None

            if dashboard_state is not None:
                dashboard_state.update_frame(frame)
                dashboard_state.update_media_status(media_controller.get_status())

            if config.overlay.show_window:
                cv2.imshow(config.overlay.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        pose_estimator.close()
        if detect_overlay is not None:
            detect_overlay.close()
        media_controller.close()
        cv2.destroyAllWindows()
        if dashboard_server is not None:
            dashboard_server.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pose-Controller: pose/position-triggered media control")
    parser.add_argument(
        "--config", default="configs/dev_laptop.yaml",
        help="Path to a YAML config (default: configs/dev_laptop.yaml)",
    )
    args = parser.parse_args(argv)

    config = AppConfig.from_yaml(args.config)
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
