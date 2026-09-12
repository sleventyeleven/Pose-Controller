from __future__ import annotations

import argparse
import sys

import cv2

from pose_controller.capture import Camera, assess_exposure
from pose_controller.config import AppConfig, TrackingConfig
from pose_controller.gestures import GestureStateMachine
from pose_controller.inference.backends import build_pose_estimator
from pose_controller.overlay import draw_action_banner, draw_poses
from pose_controller.tracking import ReidEmbedder, Tracker

# How long a triggered action stays shown in the overlay banner, in
# frames -- not real time, since fps varies by backend/hardware. ~1.5s at
# a typical 30fps dev-loop rate.
ACTION_BANNER_FRAMES = 45


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


def run(config: AppConfig) -> None:
    camera = Camera(config.capture)
    pose_estimator = build_pose_estimator(config.inference)
    tracker = build_tracker(config.tracking)
    gestures = GestureStateMachine()

    banner_text: str | None = None
    banner_frames_remaining = 0
    last_exposure_issues: list[str] = []

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

            arm_states = {}
            for pose in tracked_poses:
                if pose.track_id is None:
                    continue
                state = gestures.get_arm_states(pose.track_id)
                if state is not None:
                    arm_states[pose.track_id] = state
            draw_poses(frame, tracked_poses, arm_states=arm_states)

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

            if config.overlay.show_window:
                cv2.imshow(config.overlay.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        pose_estimator.close()
        cv2.destroyAllWindows()


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
