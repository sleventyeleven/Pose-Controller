from __future__ import annotations

import argparse
import sys

import cv2

from pose_controller.capture import Camera
from pose_controller.config import AppConfig, TrackingConfig
from pose_controller.inference.backends import build_pose_estimator
from pose_controller.overlay import draw_poses
from pose_controller.tracking import ReidEmbedder, Tracker


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

    try:
        while True:
            frame = camera.read()
            poses = pose_estimator.estimate(frame)
            tracked_poses = tracker.update(poses, frame_bgr=frame)
            draw_poses(frame, tracked_poses)

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
