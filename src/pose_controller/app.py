from __future__ import annotations

import argparse
import sys

import cv2

from pose_controller.capture import Camera
from pose_controller.config import AppConfig
from pose_controller.inference.backends import build_pose_estimator
from pose_controller.overlay import draw_pose


def run(config: AppConfig) -> None:
    camera = Camera(config.capture)
    pose_estimator = build_pose_estimator(config.inference)

    try:
        while True:
            frame = camera.read()
            pose = pose_estimator.estimate(frame)
            draw_pose(frame, pose, label="person" if pose else None)

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
