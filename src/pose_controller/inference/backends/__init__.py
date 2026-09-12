from __future__ import annotations

from pose_controller.config import InferenceConfig
from pose_controller.inference.pose import PoseEstimator


def build_pose_estimator(config: InferenceConfig) -> PoseEstimator:
    if config.backend == "cpu":
        from .cpu import MediaPipePoseEstimator

        return MediaPipePoseEstimator()
    if config.backend == "qnn":
        from .qnn import QnnPoseEstimator

        return QnnPoseEstimator(
            model_dir=config.model_dir, detector_model_dir=config.detector_model_dir
        )
    raise ValueError(f"Unknown inference backend: {config.backend!r}")
