from __future__ import annotations

import numpy as np

from pose_controller.inference.pose import Keypoint, PoseEstimator, PoseResult


class MediaPipePoseEstimator(PoseEstimator):
    """Dev-loop backend: runs Google MediaPipe Pose on CPU.

    Single-person only (matches MediaPipe Pose's own limitation) -- this is
    intentional for milestone 2 (capture + single-pose baseline). Multi-person
    support arrives in milestone 3 via a detector + per-crop pose model, at
    which point this class is reused per-crop.
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        import mediapipe as mp

        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame_bgr: np.ndarray) -> PoseResult | None:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mp_pose.process(rgb)
        if not result.pose_landmarks:
            return None

        keypoints = {
            idx: Keypoint(x=lm.x, y=lm.y, visibility=lm.visibility)
            for idx, lm in enumerate(result.pose_landmarks.landmark)
        }
        return PoseResult(keypoints=keypoints)

    def close(self) -> None:
        self._mp_pose.close()
