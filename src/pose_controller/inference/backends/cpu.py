from __future__ import annotations

import numpy as np

from pose_controller.inference.pose import Keypoint, PoseEstimator, PoseResult


class MediaPipePoseEstimator(PoseEstimator):
    """Dev-loop backend: runs Google MediaPipe Pose on CPU.

    Single-person only -- `mp.solutions.pose` (the legacy "solutions" API)
    doesn't support multi-person detection; Google's newer Tasks API
    (`mediapipe.tasks.python.vision.PoseLandmarker` with `num_poses`) does,
    but migrating this dev-only backend to it is deferred (see
    docs/backlog.md) since the real multi-person target is the QNN backend
    (which is genuinely multi-person via NMS over its detector's anchors,
    see `_blazepose.py`). `estimate` still returns a list for interface
    consistency -- just one of length 0 or 1 here.
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        import mediapipe as mp

        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame_bgr: np.ndarray) -> list[PoseResult]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mp_pose.process(rgb)
        if not result.pose_landmarks:
            return []

        keypoints = {
            idx: Keypoint(x=lm.x, y=lm.y, visibility=lm.visibility)
            for idx, lm in enumerate(result.pose_landmarks.landmark)
        }
        return [PoseResult(keypoints=keypoints)]

    def close(self) -> None:
        self._mp_pose.close()
