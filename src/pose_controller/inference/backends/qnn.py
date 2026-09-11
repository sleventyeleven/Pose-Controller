from __future__ import annotations

import numpy as np

from pose_controller.inference.pose import PoseEstimator, PoseResult


class QnnPoseEstimator(PoseEstimator):
    """On-device backend: runs a pose model on the Q6A's Hexagon NPU via
    Qualcomm AI Hub Models / QNN.

    NOT YET IMPLEMENTED -- this is milestone 1's NPU spike. See
    docs/hardware.md and scripts/qnn_spike.md for the plan to validate the
    QNN toolchain on the Q6A before filling this in. Left as an explicit
    stub (rather than a fallback to CPU) so a config pointed at "qnn" fails
    loudly instead of silently running the wrong backend.
    """

    def __init__(self, model_dir: str):
        raise NotImplementedError(
            "QNN backend is pending the on-device NPU spike -- see scripts/qnn_spike.md"
        )

    def estimate(self, frame_bgr: np.ndarray) -> PoseResult | None:
        raise NotImplementedError
