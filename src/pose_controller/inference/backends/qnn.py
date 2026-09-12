from __future__ import annotations

from pathlib import Path

import numpy as np

from pose_controller.inference.pose import Keypoint, PoseEstimator, PoseResult
from pose_controller.inference.backends import _blazepose
from pose_controller.inference.backends._qnn_runtime import OnnxQnnRunner, QuantSpec

# Quantization scale/offset for each model's I/O tensors, read once via
# `qnn-context-binary-utility --context_binary=model.bin --json_file=...`
# against the compiled models in models/mediapipe_pose_qcs6490/ (see
# scripts/qnn_spike.md). Both models share the same input quantization
# (standard [0,1] float -> uint8 image scaling).
_IMAGE_INPUT_QUANT = QuantSpec(scale=0.003921568859368563, offset=0)

_DETECTOR_OUTPUT_QUANT = {
    "box_scores_1": QuantSpec(scale=5.552783966064453, offset=-255),
    "box_coords_1": QuantSpec(scale=0.7927474975585938, offset=-89),
    "box_scores_2": QuantSpec(scale=4.8334197998046875, offset=-254),
    "box_coords_2": QuantSpec(scale=1.2209054231643677, offset=-99),
}
_LANDMARK_OUTPUT_QUANT = {
    "scores": QuantSpec(scale=0.00390625, offset=0),
    "landmarks": QuantSpec(scale=0.006143786944448948, offset=-112),
}


class QnnPoseEstimator(PoseEstimator):
    """On-device backend: runs the mediapipe_pose detector + landmark
    models on the Q6A's Hexagon NPU via onnxruntime's QNN execution
    provider (see _qnn_runtime.OnnxQnnRunner), using the pure numpy/opencv
    BlazePose pipeline in `_blazepose.py`.

    `model_dir` must contain `pose_detector/model.onnx` (+ `model.bin`),
    `pose_landmark_detector/model.onnx` (+ `model.bin`), and
    `anchors_pose.npy` -- see models/mediapipe_pose_qcs6490/ and
    scripts/qnn_spike.md for how these were produced and validated.
    """

    def __init__(self, model_dir: str):
        model_path = Path(model_dir)
        self._anchors = np.load(model_path / "anchors_pose.npy").astype(np.float32).reshape(-1, 2, 2)

        self._detector = OnnxQnnRunner(
            onnx_path=str(model_path / "pose_detector" / "model.onnx"),
            input_quant=_IMAGE_INPUT_QUANT,
            output_quant=_DETECTOR_OUTPUT_QUANT,
        )
        self._landmark_detector = OnnxQnnRunner(
            onnx_path=str(model_path / "pose_landmark_detector" / "model.onnx"),
            input_quant=_IMAGE_INPUT_QUANT,
            output_quant=_LANDMARK_OUTPUT_QUANT,
        )

    def _run_detector(self, input_nhwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = self._detector.run(input_nhwc)
        box_coords = np.concatenate(
            [out["box_coords_1"].reshape(-1, 12), out["box_coords_2"].reshape(-1, 12)]
        )
        box_scores = np.concatenate([out["box_scores_1"].reshape(-1), out["box_scores_2"].reshape(-1)])
        return box_coords, box_scores

    def _run_landmark_detector(self, input_nhwc: np.ndarray) -> tuple[float, np.ndarray]:
        out = self._landmark_detector.run(input_nhwc)
        score = float(out["scores"].reshape(-1)[0])
        landmarks = out["landmarks"].reshape(_blazepose.NUM_VALID_LANDMARKS, 4)
        return score, landmarks

    def estimate(self, frame_bgr: np.ndarray) -> PoseResult | None:
        landmarks = _blazepose.detect_pose(
            frame_bgr, self._run_detector, self._run_landmark_detector, self._anchors
        )
        if landmarks is None:
            return None

        h, w = frame_bgr.shape[:2]
        keypoints = {
            idx: Keypoint(x=lm[0] / w, y=lm[1] / h, visibility=float(lm[3]))
            for idx, lm in enumerate(landmarks)
        }
        return PoseResult(keypoints=keypoints)
