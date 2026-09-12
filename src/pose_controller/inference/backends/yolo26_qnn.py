from __future__ import annotations

from pathlib import Path

import numpy as np

from pose_controller.inference.pose import PoseEstimator, PoseResult
from pose_controller.inference.backends import _yolo26_pose
from pose_controller.inference.backends._qnn_runtime import create_qnn_session


class Yolo26PoseEstimator(PoseEstimator):
    """On-device backend: YOLO26-Pose, a single end-to-end person-
    detection + 17-keypoint model, run on the Q6A/Q8B Hexagon NPU via
    onnxruntime's QNN execution provider -- an *alternate* pipeline to
    `qnn.QnnPoseEstimator` (YOLOv8n-det + BlazePose-landmark), built to
    A/B against it rather than replace it. See `_yolo26_pose.py`'s module
    docstring and `docs/journey.md` for why this architecture is being
    evaluated, and why the model comes from Ultralytics' own local QNN
    export rather than Qualcomm AI Hub (which failed to compile it).

    Unlike `qnn.QnnPoseEstimator`/`hrnet_qnn.HrnetPoseEstimator` (AI-Hub-
    compiled models: opaque uint8 I/O needing manual `QuantSpec` math),
    this model's own graph carries explicit `QuantizeLinear`/
    `DequantizeLinear` nodes wrapping its `EPContext` node, so plain
    float32 in, float32 out -- `create_qnn_session` is used directly with
    no separate quantize/dequantize step needed.

    `model_dir` must contain a single self-contained `model.onnx` (the
    QNN context binary is embedded inline, not a separate `.bin` file) --
    see `models/yolo26n_pose_qcs6490/`.
    """

    def __init__(self, model_dir: str):
        model_path = Path(model_dir)
        self._session = create_qnn_session(str(model_path / "model.onnx"))
        self._input_name = self._session.get_inputs()[0].name

    def _run_detector(self, input_nhwc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        (raw_output,) = self._session.run(None, {self._input_name: input_nhwc})
        return _yolo26_pose.decode_raw_output(raw_output[0])

    def estimate(self, frame_bgr: np.ndarray) -> list[PoseResult]:
        return _yolo26_pose.detect_poses(frame_bgr, self._run_detector)
