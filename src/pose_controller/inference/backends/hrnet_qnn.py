from __future__ import annotations

from pathlib import Path

import numpy as np

from pose_controller.inference.pose import PoseEstimator, PoseResult
from pose_controller.inference.backends import _hrnet_pose, _yolo_detect
from pose_controller.inference.backends._qnn_runtime import OnnxQnnRunner, QuantSpec
from pose_controller.inference.backends.qnn import _DETECTOR_OUTPUT_QUANT, _IMAGE_INPUT_QUANT

# Confirmed directly from the compiled model's own input_spec/output_spec
# (qai_hub.get_job(job_id).get_target_model()) -- unlike qnn.py's
# mediapipe model, this export's compile-job metadata gave these values
# straight up, no qnn-context-binary-utility digging needed. Note the
# input scale differs very slightly from _IMAGE_INPUT_QUANT (which is
# exactly 1/255) -- this model's own calibration produced a marginally
# different value, so it gets its own constant rather than reusing that
# one.
_HRNET_INPUT_QUANT = QuantSpec(scale=0.003917243331670761, offset=0)
_HRNET_OUTPUT_QUANT = {
    "heatmaps": QuantSpec(scale=0.003737130668014288, offset=-10.0),
}


class HrnetPoseEstimator(PoseEstimator):
    """On-device backend: runs the same YOLOv8n-det first stage as
    `QnnPoseEstimator` (`qnn.py`), but HRNetPose (`w8a8`, heatmap-decoded)
    as the per-person landmark stage instead of BlazePose -- see
    `_hrnet_pose.py`'s module docstring for why. An alongside evaluation,
    not a replacement for `qnn` or `yolo26` -- see `docs/backlog.md`.

    `model_dir` must contain `model.onnx` (+ `model.bin`) -- see
    `models/hrnet_pose_qcs6490/`. `detector_model_dir` is the same
    YOLOv8n-det model `QnnPoseEstimator` uses -- see
    `models/yolov8n_det_qcs6490/`.
    """

    def __init__(self, model_dir: str, detector_model_dir: str):
        model_path = Path(model_dir)
        detector_path = Path(detector_model_dir)

        self._detector = OnnxQnnRunner(
            onnx_path=str(detector_path / "model.onnx"),
            input_quant=_IMAGE_INPUT_QUANT,
            output_quant=_DETECTOR_OUTPUT_QUANT,
        )
        self._landmark_detector = OnnxQnnRunner(
            onnx_path=str(model_path / "model.onnx"),
            input_quant=_HRNET_INPUT_QUANT,
            output_quant=_HRNET_OUTPUT_QUANT,
        )

    def _run_detector(self, input_nhwc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self._detector.run(input_nhwc)
        boxes = out["boxes"].reshape(-1, 4)
        scores = out["scores"].reshape(-1)
        classes = out["class_idx"].reshape(-1).astype(np.int64)
        return boxes, scores, classes

    def _run_landmark_detector(self, input_nhwc: np.ndarray) -> np.ndarray:
        out = self._landmark_detector.run(input_nhwc)
        h, w = _hrnet_pose.HEATMAP_SIZE
        return out["heatmaps"].reshape(h, w, 17)

    def estimate(self, frame_bgr: np.ndarray) -> list[PoseResult]:
        person_boxes = _yolo_detect.detect_persons(frame_bgr, self._run_detector)
        return _hrnet_pose.detect_poses_from_boxes(frame_bgr, person_boxes, self._run_landmark_detector)
