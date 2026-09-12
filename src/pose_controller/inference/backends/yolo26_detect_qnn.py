from __future__ import annotations

from pathlib import Path

import numpy as np

from pose_controller.inference.backends import _yolo26_detect
from pose_controller.inference.backends._qnn_runtime import create_qnn_session


class Yolo26Detector:
    """On-device general object detector (YOLO26-Detection, all 80 COCO
    classes), staged for dashboard visualization and future scene-context
    reasoning -- not part of the pose/gesture pipeline. See
    `_yolo26_detect.py`'s module docstring for why this exists as its own
    thing rather than reusing `_yolo_detect.py` (which is person-only,
    feeding the pose pipeline's first stage), and why the model comes
    from Ultralytics' own local QNN export rather than Qualcomm AI Hub
    (which failed to compile it).

    Deliberately not a `PoseEstimator` -- returns `Detection` objects
    (box + class + score), not `PoseResult`s, since there's no pose
    involved here at all. Plain float32 I/O -- see `yolo26_qnn.py`'s
    docstring for why no manual `QuantSpec` is needed for this model
    source.

    `model_dir` must contain a single self-contained `model.onnx` -- see
    `models/yolo26_det_qcs6490/`.
    """

    def __init__(self, model_dir: str):
        model_path = Path(model_dir)
        self._session = create_qnn_session(str(model_path / "model.onnx"))
        self._input_name = self._session.get_inputs()[0].name

    def _run_detector(self, input_nhwc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        (raw_output,) = self._session.run(None, {self._input_name: input_nhwc})
        return _yolo26_detect.decode_raw_output(raw_output[0])

    def detect(self, frame_bgr: np.ndarray) -> list[_yolo26_detect.Detection]:
        return _yolo26_detect.detect_objects(frame_bgr, self._run_detector)

    def close(self) -> None:
        pass
