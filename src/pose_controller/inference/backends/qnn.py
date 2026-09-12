from __future__ import annotations

from pathlib import Path

import numpy as np

from pose_controller.inference.pose import Keypoint, PoseEstimator, PoseResult
from pose_controller.inference.backends import _blazepose, _yolo_detect
from pose_controller.inference.backends._qnn_runtime import OnnxQnnRunner, QuantSpec

# Quantization scale/offset for each model's I/O tensors. Both models take
# the same standard [0,1] float -> uint8 image scaling as input.
_IMAGE_INPUT_QUANT = QuantSpec(scale=0.003921568859368563, offset=0)

# Read once via `qnn-context-binary-utility --context_binary=model.bin
# --json_file=...` against the compiled model in
# models/mediapipe_pose_qcs6490/ (see scripts/qnn_spike.md) -- that model's
# I/O has no quantization metadata in the ONNX graph itself (it's one
# opaque EPContext node), so these values had to come from the compiled
# binary directly.
_LANDMARK_OUTPUT_QUANT = {
    "scores": QuantSpec(scale=0.00390625, offset=0),
    "landmarks": QuantSpec(scale=0.006143786944448948, offset=-112),
}

# Unlike the mediapipe models above, this export's ONNX graph has explicit
# QuantizeLinear/DequantizeLinear nodes wrapping its EPContext node, so
# these came straight from the graph's own initializers (`onnx.load` +
# reading each output QuantizeLinear node's scale/zero_point) rather than
# qnn-context-binary-utility -- see models/yolov8n_det_qcs6490/README.md.
# QuantSpec's offset is the negated ONNX zero_point (real = (raw+offset)*
# scale here, vs ONNX's (raw-zero_point)*scale).
#
# class_idx's graph-declared scale is literally 0.0 -- meaningless taken
# literally (every value would dequantize to exactly 0 regardless of the
# raw byte), and not a plausible real quantization scale for an 80-class
# index. Treated as an exporter formality for an integer-valued output
# smuggled through a uint8 QDQ wrapper: scale=1/offset=0 (pure passthrough)
# is used instead, confirmed correct on-device (a person in frame reliably
# decodes to class_idx == PERSON_CLASS_ID == 0, not a garbage float).
_DETECTOR_OUTPUT_QUANT = {
    "boxes": QuantSpec(scale=3.1010673, offset=-25),
    "scores": QuantSpec(scale=0.00390625, offset=0),
    "class_idx": QuantSpec(scale=1.0, offset=0),
}


class QnnPoseEstimator(PoseEstimator):
    """On-device backend: runs a YOLOv8n person detector + the
    mediapipe_pose landmark model on the Q6A's Hexagon NPU via
    onnxruntime's QNN execution provider (see _qnn_runtime.OnnxQnnRunner).

    The detector stage is YOLOv8n (_yolo_detect.py), not BlazePose's own
    bundled 128x128 detector: that detector was found to lose small/
    distant subjects at ordinary room camera-to-subject distance
    regardless of image quality -- a person who reads clearly in a
    full-resolution photo becomes too small a silhouette once the whole
    frame is squeezed down to 128x128. YOLOv8n's 640x640 input preserves
    far more detail before any localization step. See
    scripts/qnn_spike.md for the full investigation. BlazePose's landmark
    model is unchanged -- only how its input crop is located changed, via
    _blazepose.detect_poses_from_boxes (box-derived ROI, no rotation cue)
    instead of _blazepose.detect_poses (BlazePose's own detector +
    aux-keypoint-derived ROI).

    `model_dir` must contain `pose_landmark_detector/model.onnx` (+
    `model.bin`) -- see models/mediapipe_pose_qcs6490/.
    `detector_model_dir` must contain `model.onnx` (+ `model.bin`) -- see
    models/yolov8n_det_qcs6490/.
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
            onnx_path=str(model_path / "pose_landmark_detector" / "model.onnx"),
            input_quant=_IMAGE_INPUT_QUANT,
            output_quant=_LANDMARK_OUTPUT_QUANT,
        )

    def _run_detector(self, input_nhwc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self._detector.run(input_nhwc)
        boxes = out["boxes"].reshape(-1, 4)
        scores = out["scores"].reshape(-1)
        classes = out["class_idx"].reshape(-1).astype(np.int64)
        return boxes, scores, classes

    def _run_landmark_detector(self, input_nhwc: np.ndarray) -> tuple[float, np.ndarray]:
        out = self._landmark_detector.run(input_nhwc)
        score = float(out["scores"].reshape(-1)[0])
        landmarks = out["landmarks"].reshape(_blazepose.NUM_VALID_LANDMARKS, 4)
        return score, landmarks

    def estimate(self, frame_bgr: np.ndarray) -> list[PoseResult]:
        person_boxes = _yolo_detect.detect_persons(frame_bgr, self._run_detector)
        all_landmarks = _blazepose.detect_poses_from_boxes(
            frame_bgr, person_boxes, self._run_landmark_detector
        )

        h, w = frame_bgr.shape[:2]
        results = []
        for landmarks in all_landmarks:
            keypoints = {
                idx: Keypoint(x=lm[0] / w, y=lm[1] / h, visibility=float(lm[3]))
                for idx, lm in enumerate(landmarks)
            }
            results.append(PoseResult(keypoints=keypoints))
        return results
