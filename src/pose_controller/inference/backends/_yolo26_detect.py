"""Pure numpy/opencv pre/post-processing for YOLO26-Detection: a general
80-class (COCO) object detector, staged for dashboard visualization and
future scene-context reasoning -- not part of the pose/gesture pipeline.

Unlike `_yolo_detect.py` (YOLOv8n-det, filtered to person-only, feeding
BlazePose's landmark model as the pose pipeline's first stage), this
module deliberately returns *every* detected class, not just people. Two
motivating uses, per the request this was built for:

1. Immediate: draw all detected objects (with class labels) on the web
   dashboard's overlay feed, as an independent visualization layer
   alongside the pose skeleton -- useful for seeing what the camera
   actually sees as "objects" versus what the pose pipeline is tracking
   as "people", which is a real, separate diagnostic signal.
2. Future (not built yet): in a cluttered or distorting scene -- the
   original brief's eventual swim-safety-monitoring target explicitly
   named a pool as an example -- knowing *what else* is in frame (pool
   equipment, reflections mis-detected as something else, etc.) alongside
   where people are could help explain or filter out false pose/track
   detections, rather than only ever seeing a person-or-nothing signal.
   This module only provides the raw detections; that reasoning doesn't
   exist yet.

**Model source, corrected 2026-09-12**: originally attempted via
Qualcomm AI Hub (`qai_hub_models.models.yolo26_det`), which failed with
the same reproducible QNN compile error as YOLO26-Pose (see
`docs/backlog.md`). The actual deployed model comes from **Ultralytics'
own local QNN export**, targeting QCS6490's real `soc_model` value
(`93`) rather than the generic `htp_arch=68` -- see
`models/yolo26_det_qcs6490/README.md` for why (the generic target
loaded on the Q8B but failed on the Q6A). Confirmed running on real
Hexagon NPU hardware on both boards. Unlike an
`include_postprocessing=True` AI Hub export, this leaves class
scores as 80 independent per-anchor sigmoid values (not yet argmax'd)
and boxes undecoded to xyxy (still cxcywh) -- `decode_raw_output` below
does that missing step before handing off to `detect_objects`, which
still does score filtering + NMS exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from pose_controller.inference.backends._blazepose import _cxcywh_to_xyxy, _greedy_nms, resize_pad

DETECTOR_INPUT_SIZE = (640, 640)  # H, W
NMS_SCORE_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.7

# COCO's 80 class names, in class-index order -- from Ultralytics'
# cfg/datasets/coco.yaml (the same source this model's own training/
# export pipeline uses), hardcoded here rather than importing
# `ultralytics` at runtime: that package is a dev-machine-only export-time
# dependency for this project (see docs/storage-footprint.md), not
# something that should need installing on the Q6A just to label a box.
COCO_CLASS_NAMES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    """One detected object, in original-frame pixel space."""

    box_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int

    @property
    def class_name(self) -> str:
        if 0 <= self.class_id < len(COCO_CLASS_NAMES):
            return COCO_CLASS_NAMES[self.class_id]
        return f"class_{self.class_id}"


def decode_raw_output(raw_output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode Ultralytics' raw YOLO26-Detection export output into the
    `(boxes[N,4] xyxy, scores[N], class_idx[N])` contract `detect_objects`
    expects.

    `raw_output`: `[84, num_anchors]` (already squeezed of its batch dim)
    -- channels `0:4` are the box in (cx, cy, w, h) pixel space (640x640
    input, already stride/anchor-decoded by the graph itself), channels
    `4:84` are 80 independent per-class sigmoid scores (not yet argmax'd
    -- confirmed from `ultralytics.nn.modules.head.Detect._inference`
    source). Still per-anchor (not yet score-filtered or NMS'd) --
    `detect_objects` does that next, unchanged from the AI-Hub-output code
    path."""
    raw = raw_output.T  # [num_anchors, 84]
    boxes_xyxy = _cxcywh_to_xyxy(raw[:, :4])
    class_scores = raw[:, 4:]
    scores = class_scores.max(axis=1)
    class_idx = class_scores.argmax(axis=1)
    return boxes_xyxy, scores, class_idx


def detect_objects(frame_bgr_uint8: np.ndarray, detector_infer: callable) -> list[Detection]:
    """Run YOLO26-Detection on one BGR frame.

    `detector_infer(input_nhwc_float32) -> (boxes[N,4] xyxy, scores[N],
    class_idx[N])` is a backend-provided callable (e.g. wrapping
    OnnxQnnRunner), in the padded/scaled 640x640 input space -- already
    box-decoded and argmax'd over classes by the exported model itself
    (include_postprocessing at export time), but not yet score-filtered
    or NMS'd.

    Returns every surviving detection (any class, not just person) as a
    `Detection`, highest score first, in original-frame pixel space.
    """
    frame_rgb = cv2.cvtColor(frame_bgr_uint8, cv2.COLOR_BGR2RGB)
    det_input, scale, pad = resize_pad(frame_rgb, DETECTOR_INPUT_SIZE)
    det_input_nhwc = (det_input.astype(np.float32) / 255.0)[None]

    boxes, scores, classes = detector_infer(det_input_nhwc)

    mask = scores >= NMS_SCORE_THRESHOLD
    if not np.any(mask):
        return []
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    keep = _greedy_nms(boxes, scores, NMS_IOU_THRESHOLD)

    pad_x, pad_y = pad
    results = []
    for idx in keep:
        box = boxes[idx].astype(np.float32).copy()
        box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
        box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
        results.append(
            Detection(box_xyxy=tuple(box.tolist()), score=float(scores[idx]), class_id=int(classes[idx]))
        )
    return results
