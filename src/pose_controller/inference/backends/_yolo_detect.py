"""Pure numpy/opencv pre/post-processing for the YOLOv8n person detector,
used as the first-stage person locator ahead of BlazePose's landmark model
(see qnn.py).

Replaces BlazePose's own bundled detector for this role: that detector's
fixed 128x128 input was found to lose small/distant subjects at ordinary
room camera-to-subject distance regardless of image quality (a person
that reads clearly in a full-resolution photo becomes too small a
silhouette once the whole frame is squeezed down to 128x128) -- see
scripts/qnn_spike.md for the full investigation. YOLOv8n's 640x640 input
preserves far more detail before any localization step, which is exactly
the axis the earlier failure needed fixed.

Ported from Qualcomm AI Hub Models' YoloV8Detector export/app pipeline
(qai_hub_models.models.yolov8_det.model / qai_hub_models.models.templates.
yolo.{model,app}, BSD-3-Clause): the model is exported with
`include_postprocessing=True` (the default), so it already does box
decoding and per-box argmax-over-classes internally -- this module only
has to do score/class filtering and NMS on top of that, not raw anchor
decoding. NMS thresholds (score=0.45, iou=0.7) and the resize/pad +
unscale coordinate math match that reference's defaults
(qai_hub_models.models.templates.yolo.app.YoloObjectDetectionApp) exactly,
cross-checked against that source rather than re-derived, to avoid
subtly-wrong box math the way _blazepose.py avoids it for BlazePose.
"""

from __future__ import annotations

import numpy as np
import cv2

from pose_controller.inference.backends._blazepose import _greedy_nms, resize_pad

DETECTOR_INPUT_SIZE = (640, 640)  # H, W
PERSON_CLASS_ID = 0  # COCO class index for "person"
NMS_SCORE_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.7


def detect_persons(frame_bgr_uint8: np.ndarray, detector_infer: callable) -> list[np.ndarray]:
    """Run the YOLOv8n person detector on one BGR frame.

    `detector_infer(input_nhwc_float32) -> (boxes[N,4] xyxy, scores[N],
    classes[N])` is a backend-provided callable (e.g. wrapping
    OnnxQnnRunner) returning the exported model's raw per-prediction
    output in the padded/scaled 640x640 input space -- already box-decoded
    and argmax'd over classes by the model itself (include_postprocessing
    export option), but not yet score-filtered or NMS'd.

    Returns person-class boxes as [4] xyxy arrays in the original frame's
    pixel space, highest score first, after score-thresholding + NMS.
    Empty list if no person clears the score threshold.
    """
    frame_rgb = cv2.cvtColor(frame_bgr_uint8, cv2.COLOR_BGR2RGB)
    det_input, scale, pad = resize_pad(frame_rgb, DETECTOR_INPUT_SIZE)
    det_input_nhwc = (det_input.astype(np.float32) / 255.0)[None]

    boxes, scores, classes = detector_infer(det_input_nhwc)

    person_mask = (classes == PERSON_CLASS_ID) & (scores >= NMS_SCORE_THRESHOLD)
    if not np.any(person_mask):
        return []
    boxes = boxes[person_mask]
    scores = scores[person_mask]

    keep = _greedy_nms(boxes, scores, NMS_IOU_THRESHOLD)

    pad_x, pad_y = pad
    results = []
    for idx in keep:
        box = boxes[idx].astype(np.float32).copy()
        box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
        box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
        results.append(box)
    return results
