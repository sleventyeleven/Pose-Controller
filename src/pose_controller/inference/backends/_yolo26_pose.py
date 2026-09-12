"""Pure numpy/opencv pre/post-processing for YOLO26-Pose: a single,
end-to-end person-detection + 17-keypoint model, built as an *alternate*
on-device pipeline to A/B against the current YOLOv8n-det + BlazePose-
landmark two-stage one (see `qnn.py`) -- not a replacement yet. See
`docs/journey.md` for why this is being evaluated: two real, already-
diagnosed weak points in the current pipeline that this architecture
sidesteps by construction --

1. The current pipeline's `_roi_corners_from_box` crop-margin step (a
   second model invocation on a manually-cropped square region) is a
   whole extra layer that's already needed real tuning and produced real
   bugs this project found the hard way (scripts/qnn_spike.md). YOLO26-
   Pose detects boxes *and* keypoints in one forward pass over the whole
   frame -- no separate crop step exists to get wrong.
2. BlazePose's landmark model gates its *entire* output behind one
   scalar confidence (`MIN_LANDMARK_SCORE`) -- a pose with a perfectly
   visible arm can be rejected wholesale if that one number dips, which
   is a real candidate for the "only works facing the camera" symptom
   found during live testing. YOLO26-Pose reports a detection confidence
   plus independent per-keypoint visibility instead, closer to how this
   project's own gesture code already wants to consume data (see
   `gestures/normalize.py`'s per-keypoint `_visible()` checks).

**Model source, corrected 2026-09-12**: originally attempted via
Qualcomm AI Hub (`qai_hub_models.models.yolo26_pose`), which failed with
a reproducible QNN context-binary compile error ("exit code 14") for
every attempt -- see `docs/backlog.md`. The actual deployed model instead
comes from **Ultralytics' own local QNN export**
(`YOLO("yolo26n-pose.pt").export(format="qnn", name="qcs6490")` with
`soc_model=93` registered for that name -- fully offline, no Qualcomm
account needed -- see https://docs.ultralytics.com/integrations/qnn and
`models/yolo26n_pose_qcs6490/README.md` for why `soc_model` and not the
generic `htp_arch` targeting), confirmed compiling and running on real
Hexagon v68 NPU hardware on both the Q6A and Q8B, where AI Hub's
toolchain could not compile it at all. This changes the I/O contract this module decodes: Ultralytics'
export does **not** bake in NMS or box/keypoint pixel-space decoding the
way AI Hub's `include_postprocessing=True` exports do -- it stops after
per-anchor box regression + sigmoid scores/visibility (confirmed from
`ultralytics.nn.modules.head.Detect._inference` /
`Pose26.kpts_decode` source, not assumed), leaving NMS as the caller's
job. `decode_raw_output` below does that missing step (cxcywh->xyxy +
channel splitting) before handing off to `detect_poses`, which still
does score filtering + NMS + the COCO->`Landmark` mapping exactly as
before -- that part of this module needed no changes at all.
"""

from __future__ import annotations

import numpy as np
import cv2

from pose_controller.inference.backends._blazepose import _cxcywh_to_xyxy, _greedy_nms, resize_pad
from pose_controller.inference.pose import Keypoint, Landmark, PoseResult

DETECTOR_INPUT_SIZE = (640, 640)  # H, W
# Matches this project's own yolov8n-det threshold (_yolo_detect.py) --
# the evaluator's own conf_threshold=0.001 default is for exhaustive mAP
# sweeps, not a usable single-inference operating point.
NMS_SCORE_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.7  # matches qai_hub_models' YoloPoseEvaluator default

# COCO's 17-keypoint order (confirmed from
# qai_hub_models.datasets.coco.coco_keypoints.COCO_SKELETON's ordering,
# not assumed) mapped onto this project's Landmark enum -- only the
# subset gestures/ actually uses. Landmark's own integer values are
# MediaPipe indices; PoseResult.keypoints is keyed by whatever int a
# backend chooses, as long as it's consistent, so reusing Landmark's
# values here (rather than COCO's own indices) is what makes this
# backend a drop-in for every downstream consumer (Tracker,
# GestureStateMachine, overlay) with zero changes to any of them.
_COCO_TO_LANDMARK: dict[int, Landmark] = {
    0: Landmark.NOSE,
    5: Landmark.LEFT_SHOULDER,
    6: Landmark.RIGHT_SHOULDER,
    7: Landmark.LEFT_ELBOW,
    8: Landmark.RIGHT_ELBOW,
    9: Landmark.LEFT_WRIST,
    10: Landmark.RIGHT_WRIST,
    11: Landmark.LEFT_HIP,
    12: Landmark.RIGHT_HIP,
}


def decode_raw_output(raw_output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode Ultralytics' raw YOLO26-Pose export output into the
    `(boxes[N,4] xyxy, scores[N], keypoints[N,17,3])` contract
    `detect_poses` expects.

    `raw_output`: `[56, num_anchors]` (already squeezed of its batch dim)
    -- channels `0:4` are the person box in (cx, cy, w, h) pixel space
    (640x640 input, already stride/anchor-decoded by the graph itself),
    channel `4` is person confidence (already sigmoid-activated), and
    channels `5:56` are 17 keypoints x (x, y, visibility) interleaved
    triples in COCO order, x/y already in the same 640x640 pixel space
    and visibility already sigmoid-activated -- confirmed from
    `ultralytics.nn.modules.head.Pose26.kpts_decode` source. Still
    per-anchor (not yet score-filtered or NMS'd) -- `detect_poses` does
    that next, unchanged from the AI-Hub-output code path."""
    raw = raw_output.T  # [num_anchors, 56]
    boxes_xyxy = _cxcywh_to_xyxy(raw[:, :4])
    scores = raw[:, 4]
    keypoints = raw[:, 5:].reshape(-1, 17, 3)
    return boxes_xyxy, scores, keypoints


def detect_poses(frame_bgr_uint8: np.ndarray, detector_infer: callable) -> list[PoseResult]:
    """Run YOLO26-Pose on one BGR frame.

    `detector_infer(input_nhwc_float32) -> (boxes[N,4] xyxy, scores[N],
    keypoints[N,17,3])` is a backend-provided callable (e.g. wrapping
    OnnxQnnRunner), all still in the padded/scaled 640x640 input space --
    already box-decoded, confidence-scored, and keypoint-reshaped by the
    exported model itself, but not yet filtered or NMS'd.

    Returns one `PoseResult` per surviving detection (after score
    thresholding + NMS), each carrying the COCO keypoints this project's
    gesture code needs (shoulders, elbows, wrists, hips, nose) re-keyed
    onto `Landmark`, in normalized [0,1] image-space coordinates -- the
    same contract every other `PoseEstimator` backend follows, so
    `Tracker`, `GestureStateMachine`, and `overlay/` all work unchanged.
    """
    h, w = frame_bgr_uint8.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr_uint8, cv2.COLOR_BGR2RGB)
    det_input, scale, pad = resize_pad(frame_rgb, DETECTOR_INPUT_SIZE)
    det_input_nhwc = (det_input.astype(np.float32) / 255.0)[None]

    boxes, scores, keypoints = detector_infer(det_input_nhwc)

    mask = scores >= NMS_SCORE_THRESHOLD
    if not np.any(mask):
        return []
    boxes = boxes[mask]
    scores = scores[mask]
    keypoints = keypoints[mask]

    keep = _greedy_nms(boxes, scores, NMS_IOU_THRESHOLD)

    pad_x, pad_y = pad
    results = []
    for idx in keep:
        kp_dict: dict[int, Keypoint] = {}
        for coco_idx, landmark in _COCO_TO_LANDMARK.items():
            x, y, visibility = keypoints[idx, coco_idx]
            orig_x = (float(x) - pad_x) / scale
            orig_y = (float(y) - pad_y) / scale
            kp_dict[int(landmark)] = Keypoint(x=orig_x / w, y=orig_y / h, visibility=float(visibility))
        results.append(PoseResult(keypoints=kp_dict))
    return results
