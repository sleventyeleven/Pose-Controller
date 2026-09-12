"""Pure numpy/opencv pre/post-processing for HRNetPose: a top-down,
single-person 256x192 landmark model, evaluated as a second alternative
landmark stage alongside BlazePose (see `qnn.py`) and YOLO26-Pose (see
`yolo26_qnn.py`) -- not a replacement for either yet. Same *role* as
BlazePose's landmark stage (this reuses the existing YOLOv8n-det first
stage and `_blazepose.py`'s box-to-ROI crop geometry, unlike YOLO26-Pose,
which is fully end-to-end), chosen specifically because it uses the same
proven `w8a8` quantization path as the rest of this project's on-device
models where YOLO26 (`w8a16`) hit a reproducible compile failure on the
Q6A -- see `docs/backlog.md`.

Also a candidate fix for the same "gestures only register facing the
camera" symptom YOLO26-Pose was evaluated for: BlazePose's landmark model
gates its *entire* output behind one scalar confidence (`MIN_LANDMARK_
SCORE`), so a pose with a perfectly visible arm can be rejected wholesale
if that one number dips. A heatmap model naturally reports independent
per-keypoint confidence (each keypoint's own heatmap peak) with no
overall gate to reject the whole pose behind -- untested until this runs
on real footage, same caveat as the YOLO26-Pose evaluation.

Model I/O confirmed directly from the AI-Hub-compiled model's own
`input_spec`/`output_spec` (`qai_hub.get_job(job_id).get_target_model()`),
not assumed: input uint8 NHWC `(1, 256, 192, 3)`; output uint8 NHWC
heatmaps `(1, 64, 48, 17)` -- 17 COCO keypoints, each a 64x48 (H, W)
confidence map at stride 4 relative to the 256x192 input (the standard
HRNet heatmap convention: 256/4=64, 192/4=48).
"""

from __future__ import annotations

import cv2
import numpy as np

from pose_controller.inference.backends._blazepose import _crop_resize_matrix, _roi_corners_from_box
from pose_controller.inference.backends._yolo26_pose import _COCO_TO_LANDMARK
from pose_controller.inference.pose import Keypoint, PoseResult

INPUT_SIZE = (256, 192)  # H, W -- HRNetPose::get_input_spec default
HEATMAP_SIZE = (64, 48)  # H, W -- confirmed from the compiled model's real output_spec, see module docstring
HEATMAP_STRIDE = 4  # INPUT_SIZE[i] / HEATMAP_SIZE[i]


def _decode_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """`heatmaps`: `[H, W, 17]` dequantized confidence maps, one per COCO
    keypoint. Returns `[17, 3]` (x, y, confidence) in `INPUT_SIZE` pixel
    space. Confidence is simply the heatmap's peak value, clamped to
    [0, 1] -- no sub-pixel refinement (e.g. neighbor-weighted offset) is
    done; this is a first pass, worth revisiting once this has real
    footage to check accuracy against (see docs/backlog.md)."""
    h, w, num_kp = heatmaps.shape
    flat = heatmaps.reshape(-1, num_kp)
    peak_idx = np.argmax(flat, axis=0)
    confidence = np.clip(flat[peak_idx, np.arange(num_kp)], 0.0, 1.0)
    py, px = np.unravel_index(peak_idx, (h, w))
    return np.stack(
        [px.astype(np.float32) * HEATMAP_STRIDE, py.astype(np.float32) * HEATMAP_STRIDE, confidence],
        axis=1,
    )


def detect_pose_from_box(
    frame_rgb: np.ndarray, box_xyxy: np.ndarray, landmark_infer: callable
) -> PoseResult | None:
    """Crop `box_xyxy`'s region out of `frame_rgb` (already RGB, matching
    `_blazepose`'s convention), run HRNetPose, and map the result back to
    original-frame normalized [0,1] coordinates.

    `landmark_infer(input_nhwc_float32) -> heatmaps[64, 48, 17]` is a
    backend-provided callable (e.g. wrapping `OnnxQnnRunner`).

    Returns None only if `_COCO_TO_LANDMARK` maps to nothing (never
    happens in practice -- kept as a defensive contract match with
    `_blazepose`'s box-driven path, which can return None from
    `MIN_LANDMARK_SCORE`; this model has no equivalent single gate)."""
    frame_h, frame_w = frame_rgb.shape[:2]
    roi_corners = _roi_corners_from_box(box_xyxy, frame_shape_hw=(frame_h, frame_w))
    output_wh = (INPUT_SIZE[1], INPUT_SIZE[0])
    affine = _crop_resize_matrix(roi_corners, output_wh)
    cropped = cv2.warpAffine(frame_rgb, affine, output_wh)
    input_nhwc = (cropped.astype(np.float32) / 255.0)[None]

    heatmaps = landmark_infer(input_nhwc)
    xy_conf = _decode_heatmaps(heatmaps)

    inverse_affine = cv2.invertAffineTransform(affine)
    xy = xy_conf[:, :2]
    orig_xy = (inverse_affine[:, :2] @ xy.T + inverse_affine[:, 2:]).T

    kp_dict: dict[int, Keypoint] = {}
    for coco_idx, landmark in _COCO_TO_LANDMARK.items():
        x, y = orig_xy[coco_idx]
        confidence = float(xy_conf[coco_idx, 2])
        kp_dict[int(landmark)] = Keypoint(x=x / frame_w, y=y / frame_h, visibility=confidence)
    if not kp_dict:
        return None
    return PoseResult(keypoints=kp_dict)


def detect_poses_from_boxes(
    frame_bgr_uint8: np.ndarray, boxes_xyxy: list[np.ndarray], landmark_infer: callable
) -> list[PoseResult]:
    """Run HRNetPose against each pre-computed person box (already in
    original-frame pixel space, e.g. from `_yolo_detect.detect_persons`)
    -- same role as `_blazepose.detect_poses_from_boxes`, different
    model/decode. Returns one `PoseResult` per box (never fewer -- see
    `detect_pose_from_box`)."""
    frame_rgb = cv2.cvtColor(frame_bgr_uint8, cv2.COLOR_BGR2RGB)
    results = []
    for box in boxes_xyxy:
        pose = detect_pose_from_box(frame_rgb, box, landmark_infer)
        if pose is not None:
            results.append(pose)
    return results
