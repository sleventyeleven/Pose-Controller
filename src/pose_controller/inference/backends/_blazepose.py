"""Pure numpy/opencv port of the BlazePose (MediaPipe Pose) two-stage
detector + landmark pipeline.

Ported from Qualcomm AI Hub Models' reference implementation
(qai_hub_models.models.templates.mediapipe.app / mediapipe_pose.model,
BSD-3-Clause) so this can run on-device without requiring PyTorch (see
docs/storage-footprint.md -- PyTorch is a dev-machine-only dependency for
exporting models, not something that should ship on the Q6A). Every
constant and formula here was cross-checked against that reference and
against real outputs from the compiled models running on this project's
physical Q6A (see scripts/qnn_spike.md) rather than re-derived from
scratch, to avoid subtly-wrong BlazePose math.

Because the reference app's own box-selection logic only ever keeps the
single top-scoring detection (`_run_box_detector` breaks after the first
NMS survivor), this port only supports one person per frame -- matching
the CPU dev backend's single-person MediaPipe behavior, and this
project's current milestone scope.
"""

from __future__ import annotations

import numpy as np
import cv2

DETECTOR_INPUT_SIZE = (128, 128)  # H, W
LANDMARK_INPUT_SIZE = (256, 256)  # H, W
NUM_ANCHORS = 896
NUM_VALID_LANDMARKS = 25  # landmarks past this index are untrained/unused
MIN_DETECTOR_SCORE = 0.75
NMS_IOU_THRESHOLD = 0.3
MIN_LANDMARK_SCORE = 0.5
DETECT_SCORE_CLIP = 100.0
DETECT_BOX_OFFSET_XY = 0.0  # no-op for the pose model specifically (kept for clarity)
DETECT_BOX_SCALE = 1.5
# Indices into the detector's 4 auxiliary keypoints (not the final 25
# landmarks) used to compute the ROI's rotation.
KEYPOINT_ROTATION_START_IDX = 2
KEYPOINT_ROTATION_END_IDX = 3
ROTATION_OFFSET_RADS = np.pi / 2


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable form -- np.where evaluates both branches
    # unconditionally, so exp() must never overflow in either one
    # regardless of which branch is selected. exp(-|x|) never does; the
    # detector scores are clipped to +-100 before this, which would
    # overflow float32 exp(x) directly.
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def resize_pad(
    frame_hwc_uint8: np.ndarray, dst_size_hw: tuple[int, int]
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize (preserving aspect ratio) and center-pad `frame_hwc_uint8` to
    `dst_size_hw`. Mirrors qai_hub_models.utils.image_processing.resize_pad's
    default (centered) behavior, adapted to operate on a numpy HWC uint8
    image via cv2 instead of a torch NCHW tensor.

    Returns the resized+padded image plus the scale factor and (left, top)
    pixel padding applied, both needed to map coordinates back later.
    """
    height, width = frame_hwc_uint8.shape[:2]
    dst_h, dst_w = dst_size_hw

    scale = min(dst_h / height, dst_w / width)
    new_h = int(height * scale)
    new_w = int(width * scale)

    resized = cv2.resize(frame_hwc_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = dst_h - new_h
    pad_w = dst_w - new_w
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    return padded, scale, (pad_left, pad_top)


def _decode_boxes(raw_coords: np.ndarray, anchors: np.ndarray, img_size_hw: tuple[int, int]) -> np.ndarray:
    """Decode raw detector box/keypoint regressions using the model's
    anchors. `raw_coords` is [N, 12] (indices 0-3: box cx,cy,w,h; 4-11: 4
    keypoint x,y pairs). `anchors` is [N, 2, 2] ([[x_off, y_off], [x_scale,
    y_scale]], offsets normalized [0,1]). Returns [N, 6, 2] decoded pairs
    (pair 0 = box center, pair 1 = box w/h, pairs 2-5 = keypoints)."""
    h_size, w_size = img_size_hw
    coords = raw_coords.reshape(-1, 6, 2)

    offset = anchors[:, 0:1, :] * np.array([w_size, h_size], dtype=np.float32)
    scale = anchors[:, 1:2, :]

    # Only the box w/h pair (index 1) is exempt from the anchor offset.
    mask = (np.arange(6) != 1).reshape(6, 1)
    return coords * scale + (offset * mask)


def _top_detection(
    box_coords_flat: np.ndarray,
    box_scores_flat: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray | None:
    """Decode detector output and return the 4 auxiliary keypoints ([4, 2])
    for the single highest-scoring detection above MIN_DETECTOR_SCORE, or
    None. The detector's box itself isn't needed: the pose-specific ROI
    computation (_compute_roi_corners) derives the ROI entirely from these
    keypoints, not the box (see its docstring). NMS is intentionally
    skipped beyond taking the top score: the reference app itself only
    ever keeps the first NMS survivor (see module docstring), so for a
    single detection, thresholding + argmax is equivalent."""
    scores = _sigmoid(np.clip(box_scores_flat, -DETECT_SCORE_CLIP, DETECT_SCORE_CLIP))
    best_idx = int(np.argmax(scores))
    if scores[best_idx] < MIN_DETECTOR_SCORE:
        return None

    decoded = _decode_boxes(box_coords_flat, anchors, DETECTOR_INPUT_SIZE)
    return decoded[best_idx, 2:]  # 4 keypoints, [4, 2]


def _compute_roi_corners(keypoints: np.ndarray) -> np.ndarray:
    """Compute the (possibly rotated) square ROI used as landmark detector
    input.

    Unlike the generic MediaPipe template (which derives the ROI from the
    detector's bounding box), the pose-specific pipeline
    (MediaPipePoseApp._compute_object_roi) derives it entirely from the
    detector's auxiliary keypoints: the ROI is centered on keypoint
    KEYPOINT_ROTATION_START_IDX, with a square side length of 2x the
    distance to keypoint KEYPOINT_ROTATION_END_IDX (times DETECT_BOX_SCALE).
    The detector's box is not used here at all -- easy to miss when porting
    from the generic template.

    Returns 4 corners [top-left, bottom-left, top-right, bottom-right],
    shape [4, 2]."""
    kp_start = keypoints[KEYPOINT_ROTATION_START_IDX]
    kp_end = keypoints[KEYPOINT_ROTATION_END_IDX]
    theta = np.arctan2(kp_start[1] - kp_end[1], kp_start[0] - kp_end[0]) - ROTATION_OFFSET_RADS

    xc, yc = kp_start
    side = np.hypot(kp_end[0] - kp_start[0], kp_end[1] - kp_start[1]) * 2 * DETECT_BOX_SCALE

    half = side / 2
    unit_square = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32) * half
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32
    )
    return unit_square @ rotation.T + np.array([xc, yc], dtype=np.float32)


def _crop_resize_matrix(roi_corners: np.ndarray, output_size_wh: tuple[int, int]) -> np.ndarray:
    """Affine matrix mapping the (top-left, bottom-left, top-right) ROI
    corners to the output image's corners -- crops, un-rotates, and resizes
    the ROI to `output_size_wh` in a single warp."""
    dst = np.array(
        [[0, 0], [0, output_size_wh[1] - 1], [output_size_wh[0] - 1, 0]], dtype=np.float32
    )
    return cv2.getAffineTransform(roi_corners[:3].astype(np.float32), dst)


def detect_pose(
    frame_bgr_uint8: np.ndarray,
    detector_infer: callable,
    landmark_infer: callable,
    anchors: np.ndarray,
) -> np.ndarray | None:
    """Run the full detector -> ROI -> landmark pipeline on one BGR frame.

    `detector_infer(input_nhwc_float32) -> (box_coords[896,12], box_scores[896])`
    `landmark_infer(input_nhwc_float32) -> (score: float, landmarks[25,4])`
    are backend-provided callables (e.g. wrapping OnnxQnnRunner or the
    CPU dev path) so this function stays backend-agnostic. Inputs are
    NHWC (batch, H, W, C), matching the compiled QNN context binaries'
    actual tensor layout -- confirmed via `qnn-context-binary-utility`
    (see scripts/qnn_spike.md); this differs from the original PyTorch
    model's NCHW input_spec, which describes the pre-export model, not
    the AI-Hub-compiled artifact.

    Returns landmarks as [25, 4] (x, y, z, visibility) in pixel space of
    the original frame, with visibility passed through a sigmoid into
    [0, 1], or None if no person was detected above threshold.
    """
    frame_rgb = cv2.cvtColor(frame_bgr_uint8, cv2.COLOR_BGR2RGB)

    det_input, det_scale, det_pad = resize_pad(frame_rgb, DETECTOR_INPUT_SIZE)
    det_input_nhwc = (det_input.astype(np.float32) / 255.0)[None]

    box_coords, box_scores = detector_infer(det_input_nhwc)
    keypoints = _top_detection(box_coords, box_scores, anchors)
    if keypoints is None:
        return None

    # Map from the padded/scaled 128x128 detector-input space back to the
    # original frame's pixel space.
    pad_x, pad_y = det_pad
    keypoints = keypoints.copy()
    keypoints[:, 0] = (keypoints[:, 0] - pad_x) / det_scale
    keypoints[:, 1] = (keypoints[:, 1] - pad_y) / det_scale

    roi_corners = _compute_roi_corners(keypoints)
    landmark_wh = (LANDMARK_INPUT_SIZE[1], LANDMARK_INPUT_SIZE[0])
    affine = _crop_resize_matrix(roi_corners, landmark_wh)
    cropped = cv2.warpAffine(frame_rgb, affine, landmark_wh)
    landmark_input_nhwc = (cropped.astype(np.float32) / 255.0)[None]

    ld_score, landmarks = landmark_infer(landmark_input_nhwc)
    if ld_score < MIN_LANDMARK_SCORE:
        return None

    landmarks = landmarks.copy()
    landmarks[:, 0] *= LANDMARK_INPUT_SIZE[1]
    landmarks[:, 1] *= LANDMARK_INPUT_SIZE[0]

    inverse_affine = cv2.invertAffineTransform(affine)
    xy = landmarks[:, :2]
    landmarks[:, :2] = (inverse_affine[:, :2] @ xy.T + inverse_affine[:, 2:]).T
    landmarks[:, 3] = _sigmoid(landmarks[:, 3])

    return landmarks
