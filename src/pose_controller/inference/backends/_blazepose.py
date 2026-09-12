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

The reference app itself (`MediaPipeApp._run_box_detector`) only ever
keeps the single top-scoring NMS survivor -- genuinely multi-person
detection (milestone 3) required extending past that reference behavior
rather than porting it directly: `_select_detections` below runs full
greedy NMS and returns every surviving detection up to `MAX_PERSONS`,
each independently run through the landmark model.
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
MAX_PERSONS = 8  # bounds landmark-model calls per frame; well above realistic use
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


def _boxes_xyxy(decoded: np.ndarray) -> np.ndarray:
    """decoded is [N, 6, 2] from _decode_boxes; pair 0 = box center,
    pair 1 = box w/h. Returns [N, 4] xyxy, used only for NMS -- the
    landmark ROI itself comes from keypoints, not this box (see
    _compute_roi_corners)."""
    center = decoded[:, 0]
    half_wh = decoded[:, 1] / 2
    return np.concatenate([center - half_wh, center + half_wh], axis=1)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box and an array of xyxy boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / union, 0.0)


def _greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Standard greedy NMS. Returns kept indices (into `boxes`/`scores`),
    highest score first."""
    order = list(np.argsort(-scores))
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        remaining = np.array(order)
        ious = _iou(boxes[i], boxes[remaining])
        order = [idx for idx, iou in zip(order, ious) if iou <= iou_threshold]
    return keep


def _cxcywh_to_xyxy(boxes_cxcywh: np.ndarray) -> np.ndarray:
    """Convert `[N, 4]` boxes from (center-x, center-y, width, height) --
    Ultralytics' own head-decode output format (see
    `ultralytics.nn.modules.head.Detect._get_decode_boxes`, confirmed from
    source, not assumed) -- to `[N, 4]` xyxy, the format every NMS/box
    consumer in this project already expects. Used by the raw-output
    decode for Ultralytics-QNN-exported models (`_yolo26_pose.py`,
    `_yolo26_detect.py`) -- AI-Hub-exported models decode boxes to xyxy
    inside their own graph already, so they never needed this."""
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    half_w, half_h = w / 2, h / 2
    return np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)


def _select_detections(
    box_coords_flat: np.ndarray,
    box_scores_flat: np.ndarray,
    anchors: np.ndarray,
) -> list[np.ndarray]:
    """Decode detector output and return the 4 auxiliary keypoints
    ([4, 2] each) for every detection surviving score-thresholding + NMS,
    highest score first, capped at MAX_PERSONS. Empty list if none pass
    threshold. The detector's box is used only for NMS dedup -- the
    pose-specific ROI computation (_compute_roi_corners) derives the ROI
    entirely from keypoints, not the box (see its docstring)."""
    scores = _sigmoid(np.clip(box_scores_flat, -DETECT_SCORE_CLIP, DETECT_SCORE_CLIP))
    above_threshold = np.nonzero(scores >= MIN_DETECTOR_SCORE)[0]
    if above_threshold.size == 0:
        return []

    decoded = _decode_boxes(box_coords_flat, anchors, DETECTOR_INPUT_SIZE)
    boxes_xyxy = _boxes_xyxy(decoded[above_threshold])
    kept_local = _greedy_nms(boxes_xyxy, scores[above_threshold], NMS_IOU_THRESHOLD)
    kept_global = above_threshold[kept_local][:MAX_PERSONS]

    return [decoded[idx, 2:] for idx in kept_global]


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


def _run_landmark_on_roi(
    frame_rgb: np.ndarray, roi_corners: np.ndarray, landmark_infer: callable
) -> np.ndarray | None:
    """Crop `roi_corners` out of `frame_rgb`, run the landmark model, and
    map the result back to original-frame pixel space. Shared by both the
    keypoint-derived ROI path (_landmarks_for_keypoints, BlazePose's own
    detector) and the box-derived ROI path (_roi_corners_from_box, an
    external detector like YOLO -- see detect_poses_from_boxes). Returns
    [25, 4] landmarks, or None if below MIN_LANDMARK_SCORE."""
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


def _landmarks_for_keypoints(
    frame_rgb: np.ndarray, keypoints: np.ndarray, landmark_infer: callable
) -> np.ndarray | None:
    """Run the ROI -> crop -> landmark stage for one detection's auxiliary
    keypoints (already in original-frame pixel space). Returns [25, 4]
    landmarks in original-frame pixel space, or None if below
    MIN_LANDMARK_SCORE."""
    roi_corners = _compute_roi_corners(keypoints)
    return _run_landmark_on_roi(frame_rgb, roi_corners, landmark_infer)


# Margin applied around an external detector's (e.g. YOLO) person box when
# building the landmark model's square crop -- the box already tightly
# bounds head-to-feet, so this only needs to add framing slack, unlike
# DETECT_BOX_SCALE above (which inflates a much smaller aux-keypoint
# distance for BlazePose's own detector). 1.75, not a smaller "just a bit
# of margin" value, because the landmark model was trained on crops from
# BlazePose's own detector's specific keypoint-derived ROI convention, not
# a generic tight bounding box -- empirically swept against a known-good
# two-person reference photo (see scripts/qnn_spike.md): 1.25 scored 0.09
# (fails MIN_LANDMARK_SCORE) for both people, 1.75 scored 0.99 / 0.71,
# matching BlazePose's own detector's confidence on the same photo.
BOX_ROI_SCALE = 1.75


def _roi_corners_from_box(
    box_xyxy: np.ndarray, scale: float = BOX_ROI_SCALE, frame_shape_hw: tuple[int, int] | None = None
) -> np.ndarray:
    """Build an axis-aligned (unrotated) square ROI centered on an external
    detector's bounding box, sized to the box's longer side times `scale`.
    Unlike _compute_roi_corners, there's no rotation cue available from a
    plain xyxy box, so theta is always 0 -- fine for the upright,
    camera-facing framing this project targets. Returns 4 corners
    [top-left, bottom-left, top-right, bottom-right], shape [4, 2],
    matching _compute_roi_corners's contract so both feed
    _run_landmark_on_roi unchanged.

    When `frame_shape_hw` is given, the ROI's center is shifted (not
    resized) to stay within the frame bounds wherever the ROI is smaller
    than the frame in that dimension -- otherwise a person standing near
    an edge (very common: BOX_ROI_SCALE's margin is tuned against a
    reference photo where people had comfortable headroom, but a person
    filling most of a real camera's vertical FOV does not) gets a chunk of
    the ROI sampling out-of-bounds black padding instead of margin,
    tanking the landmark model's confidence for no visual reason -- found
    via real captures on the physical Q6A scoring far below a demo photo
    at the same scale, see scripts/qnn_spike.md."""
    x1, y1, x2, y2 = box_xyxy
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    half = side / 2

    if frame_shape_hw is not None:
        frame_h, frame_w = frame_shape_hw
        if side <= frame_w:
            xc = np.clip(xc, half, frame_w - half)
        if side <= frame_h:
            yc = np.clip(yc, half, frame_h - half)

    return np.array(
        [[xc - half, yc - half], [xc - half, yc + half], [xc + half, yc - half], [xc + half, yc + half]],
        dtype=np.float32,
    )


def detect_poses_from_boxes(
    frame_bgr_uint8: np.ndarray,
    boxes_xyxy: list[np.ndarray],
    landmark_infer: callable,
) -> list[np.ndarray]:
    """Like detect_poses, but skips BlazePose's own low-resolution (128x128)
    detector entirely and instead runs the landmark model against
    pre-computed person boxes (already in original-frame pixel space, e.g.
    from a higher-resolution external detector). See scripts/qnn_spike.md
    for why BlazePose's bundled detector loses small/distant subjects at
    ordinary room framing distance -- this is the fix.

    Returns a list of [25, 4] (x, y, z, visibility) landmark arrays, one
    per box that clears MIN_LANDMARK_SCORE, in the same order as
    `boxes_xyxy`."""
    frame_rgb = cv2.cvtColor(frame_bgr_uint8, cv2.COLOR_BGR2RGB)
    frame_shape_hw = frame_bgr_uint8.shape[:2]

    results = []
    for box in boxes_xyxy:
        roi_corners = _roi_corners_from_box(box, frame_shape_hw=frame_shape_hw)
        landmarks = _run_landmark_on_roi(frame_rgb, roi_corners, landmark_infer)
        if landmarks is not None:
            results.append(landmarks)
    return results


def detect_poses(
    frame_bgr_uint8: np.ndarray,
    detector_infer: callable,
    landmark_infer: callable,
    anchors: np.ndarray,
) -> list[np.ndarray]:
    """Run the full detector -> ROI -> landmark pipeline on one BGR frame,
    for every person detected (up to MAX_PERSONS).

    `detector_infer(input_nhwc_float32) -> (box_coords[896,12], box_scores[896])`
    `landmark_infer(input_nhwc_float32) -> (score: float, landmarks[25,4])`
    are backend-provided callables (e.g. wrapping OnnxQnnRunner or the
    CPU dev path) so this function stays backend-agnostic. Inputs are
    NHWC (batch, H, W, C), matching the compiled QNN context binaries'
    actual tensor layout -- confirmed via `qnn-context-binary-utility`
    (see scripts/qnn_spike.md); this differs from the original PyTorch
    model's NCHW input_spec, which describes the pre-export model, not
    the AI-Hub-compiled artifact.

    Returns a list of [25, 4] (x, y, z, visibility) landmark arrays, one
    per person, in pixel space of the original frame, with visibility
    passed through a sigmoid into [0, 1]. Empty list if no one was
    detected above threshold.
    """
    frame_rgb = cv2.cvtColor(frame_bgr_uint8, cv2.COLOR_BGR2RGB)

    det_input, det_scale, det_pad = resize_pad(frame_rgb, DETECTOR_INPUT_SIZE)
    det_input_nhwc = (det_input.astype(np.float32) / 255.0)[None]

    box_coords, box_scores = detector_infer(det_input_nhwc)
    detections = _select_detections(box_coords, box_scores, anchors)
    if not detections:
        return []

    # Map from the padded/scaled 128x128 detector-input space back to the
    # original frame's pixel space.
    pad_x, pad_y = det_pad
    results = []
    for keypoints in detections:
        keypoints = keypoints.copy()
        keypoints[:, 0] = (keypoints[:, 0] - pad_x) / det_scale
        keypoints[:, 1] = (keypoints[:, 1] - pad_y) / det_scale

        landmarks = _landmarks_for_keypoints(frame_rgb, keypoints, landmark_infer)
        if landmarks is not None:
            results.append(landmarks)

    return results
