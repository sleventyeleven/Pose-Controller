import numpy as np

from pose_controller.inference.backends import _blazepose


def test_sigmoid_matches_naive_for_small_values():
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    naive = 1.0 / (1.0 + np.exp(-x))
    assert np.allclose(_blazepose._sigmoid(x), naive, atol=1e-6)


def test_sigmoid_saturates_without_overflow_warning():
    x = np.array([-1e4, 1e4], dtype=np.float32)
    # Underflow (exp(-10000) -> 0.0) is the correct, expected result here --
    # only overflow/invalid (nan-producing) results indicate a real bug.
    with np.errstate(over="raise", invalid="raise"):
        result = _blazepose._sigmoid(x)
    assert np.allclose(result, [0.0, 1.0])


def test_resize_pad_preserves_aspect_ratio_and_target_size():
    frame = np.zeros((200, 100, 3), dtype=np.uint8)  # tall H=200, W=100
    padded, scale, pad = _blazepose.resize_pad(frame, (128, 128))

    assert padded.shape == (128, 128, 3)
    assert scale == 128 / 200  # height is the binding dimension
    # Width 100 * scale ~= 64, so horizontal padding should split ~(32, 32)
    pad_left, pad_top = pad
    assert pad_top == 0  # height exactly fills the target, no vertical pad
    assert 30 <= pad_left <= 34


def test_decode_boxes_applies_anchor_offset_and_scale():
    # One anchor: offset (0.5, 0.5) normalized, scale (2.0, 2.0).
    anchors = np.array([[[0.5, 0.5], [2.0, 2.0]]], dtype=np.float32)
    img_size = (100, 100)  # H, W
    # 6 pairs: box center, box wh, 4 keypoints -- use 1.0 for every value.
    raw = np.ones((1, 12), dtype=np.float32)

    decoded = _blazepose._decode_boxes(raw, anchors, img_size)

    # Box center (pair 0): raw(1,1) * scale(2,2) + offset(50,50) = (52, 52)
    assert np.allclose(decoded[0, 0], [52.0, 52.0])
    # Box w/h (pair 1): raw(1,1) * scale(2,2), NO offset = (2, 2)
    assert np.allclose(decoded[0, 1], [2.0, 2.0])
    # Keypoint pairs (2-5) get the same offset treatment as the box center.
    assert np.allclose(decoded[0, 2], [52.0, 52.0])


def test_top_detection_returns_none_below_threshold():
    anchors = np.zeros((896, 2, 2), dtype=np.float32)
    anchors[:, 1, :] = 1.0  # scale=1 so decode doesn't blow up
    box_coords = np.zeros((896, 12), dtype=np.float32)
    # Raw score of 0 -> sigmoid 0.5, well below MIN_DETECTOR_SCORE (0.75).
    box_scores = np.zeros(896, dtype=np.float32)

    assert _blazepose._top_detection(box_coords, box_scores, anchors) is None


def test_top_detection_picks_highest_scoring_anchor():
    anchors = np.zeros((896, 2, 2), dtype=np.float32)
    anchors[:, 1, :] = 1.0
    box_coords = np.zeros((896, 12), dtype=np.float32)
    box_coords[42, 2:] = 10.0  # keypoints for the winning anchor
    box_scores = np.full(896, -100.0, dtype=np.float32)
    box_scores[42] = 100.0  # sigmoid ~1.0, clearly above threshold

    keypoints = _blazepose._top_detection(box_coords, box_scores, anchors)

    assert keypoints is not None
    assert keypoints.shape == (4, 2)
    assert np.allclose(keypoints, 10.0)


def test_compute_roi_corners_square_and_centered_when_axis_aligned():
    # kp_start (rotation index 2) directly above kp_end (index 3) by 10px:
    # theta = atan2(50-40, 50-50) - pi/2 = atan2(10, 0) - pi/2 = pi/2 - pi/2 = 0,
    # i.e. no rotation -- an axis-aligned square.
    keypoints = np.zeros((4, 2), dtype=np.float32)
    keypoints[2] = [50.0, 50.0]
    keypoints[3] = [50.0, 40.0]

    corners = _blazepose._compute_roi_corners(keypoints)

    # Side length = distance(start, end) * 2 * DETECT_BOX_SCALE = 10*2*1.5 = 30,
    # centered on kp_start (50, 50) -> corners at (35,35),(35,65),(65,35),(65,65).
    assert corners.shape == (4, 2)
    expected = np.array([[35.0, 35.0], [35.0, 65.0], [65.0, 35.0], [65.0, 65.0]])
    assert np.allclose(corners, expected, atol=1e-3)


def test_crop_resize_matrix_maps_corners_to_output_bounds():
    roi_corners = np.array(
        [[0.0, 0.0], [0.0, 99.0], [99.0, 0.0], [99.0, 99.0]], dtype=np.float32
    )
    affine = _blazepose._crop_resize_matrix(roi_corners, (256, 256))

    def apply(pt):
        return affine[:, :2] @ pt + affine[:, 2]

    assert np.allclose(apply(roi_corners[0]), [0.0, 0.0], atol=1e-3)
    assert np.allclose(apply(roi_corners[1]), [0.0, 255.0], atol=1e-3)
    assert np.allclose(apply(roi_corners[2]), [255.0, 0.0], atol=1e-3)
