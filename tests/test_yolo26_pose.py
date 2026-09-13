import numpy as np

from pose_controller.inference.backends import _yolo26_pose
from pose_controller.inference.pose import Landmark

NUM_KEYPOINTS = 17


def _keypoints_array(visible_value=0.9):
    """[1, 17, 3] keypoints array with all points at a fixed position and
    visibility, so tests only need to vary what they care about."""
    kp = np.zeros((1, NUM_KEYPOINTS, 3), dtype=np.float32)
    kp[0, :, 0] = 320.0  # x, model-space (640x640, square frame -> no pad/scale)
    kp[0, :, 1] = 320.0  # y
    kp[0, :, 2] = visible_value
    return kp


def test_detect_poses_filters_below_score_threshold():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array([[100.0, 100.0, 300.0, 500.0]], dtype=np.float32)
    scores = np.array([0.1], dtype=np.float32)  # below NMS_SCORE_THRESHOLD (0.45)
    keypoints = _keypoints_array()

    def detector_infer(_input_nhwc):
        return boxes, scores, keypoints

    assert _yolo26_pose.detect_poses(frame, detector_infer) == []


def test_detect_poses_returns_a_result_above_threshold():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array([[100.0, 100.0, 300.0, 500.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    keypoints = _keypoints_array()

    def detector_infer(_input_nhwc):
        return boxes, scores, keypoints

    results = _yolo26_pose.detect_poses(frame, detector_infer)

    assert len(results) == 1


def test_detect_poses_maps_coco_keypoints_onto_landmark_enum():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array([[100.0, 100.0, 300.0, 500.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    keypoints = np.zeros((1, NUM_KEYPOINTS, 3), dtype=np.float32)
    # COCO index 9 = left_wrist, 10 = right_wrist -- give them distinct,
    # identifiable positions to confirm the index mapping, not just that
    # *some* keypoint landed somewhere.
    keypoints[0, 9] = [64.0, 128.0, 0.8]  # left_wrist
    keypoints[0, 10] = [192.0, 256.0, 0.7]  # right_wrist
    keypoints[0, 5] = [96.0, 96.0, 0.9]  # left_shoulder
    keypoints[0, 6] = [224.0, 96.0, 0.9]  # right_shoulder

    def detector_infer(_input_nhwc):
        return boxes, scores, keypoints

    results = _yolo26_pose.detect_poses(frame, detector_infer)

    assert len(results) == 1
    pose = results[0]
    left_wrist = pose.get(Landmark.LEFT_WRIST)
    right_wrist = pose.get(Landmark.RIGHT_WRIST)
    assert left_wrist is not None and right_wrist is not None
    # 640x640 square frame, no resize_pad scaling/padding -- model-space
    # pixels divide directly by frame width/height to normalize.
    assert np.isclose(left_wrist.x, 64.0 / 640)
    assert np.isclose(left_wrist.y, 128.0 / 640)
    assert np.isclose(left_wrist.visibility, 0.8)
    assert np.isclose(right_wrist.x, 192.0 / 640)
    assert np.isclose(right_wrist.visibility, 0.7)


def test_detect_poses_maps_the_full_17_coco_keypoints_not_just_upper_body():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array([[100.0, 100.0, 300.0, 500.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    keypoints = np.zeros((1, NUM_KEYPOINTS, 3), dtype=np.float32)
    keypoints[0, 1] = [10.0, 10.0, 0.9]  # left_eye
    keypoints[0, 4] = [20.0, 10.0, 0.9]  # right_ear
    keypoints[0, 13] = [30.0, 300.0, 0.9]  # left_knee
    keypoints[0, 16] = [40.0, 400.0, 0.9]  # right_ankle

    def detector_infer(_input_nhwc):
        return boxes, scores, keypoints

    pose = _yolo26_pose.detect_poses(frame, detector_infer)[0]

    assert pose.get(Landmark.LEFT_EYE) is not None
    assert pose.get(Landmark.RIGHT_EAR) is not None
    assert pose.get(Landmark.LEFT_KNEE) is not None
    assert np.isclose(pose.get(Landmark.LEFT_KNEE).y, 300.0 / 640)
    assert pose.get(Landmark.RIGHT_ANKLE) is not None
    assert np.isclose(pose.get(Landmark.RIGHT_ANKLE).y, 400.0 / 640)


def test_detect_poses_applies_nms_across_overlapping_boxes():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array(
        [[100.0, 100.0, 300.0, 500.0], [110.0, 110.0, 310.0, 510.0], [400.0, 400.0, 500.0, 500.0]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keypoints = np.concatenate([_keypoints_array()] * 3, axis=0)

    def detector_infer(_input_nhwc):
        return boxes, scores, keypoints

    results = _yolo26_pose.detect_poses(frame, detector_infer)

    # The heavily-overlapping lower-score box is suppressed; the far-away
    # box survives independently.
    assert len(results) == 2


def test_decode_raw_output_splits_box_score_and_keypoints():
    # 2 anchors, 56 channels each: [cx, cy, w, h, conf, 17*(x,y,v)...]
    raw = np.zeros((56, 2), dtype=np.float32)
    raw[:4, 0] = [100.0, 100.0, 40.0, 60.0]  # anchor 0 box (cxcywh)
    raw[4, 0] = 0.9  # anchor 0 confidence
    raw[5 + 9 * 3 : 5 + 9 * 3 + 3, 0] = [64.0, 128.0, 0.8]  # anchor 0 left_wrist (coco 9)
    raw[:4, 1] = [300.0, 300.0, 20.0, 20.0]  # anchor 1 box
    raw[4, 1] = 0.2  # anchor 1 confidence

    boxes_xyxy, scores, keypoints = _yolo26_pose.decode_raw_output(raw)

    assert boxes_xyxy.shape == (2, 4)
    assert np.allclose(boxes_xyxy[0], [80.0, 70.0, 120.0, 130.0])  # cx-w/2, cy-h/2, cx+w/2, cy+h/2
    assert np.isclose(scores[0], 0.9)
    assert np.isclose(scores[1], 0.2)
    assert keypoints.shape == (2, 17, 3)
    assert np.allclose(keypoints[0, 9], [64.0, 128.0, 0.8])


def test_detect_poses_decodes_raw_ultralytics_output_end_to_end():
    # Confirms detect_poses works when fed decode_raw_output's contract
    # directly (as yolo26_qnn.Yolo26PoseEstimator wires it), not just the
    # AI-Hub-style already-decoded contract the other tests in this file use.
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    raw = np.zeros((56, 1), dtype=np.float32)
    raw[:4, 0] = [200.0, 200.0, 100.0, 100.0]
    raw[4, 0] = 0.9
    raw[5 + 5 * 3 : 5 + 5 * 3 + 3, 0] = [96.0, 96.0, 0.9]  # left_shoulder

    def detector_infer(_input_nhwc):
        return _yolo26_pose.decode_raw_output(raw)

    results = _yolo26_pose.detect_poses(frame, detector_infer)

    assert len(results) == 1
    left_shoulder = results[0].get(Landmark.LEFT_SHOULDER)
    assert np.isclose(left_shoulder.x, 96.0 / 640)
    assert np.isclose(left_shoulder.y, 96.0 / 640)


def test_detect_poses_unscales_keypoints_for_non_square_frame():
    # 200x400 (HxW) frame padded/scaled into 640x640: scale = 640/400 = 1.6,
    # resized content is 320x640 (HxW), vertical pad = (640-320)/2 = 160.
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    boxes = np.array([[0.0, 160.0, 64.0, 224.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    keypoints = np.zeros((1, NUM_KEYPOINTS, 3), dtype=np.float32)
    # A keypoint sitting exactly at the padded content's top-left corner.
    keypoints[0, 5] = [0.0, 160.0, 0.9]  # left_shoulder

    def detector_infer(_input_nhwc):
        return boxes, scores, keypoints

    results = _yolo26_pose.detect_poses(frame, detector_infer)

    left_shoulder = results[0].get(Landmark.LEFT_SHOULDER)
    # (x - pad_x) / scale = 0, (y - pad_y) / scale = 0 -> normalized (0, 0)
    assert np.isclose(left_shoulder.x, 0.0)
    assert np.isclose(left_shoulder.y, 0.0)
