import numpy as np

from pose_controller.inference.backends import _yolo_detect


def test_detect_persons_filters_non_person_classes():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 100.0, 100.0]], dtype=np.float32)
    scores = np.array([0.99], dtype=np.float32)
    classes = np.array([2], dtype=np.int64)  # COCO class 2 = "car", not person

    def detector_infer(_input_nhwc):
        return boxes, scores, classes

    assert _yolo_detect.detect_persons(frame, detector_infer) == []


def test_detect_persons_filters_below_score_threshold():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 100.0, 100.0]], dtype=np.float32)
    scores = np.array([0.1], dtype=np.float32)  # below NMS_SCORE_THRESHOLD (0.45)
    classes = np.array([_yolo_detect.PERSON_CLASS_ID], dtype=np.int64)

    def detector_infer(_input_nhwc):
        return boxes, scores, classes

    assert _yolo_detect.detect_persons(frame, detector_infer) == []


def test_detect_persons_returns_box_in_original_frame_space():
    # Square 640x640 frame -> resize_pad is a no-op (scale=1, pad=(0,0)),
    # so the returned box should match the model-space box exactly.
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 20.0, 110.0, 220.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([_yolo_detect.PERSON_CLASS_ID], dtype=np.int64)

    def detector_infer(_input_nhwc):
        return boxes, scores, classes

    results = _yolo_detect.detect_persons(frame, detector_infer)

    assert len(results) == 1
    assert np.allclose(results[0], [10.0, 20.0, 110.0, 220.0])


def test_detect_persons_unscales_box_for_non_square_frame():
    # 200x400 (HxW) frame padded/scaled into 640x640: scale = 640/400 = 1.6,
    # resized content is 320x640 (HxW), vertical pad = (640-320)/2 = 160 each side.
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    # A model-space box sitting exactly at the padded content's top-left corner.
    boxes = np.array([[0.0, 160.0, 64.0, 224.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([_yolo_detect.PERSON_CLASS_ID], dtype=np.int64)

    def detector_infer(_input_nhwc):
        return boxes, scores, classes

    results = _yolo_detect.detect_persons(frame, detector_infer)

    assert len(results) == 1
    # (x - pad_x) / scale, (y - pad_y) / scale
    assert np.allclose(results[0], [0.0, 0.0, 40.0, 40.0])


def test_detect_persons_applies_nms_across_overlapping_person_boxes():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array(
        [[10.0, 10.0, 110.0, 110.0], [15.0, 15.0, 115.0, 115.0], [400.0, 400.0, 500.0, 500.0]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.array([_yolo_detect.PERSON_CLASS_ID] * 3, dtype=np.int64)

    def detector_infer(_input_nhwc):
        return boxes, scores, classes

    results = _yolo_detect.detect_persons(frame, detector_infer)

    # The heavily-overlapping lower-score box is suppressed; the far-away
    # box survives independently -- highest score first.
    assert len(results) == 2
    assert np.allclose(results[0], boxes[0])
    assert np.allclose(results[1], boxes[2])
