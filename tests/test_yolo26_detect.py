import numpy as np

from pose_controller.inference.backends import _yolo26_detect


def test_detect_objects_filters_below_score_threshold():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 100.0, 100.0]], dtype=np.float32)
    scores = np.array([0.1], dtype=np.float32)
    classes = np.array([0], dtype=np.int64)

    def detector_infer(_input_nhwc):
        return boxes, scores, classes

    assert _yolo26_detect.detect_objects(frame, detector_infer) == []


def test_detect_objects_keeps_every_class_not_just_person():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 100.0, 100.0], [400.0, 400.0, 500.0, 500.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 39], dtype=np.int64)  # 0=person, 39=bottle

    def detector_infer(_input_nhwc):
        return boxes, scores, classes

    results = _yolo26_detect.detect_objects(frame, detector_infer)

    assert len(results) == 2
    class_ids = {d.class_id for d in results}
    assert class_ids == {0, 39}


def test_detection_class_name_resolves_from_class_id():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 100.0, 100.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([39], dtype=np.int64)

    def detector_infer(_input_nhwc):
        return boxes, scores, classes

    results = _yolo26_detect.detect_objects(frame, detector_infer)

    assert results[0].class_name == "bottle"


def test_detection_class_name_falls_back_for_out_of_range_id():
    d = _yolo26_detect.Detection(box_xyxy=(0.0, 0.0, 1.0, 1.0), score=0.9, class_id=999)

    assert d.class_name == "class_999"


def test_detect_objects_returns_highest_score_first_after_nms():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = np.array(
        [[10.0, 10.0, 110.0, 110.0], [15.0, 15.0, 115.0, 115.0], [400.0, 400.0, 500.0, 500.0]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.array([0, 0, 16], dtype=np.int64)  # 16=dog

    def detector_infer(_input_nhwc):
        return boxes, scores, classes

    results = _yolo26_detect.detect_objects(frame, detector_infer)

    # The heavily-overlapping lower-score box is suppressed regardless of
    # its class (NMS in this module is class-agnostic, unlike
    # _yolo_detect.py's person-only NMS -- matches this module's own
    # "keep everything" purpose).
    assert len(results) == 2
    assert np.isclose(results[0].score, 0.9)
    assert np.isclose(results[1].score, 0.7)


def test_decode_raw_output_argmaxes_over_classes_and_converts_box():
    # 2 anchors, 84 channels each: [cx, cy, w, h, 80 class scores...]
    raw = np.zeros((84, 2), dtype=np.float32)
    raw[:4, 0] = [100.0, 100.0, 40.0, 60.0]  # anchor 0 box (cxcywh)
    raw[4 + 39, 0] = 0.9  # anchor 0: class 39 (bottle) scores highest
    raw[4 + 5, 0] = 0.3  # a lower-scoring class, should not win
    raw[:4, 1] = [300.0, 300.0, 20.0, 20.0]
    raw[4 + 0, 1] = 0.2  # anchor 1: class 0 (person)

    boxes_xyxy, scores, class_idx = _yolo26_detect.decode_raw_output(raw)

    assert boxes_xyxy.shape == (2, 4)
    assert np.allclose(boxes_xyxy[0], [80.0, 70.0, 120.0, 130.0])
    assert np.isclose(scores[0], 0.9)
    assert class_idx[0] == 39
    assert np.isclose(scores[1], 0.2)
    assert class_idx[1] == 0


def test_detect_objects_decodes_raw_ultralytics_output_end_to_end():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    raw = np.zeros((84, 1), dtype=np.float32)
    raw[:4, 0] = [200.0, 200.0, 100.0, 100.0]
    raw[4 + 16, 0] = 0.9  # class 16 = dog

    def detector_infer(_input_nhwc):
        return _yolo26_detect.decode_raw_output(raw)

    results = _yolo26_detect.detect_objects(frame, detector_infer)

    assert len(results) == 1
    assert results[0].class_name == "dog"


def test_detect_objects_unscales_box_for_non_square_frame():
    # 200x400 (HxW) frame padded/scaled into 640x640: scale = 640/400 = 1.6,
    # vertical pad = 160.
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    boxes = np.array([[0.0, 160.0, 64.0, 224.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([0], dtype=np.int64)

    def detector_infer(_input_nhwc):
        return boxes, scores, classes

    results = _yolo26_detect.detect_objects(frame, detector_infer)

    assert len(results) == 1
    assert np.allclose(results[0].box_xyxy, [0.0, 0.0, 40.0, 40.0])
