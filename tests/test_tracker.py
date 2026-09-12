import numpy as np

from pose_controller.inference.pose import PoseResult
from pose_controller.tracking.tracker import Tracker, _iou


def _pose(bbox):
    return PoseResult(keypoints={}, bbox=bbox)


class _FakeEmbedder:
    """Test double for ReidEmbedder: returns the mean BGR color of the
    cropped region as a 3-element L2-normalized "embedding". Lets tests
    construct frames where "appearance" is just a solid color, without
    needing the real OSNet ONNX model."""

    def embed(self, frame_bgr, bbox_xyxy_normalized):
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = bbox_xyxy_normalized
        crop = frame_bgr[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
        mean_color = crop.reshape(-1, 3).mean(axis=0).astype(np.float64)
        norm = np.linalg.norm(mean_color)
        return mean_color / norm if norm > 0 else mean_color


def _solid_frame(color_bgr, size=100):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = color_bgr
    return frame


_FULL_BBOX = (0.0, 0.0, 1.0, 1.0)


def test_iou_of_identical_boxes_is_one():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_single_person_keeps_same_id_across_frames():
    tracker = Tracker()

    frame1 = tracker.update([_pose((0.1, 0.1, 0.3, 0.5))])
    frame2 = tracker.update([_pose((0.11, 0.1, 0.31, 0.5))])  # slight movement

    assert frame1[0].track_id is not None
    assert frame1[0].track_id == frame2[0].track_id


def test_two_well_separated_people_get_different_ids():
    tracker = Tracker()

    results = tracker.update(
        [_pose((0.0, 0.0, 0.2, 0.4)), _pose((0.6, 0.0, 0.8, 0.4))]
    )

    assert results[0].track_id != results[1].track_id
    assert None not in (results[0].track_id, results[1].track_id)


def test_track_survives_brief_occlusion():
    tracker = Tracker(max_age=5)

    first = tracker.update([_pose((0.1, 0.1, 0.3, 0.5))])
    original_id = first[0].track_id

    # Person briefly not detected for a couple of frames (occlusion).
    tracker.update([])
    tracker.update([])

    # Reappears close to where predicted -- should re-link to the same ID.
    reappeared = tracker.update([_pose((0.1, 0.1, 0.3, 0.5))])

    assert reappeared[0].track_id == original_id


def test_track_dropped_after_max_age_gets_new_id():
    tracker = Tracker(max_age=2)

    first = tracker.update([_pose((0.1, 0.1, 0.3, 0.5))])
    original_id = first[0].track_id

    # Occluded longer than max_age -- track should be dropped.
    tracker.update([])
    tracker.update([])
    tracker.update([])

    reappeared = tracker.update([_pose((0.1, 0.1, 0.3, 0.5))])

    assert reappeared[0].track_id != original_id


def test_detection_without_bbox_is_dropped_not_tracked():
    tracker = Tracker()

    results = tracker.update([_pose(None)])

    assert results == []


def test_min_hits_withholds_id_until_confirmed():
    tracker = Tracker(min_hits=3)

    frame1 = tracker.update([_pose((0.1, 0.1, 0.3, 0.5))])
    frame2 = tracker.update([_pose((0.1, 0.1, 0.3, 0.5))])
    frame3 = tracker.update([_pose((0.1, 0.1, 0.3, 0.5))])

    assert frame1[0].track_id is None
    assert frame2[0].track_id is None
    assert frame3[0].track_id is not None


def test_reid_revives_id_for_matching_appearance_after_max_age():
    tracker = Tracker(max_age=2, embedder=_FakeEmbedder())
    red_frame = _solid_frame((0, 0, 255))

    first = tracker.update([_pose(_FULL_BBOX)], frame_bgr=red_frame)
    original_id = first[0].track_id

    # Occluded longer than max_age -- track drops out of active tracking,
    # but (unlike without an embedder) its appearance is kept in the gallery.
    tracker.update([], frame_bgr=red_frame)
    tracker.update([], frame_bgr=red_frame)
    tracker.update([], frame_bgr=red_frame)

    # Same red person reappears at a totally different position (so IoU
    # matching alone would never relink it) -- appearance should.
    reappeared = tracker.update([_pose((0.6, 0.6, 0.9, 0.9))], frame_bgr=red_frame)

    assert reappeared[0].track_id == original_id


def test_reid_does_not_revive_for_different_appearance():
    tracker = Tracker(max_age=2, embedder=_FakeEmbedder())
    red_frame = _solid_frame((0, 0, 255))
    blue_frame = _solid_frame((255, 0, 0))

    first = tracker.update([_pose(_FULL_BBOX)], frame_bgr=red_frame)
    original_id = first[0].track_id

    tracker.update([], frame_bgr=red_frame)
    tracker.update([], frame_bgr=red_frame)
    tracker.update([], frame_bgr=red_frame)

    # A different-colored (different-looking) person appears in roughly
    # the same place -- should NOT be treated as the same person.
    new_person = tracker.update([_pose(_FULL_BBOX)], frame_bgr=blue_frame)

    assert new_person[0].track_id != original_id


def test_reid_gallery_expires_after_ttl():
    tracker = Tracker(max_age=1, embedder=_FakeEmbedder(), reid_gallery_ttl=2)
    red_frame = _solid_frame((0, 0, 255))

    first = tracker.update([_pose(_FULL_BBOX)], frame_bgr=red_frame)
    original_id = first[0].track_id

    # Gone long enough to exceed both max_age and reid_gallery_ttl.
    for _ in range(5):
        tracker.update([], frame_bgr=red_frame)

    reappeared = tracker.update([_pose(_FULL_BBOX)], frame_bgr=red_frame)

    assert reappeared[0].track_id != original_id


def test_without_embedder_appearance_is_ignored_past_max_age():
    tracker = Tracker(max_age=2)  # no embedder -- default behavior
    red_frame = _solid_frame((0, 0, 255))

    first = tracker.update([_pose(_FULL_BBOX)], frame_bgr=red_frame)
    original_id = first[0].track_id

    tracker.update([], frame_bgr=red_frame)
    tracker.update([], frame_bgr=red_frame)
    tracker.update([], frame_bgr=red_frame)

    reappeared = tracker.update([_pose(_FULL_BBOX)], frame_bgr=red_frame)

    assert reappeared[0].track_id != original_id
