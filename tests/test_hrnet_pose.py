import numpy as np

from pose_controller.inference.backends import _hrnet_pose
from pose_controller.inference.pose import Landmark

NUM_KEYPOINTS = 17


def test_decode_heatmaps_finds_peak_location_and_scales_by_stride():
    heatmaps = np.zeros((*_hrnet_pose.HEATMAP_SIZE, NUM_KEYPOINTS), dtype=np.float32)
    heatmaps[10, 20, 0] = 0.8  # (row=y, col=x) for keypoint 0

    xy_conf = _hrnet_pose._decode_heatmaps(heatmaps)

    assert np.isclose(xy_conf[0, 0], 20 * _hrnet_pose.HEATMAP_STRIDE)  # x
    assert np.isclose(xy_conf[0, 1], 10 * _hrnet_pose.HEATMAP_STRIDE)  # y
    assert np.isclose(xy_conf[0, 2], 0.8)


def test_decode_heatmaps_clamps_confidence_to_unit_range():
    heatmaps = np.zeros((*_hrnet_pose.HEATMAP_SIZE, NUM_KEYPOINTS), dtype=np.float32)
    heatmaps[0, 0, 0] = 1.5  # above 1.0
    heatmaps[:, :, 1] = -0.3  # every cell (including the "peak") below 0.0

    xy_conf = _hrnet_pose._decode_heatmaps(heatmaps)

    assert np.isclose(xy_conf[0, 2], 1.0)
    assert np.isclose(xy_conf[1, 2], 0.0)


def test_detect_pose_from_box_maps_a_centered_peak_to_the_box_center():
    # A box centered on the frame, small enough that _roi_corners_from_box's
    # BOX_ROI_SCALE margin doesn't get clipped to the frame bounds -- so the
    # ROI (and thus the crop) is exactly centered on the frame too. A
    # heatmap peak at the exact center of the heatmap grid must then land
    # back at the frame's own center, regardless of the crop's scale/
    # aspect-ratio math, since an (unrotated) affine crop always maps its
    # own center to the ROI's center.
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    box = np.array([70.0, 70.0, 130.0, 130.0], dtype=np.float32)  # centered at (100, 100)

    heatmap_center = (_hrnet_pose.HEATMAP_SIZE[0] // 2, _hrnet_pose.HEATMAP_SIZE[1] // 2)

    def landmark_infer(_input_nhwc):
        heatmaps = np.zeros((*_hrnet_pose.HEATMAP_SIZE, NUM_KEYPOINTS), dtype=np.float32)
        heatmaps[heatmap_center[0], heatmap_center[1], :] = 0.9
        return heatmaps

    frame_rgb = frame  # already "RGB" for this synthetic all-zero frame
    pose = _hrnet_pose.detect_pose_from_box(frame_rgb, box, landmark_infer)

    assert pose is not None
    left_shoulder = pose.get(Landmark.LEFT_SHOULDER)
    assert np.isclose(left_shoulder.x, 0.5, atol=0.02)
    assert np.isclose(left_shoulder.y, 0.5, atol=0.02)
    assert np.isclose(left_shoulder.visibility, 0.9)


def test_detect_pose_from_box_maps_coco_keypoints_onto_landmark_enum():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    box = np.array([70.0, 70.0, 130.0, 130.0], dtype=np.float32)
    heatmap_center = (_hrnet_pose.HEATMAP_SIZE[0] // 2, _hrnet_pose.HEATMAP_SIZE[1] // 2)

    def landmark_infer(_input_nhwc):
        heatmaps = np.zeros((*_hrnet_pose.HEATMAP_SIZE, NUM_KEYPOINTS), dtype=np.float32)
        heatmaps[heatmap_center[0], heatmap_center[1], 9] = 0.8  # left_wrist
        heatmaps[heatmap_center[0], heatmap_center[1], 10] = 0.7  # right_wrist
        return heatmaps

    pose = _hrnet_pose.detect_pose_from_box(frame, box, landmark_infer)

    assert pose is not None
    left_wrist = pose.get(Landmark.LEFT_WRIST)
    right_wrist = pose.get(Landmark.RIGHT_WRIST)
    assert left_wrist is not None and right_wrist is not None
    assert np.isclose(left_wrist.visibility, 0.8)
    assert np.isclose(right_wrist.visibility, 0.7)


def test_detect_poses_from_boxes_returns_one_result_per_box():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes = [
        np.array([10.0, 10.0, 60.0, 60.0], dtype=np.float32),
        np.array([120.0, 120.0, 170.0, 170.0], dtype=np.float32),
    ]

    def landmark_infer(_input_nhwc):
        return np.full((*_hrnet_pose.HEATMAP_SIZE, NUM_KEYPOINTS), 0.5, dtype=np.float32)

    results = _hrnet_pose.detect_poses_from_boxes(frame, boxes, landmark_infer)

    assert len(results) == 2
