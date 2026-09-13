from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from pose_controller.capture.camera import Camera
from pose_controller.config import CaptureConfig


def _mock_cap(opened: bool = True, frame: np.ndarray | None = None) -> MagicMock:
    cap = MagicMock()
    cap.isOpened.return_value = opened
    cap.read.return_value = (True, frame if frame is not None else np.zeros((4, 4, 3), dtype=np.uint8))
    return cap


def test_uses_v4l2_backend_for_an_integer_device_index_on_linux():
    with patch("sys.platform", "linux"):
        with patch("cv2.VideoCapture", return_value=_mock_cap()) as mock_ctor:
            Camera(CaptureConfig(source="0"))

    args = mock_ctor.call_args[0]
    assert args == (0, cv2.CAP_V4L2)


def test_uses_default_backend_for_a_non_integer_source_even_on_linux():
    # A GStreamer pipeline string is itself a way to pick a backend --
    # must not be forced to V4L2.
    with patch("sys.platform", "linux"):
        with patch("cv2.VideoCapture", return_value=_mock_cap()) as mock_ctor:
            Camera(CaptureConfig(source="videotestsrc ! appsink"))

    args = mock_ctor.call_args[0]
    assert args == ("videotestsrc ! appsink", cv2.CAP_ANY)


def test_uses_default_backend_for_an_integer_device_index_off_linux():
    with patch("sys.platform", "win32"):
        with patch("cv2.VideoCapture", return_value=_mock_cap()) as mock_ctor:
            Camera(CaptureConfig(source="0"))

    args = mock_ctor.call_args[0]
    assert args == (0, cv2.CAP_ANY)


def test_retries_once_after_a_failed_open_before_succeeding():
    failed_cap = _mock_cap(opened=False)
    succeeded_cap = _mock_cap(opened=True)
    with patch("cv2.VideoCapture", side_effect=[failed_cap, succeeded_cap]):
        with patch("time.sleep") as mock_sleep:
            camera = Camera(CaptureConfig(source="0"))

    assert camera._cap is succeeded_cap
    failed_cap.release.assert_called_once()
    mock_sleep.assert_called_once()


def test_raises_runtime_error_if_every_open_attempt_fails():
    with patch("cv2.VideoCapture", return_value=_mock_cap(opened=False)):
        with patch("time.sleep"):
            try:
                Camera(CaptureConfig(source="0"))
                assert False, "expected a RuntimeError"
            except RuntimeError as exc:
                assert "0" in str(exc)


def test_read_returns_the_frame_from_the_underlying_capture():
    frame = np.ones((2, 2, 3), dtype=np.uint8)
    with patch("cv2.VideoCapture", return_value=_mock_cap(frame=frame)):
        camera = Camera(CaptureConfig(source="0"))

    result = camera.read()

    assert np.array_equal(result, frame)


def test_read_raises_if_the_underlying_capture_fails():
    cap = _mock_cap()
    cap.read.return_value = (False, None)
    with patch("cv2.VideoCapture", return_value=cap):
        camera = Camera(CaptureConfig(source="0"))

    try:
        camera.read()
        assert False, "expected a RuntimeError"
    except RuntimeError:
        pass


def test_context_manager_releases_the_capture():
    cap = _mock_cap()
    with patch("cv2.VideoCapture", return_value=cap):
        with Camera(CaptureConfig(source="0")):
            pass

    cap.release.assert_called_once()
