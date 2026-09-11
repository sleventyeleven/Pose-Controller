from __future__ import annotations

import cv2
import numpy as np

from pose_controller.config import CaptureConfig


class Camera:
    """Thin OpenCV VideoCapture wrapper.

    `source` accepts anything cv2.VideoCapture does: an integer device index
    (as a string, e.g. "0") for a UVC webcam, a /dev/videoN path, or a
    V4L2/GStreamer pipeline string. This is what lets the same capture code
    run against a laptop webcam and the Nexigo on the Q6A.
    """

    def __init__(self, config: CaptureConfig):
        self._config = config
        source: int | str = int(config.source) if config.source.isdigit() else config.source
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._cap.set(cv2.CAP_PROP_FPS, config.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open capture source: {config.source!r}")

    def read(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from capture source")
        return frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
