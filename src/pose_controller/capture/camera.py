from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from pose_controller.config import CaptureConfig

# Found live on the Q6A (docs/backlog.md): a plain integer device index
# left to OpenCV's own backend auto-selection intermittently picks
# GStreamer, whose pipeline can fail to start ("Internal data stream
# error") even though the exact same device opens reliably via the
# explicit V4L2 backend every time it was tried. `cv2.CAP_ANY` (the
# default OpenCV would auto-select anyway) is kept for non-Linux
# platforms and for GStreamer pipeline strings -- a `source` string is
# itself a documented way to request a specific pipeline/backend, so it
# must not be overridden here.
_OPEN_ATTEMPTS = 2
_OPEN_RETRY_DELAY_S = 0.5


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
        backend = (
            cv2.CAP_V4L2 if sys.platform.startswith("linux") and isinstance(source, int) else cv2.CAP_ANY
        )

        self._cap: cv2.VideoCapture | None = None
        for attempt in range(_OPEN_ATTEMPTS):
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                self._cap = cap
                break
            cap.release()
            if attempt + 1 < _OPEN_ATTEMPTS:
                time.sleep(_OPEN_RETRY_DELAY_S)
        if self._cap is None:
            raise RuntimeError(f"Could not open capture source: {config.source!r}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._cap.set(cv2.CAP_PROP_FPS, config.fps)

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
