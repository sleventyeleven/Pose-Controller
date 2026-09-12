from __future__ import annotations

import numpy as np


def _xyxy_to_cxcywh(box: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dtype=float)


def _cxcywh_to_xyxy(state: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, w, h = state[:4]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class BoxKalmanFilter:
    """Constant-velocity Kalman filter over a bounding box.

    State is [cx, cy, w, h, vx, vy, vw, vh] (box center, width, height, and
    their per-frame velocities); only [cx, cy, w, h] is observed. This is
    the standard SORT approach (Bewley et al.) of tracking box geometry
    directly rather than image content, adapted to predict width/height
    velocity too (plain SORT holds aspect ratio fixed) since a person's
    apparent box size changes plausibly smoothly as they move toward/away
    from the camera.
    """

    _STATE_DIM = 8
    _MEASUREMENT_DIM = 4

    def __init__(
        self,
        initial_box_xyxy: tuple[float, float, float, float],
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
        initial_uncertainty: float = 10.0,
    ):
        self.state = np.zeros(self._STATE_DIM, dtype=float)
        self.state[:4] = _xyxy_to_cxcywh(initial_box_xyxy)

        self._F = np.eye(self._STATE_DIM)
        for i in range(4):
            self._F[i, i + 4] = 1.0  # position += velocity per step

        self._H = np.zeros((self._MEASUREMENT_DIM, self._STATE_DIM))
        for i in range(4):
            self._H[i, i] = 1.0

        self.covariance = np.eye(self._STATE_DIM) * initial_uncertainty
        self._process_noise = np.eye(self._STATE_DIM) * process_noise
        self._measurement_noise = np.eye(self._MEASUREMENT_DIM) * measurement_noise

    def predict(self) -> tuple[float, float, float, float]:
        """Advance the state by one frame and return the predicted box.
        Call exactly once per frame, before any `update()` for that frame."""
        self.state = self._F @ self.state
        self.covariance = self._F @ self.covariance @ self._F.T + self._process_noise
        return self.box

    def update(self, observed_box_xyxy: tuple[float, float, float, float]) -> None:
        """Incorporate a real detection matched to this track this frame."""
        z = _xyxy_to_cxcywh(observed_box_xyxy)
        residual = z - self._H @ self.state
        innovation_cov = self._H @ self.covariance @ self._H.T + self._measurement_noise
        kalman_gain = self.covariance @ self._H.T @ np.linalg.inv(innovation_cov)

        self.state = self.state + kalman_gain @ residual
        self.covariance = (
            np.eye(self._STATE_DIM) - kalman_gain @ self._H
        ) @ self.covariance

    @property
    def box(self) -> tuple[float, float, float, float]:
        return _cxcywh_to_xyxy(self.state)
