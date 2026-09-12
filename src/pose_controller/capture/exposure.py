"""Frame exposure/lighting quality checks.

Discovered empirically (2026-09-11/12) while testing gesture recognition
with a live camera on the physical Q6A: the pose detector went from 0.99
confidence on well-lit reference photos to a flat, degenerate 0.5 (the
same "garbage input" signature seen elsewhere in this project when
something is structurally wrong, not just "a hard image") on real webcam
captures -- both a too-dark raw capture *and* a subsequent overexposed
one after cranking brightness/contrast in `cheese` to compensate. The
overexposed case was confirmed as real sensor clipping (8.2% of pixels
blown to near-white) combined with near-total desaturation (HSV
saturation channel mean of 4.35 out of 255, vs 70.7 on a known-good
reference photo) -- a genuinely different image regime than anything the
detector was validated against, not a bug in this project's code. See
scripts/qnn_spike.md for the full investigation.

This module doesn't fix exposure (that needs driving the camera's own
V4L2 controls -- cv2.CAP_PROP_BRIGHTNESS/EXPOSURE/GAIN/SATURATION --
which needs iterative tuning against real hardware; see docs/backlog.md),
it only detects and reports likely-bad conditions so a caller (app.py)
can warn rather than silently produce unreliable gesture detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Calibrated against real measurements, not guessed: a known-good
# reference photo had mean_brightness=96.0, mean_saturation=70.7,
# clipped_high=0.0%; a real too-dark webcam capture broke down somewhere
# around mean_brightness 19-28 (found via synthetic darkening of the
# reference image); a real overexposed webcam capture had
# mean_brightness=180, mean_saturation=4.35, clipped_high=8.2%. These
# thresholds sit with margin on the "bad" side of that known-good value.
TOO_DARK_BRIGHTNESS = 25.0
TOO_BRIGHT_BRIGHTNESS = 200.0
CLIPPED_HIGHLIGHT_FRACTION = 0.05
LOW_SATURATION = 20.0


@dataclass
class ExposureReport:
    mean_brightness: float
    mean_saturation: float
    clipped_high_fraction: float  # fraction of pixels with near-white grayscale value
    clipped_low_fraction: float  # fraction of pixels with near-black grayscale value
    issues: list[str] = field(default_factory=list)

    @property
    def is_good(self) -> bool:
        return not self.issues


def assess_exposure(frame_bgr: np.ndarray) -> ExposureReport:
    """Cheap, synchronous per-frame check -- just grayscale/HSV stats, no
    model inference. Cheap enough to run every frame without a meaningful
    performance cost."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]

    mean_brightness = float(gray.mean())
    mean_saturation = float(saturation.mean())
    clipped_high = float(np.mean(gray >= 250))
    clipped_low = float(np.mean(gray <= 5))

    issues = []
    if mean_brightness < TOO_DARK_BRIGHTNESS:
        issues.append("too dark")
    if mean_brightness > TOO_BRIGHT_BRIGHTNESS or clipped_high > CLIPPED_HIGHLIGHT_FRACTION:
        issues.append("overexposed (blown highlights)")
    if mean_saturation < LOW_SATURATION:
        issues.append("washed out / desaturated")

    return ExposureReport(
        mean_brightness=mean_brightness,
        mean_saturation=mean_saturation,
        clipped_high_fraction=clipped_high,
        clipped_low_fraction=clipped_low,
        issues=issues,
    )
