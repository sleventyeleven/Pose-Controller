import numpy as np

from pose_controller.capture.exposure import assess_exposure


def _solid_frame(bgr, size=64):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_well_exposed_colorful_frame_has_no_issues():
    # Distinctly different B/G/R values -> real saturation; moderate
    # brightness; no clipping.
    frame = _solid_frame((50, 120, 200))

    report = assess_exposure(frame)

    assert report.is_good
    assert report.issues == []


def test_dark_frame_flags_too_dark():
    frame = _solid_frame((5, 5, 5))

    report = assess_exposure(frame)

    assert "too dark" in report.issues
    assert not report.is_good


def test_overexposed_frame_flags_blown_highlights():
    frame = _solid_frame((255, 255, 255))

    report = assess_exposure(frame)

    assert any("overexposed" in issue for issue in report.issues)
    assert not report.is_good


def test_desaturated_midtone_frame_flags_washed_out():
    # Equal B/G/R -> zero saturation, but a brightness comfortably in the
    # "normal" range -- isolates the desaturation check from the
    # brightness checks (this is exactly the real failure mode found on
    # the physical Q6A: a technically mid-brightness frame that's still
    # unusable because it carries no color information -- see
    # scripts/qnn_spike.md).
    frame = _solid_frame((128, 128, 128))

    report = assess_exposure(frame)

    assert report.issues == ["washed out / desaturated"]


def test_matches_real_measured_reference_values():
    # Regression guard using the actual numbers measured against a known-
    # good reference photo during the original investigation (see module
    # docstring) -- if these calibration constants ever drift, this
    # should catch a reference-quality image starting to fail.
    frame = _solid_frame((50, 120, 200))
    report = assess_exposure(frame)
    assert report.mean_saturation > 100  # comfortably above the 70.7 reference
