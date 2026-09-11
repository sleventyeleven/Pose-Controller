# Architecture

```
Camera capture -> Person detection -> Per-person pose estimation -> Tracker (+short-term re-ID)
      -> Per-track gesture/pose-state machine -> Control action mapping -> Media controller
      -> Overlay renderer (draws over every stage's output for causality)
```

The pipeline is layered so most of it can be written and iterated on a
regular dev laptop, and only the inference backend changes when deployed to
the Q6A:

- **capture/** -- `Camera`, an OpenCV `VideoCapture` wrapper. `source` is
  backend-agnostic: a device index for a UVC webcam (laptop or the Nexigo on
  the Q6A), a `/dev/videoN` path, or a V4L2/GStreamer pipeline string.
- **inference/** -- `PoseEstimator` is the abstract interface. Two
  implementations live in `inference/backends/`:
  - `cpu.py` -- `MediaPipePoseEstimator`, single-person, CPU-only, used for
    dev-loop iteration off-device.
  - `qnn.py` -- `QnnPoseEstimator`, on-device via Qualcomm AI Hub
    Models/QNN on the Q6A's Hexagon NPU. Currently a stub pending the NPU
    spike (see `docs/hardware.md`).
  - Multi-person support (milestone 3) adds a `Detector` interface alongside
    `PoseEstimator`: a lightweight person detector produces boxes, and the
    existing per-person `PoseEstimator` runs on each crop (top-down
    multi-person pose, not a bottom-up multi-person model). This keeps
    detection and pose estimation independently swappable and gives clean
    per-entity crops for re-ID.
- **tracking/** (milestone 3) -- SORT-style Kalman+Hungarian tracker for
  frame-to-frame identity, plus a small appearance-embedding re-ID matched
  against a short-term gallery so an entity's ID survives brief
  occlusion/exit-reentry.
- **gestures/** (milestone 4) -- per-track pose normalization (scale by
  shoulder width, center on torso, correct handedness so left/right is
  anatomical to the person rather than mirrored screen-left/right), a
  static-pose classifier (arm up/out via joint angles, debounced), and a
  dynamic-gesture recognizer for the overhead sweep via wrist trajectory.
- **control/** (milestone 6) -- `MediaController` abstraction (play/pause/
  next/previous/volume). First backend emulates OS media keys / Linux D-Bus
  MPRIS against whatever player is already running and authenticated, which
  decouples the offline vision pipeline from Spotify's own auth/streaming
  requirements.
- **overlay/** -- `draw_pose` and friends render skeleton, per-entity ID,
  pose-state, in-progress gesture, and last-triggered-action directly on the
  frame, so causality is visible without a separate dashboard/server.
- **app.py + config.py** -- orchestrates the pipeline loop. `AppConfig`
  loads from YAML (`configs/dev_laptop.yaml` vs `configs/dragon_q6a.yaml`),
  so switching hardware is a config change, not a code change.

See the plan history / commit messages for the full milestone breakdown;
current status: milestone 1 (scaffolding) + the start of milestone 2
(capture + single-pose baseline) are in place. Multi-person tracking,
gestures, overlay causality UI, and media control are not yet implemented.
