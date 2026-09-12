# Architecture

```
Camera capture -> Multi-person pose estimation -> Tracker (+appearance re-ID)
      -> Per-track gesture/pose-state machine -> Control action mapping -> Media controller
      -> Overlay renderer (draws over every stage's output for causality)
```

The pipeline is layered so most of it can be written and iterated on a
regular dev laptop, and only the inference backend changes when deployed to
the Q6A:

- **capture/** -- `Camera`, an OpenCV `VideoCapture` wrapper. `source` is
  backend-agnostic: a device index for a UVC webcam (laptop or the Nexigo on
  the Q6A), a `/dev/videoN` path, or a V4L2/GStreamer pipeline string.
- **inference/** -- `PoseEstimator.estimate(frame) -> list[PoseResult]` is
  the abstract interface (one `PoseResult` per detected person; empty list
  if none). `PoseResult.bbox` is derived from keypoint visibility
  (`bbox_from_keypoints`) rather than provided by the detector, so it means
  the same thing regardless of backend -- this is what `tracking.Tracker`
  matches on. Two implementations live in `inference/backends/`:
  - `cpu.py` -- `MediaPipePoseEstimator`, CPU-only dev-loop backend using
    MediaPipe's legacy `solutions.pose` API. **Single-person only** (that
    API's own limitation) -- not yet upgraded to MediaPipe's newer
    multi-person Tasks API since the real multi-person target is the QNN
    backend; see `docs/backlog.md`.
  - `qnn.py` -- `QnnPoseEstimator`, on-device via a BlazePose
    detector+landmark model pair running on the Q6A's Hexagon NPU through
    `onnxruntime`'s QNN execution provider (`_qnn_runtime.OnnxQnnRunner`).
    **Genuinely multi-person**: `_blazepose.py` runs real greedy NMS over
    the detector's anchor-based output (not just top-1) and evaluates the
    landmark model once per surviving detection, up to `MAX_PERSONS`.
    Validated on physical hardware with 2 simultaneous people at
    ~15-20ms/frame total -- see `scripts/qnn_spike.md`.
- **tracking/** -- `Tracker.update(detections, frame_bgr) -> detections`
  (with `track_id` populated). Two layers:
  - Position/motion (always on): a constant-velocity Kalman filter per
    track (`kalman.BoxKalmanFilter`), matched to each frame's detections
    by IoU via the Hungarian algorithm
    (`scipy.optimize.linear_sum_assignment`). Unmatched tracks coast on
    their Kalman prediction for `max_age` frames before being dropped, so
    an ID survives brief occlusion.
  - Appearance (optional, via `reid.ReidEmbedder`): when a track ages out
    past `max_age`, its last-seen appearance embedding (OSNet-x0.25, run
    in plain FP32 on CPU -- see `models/osnet_x0_25/README.md`) moves to
    a "lost gallery" for `reid_gallery_ttl` further frames. A new
    detection that doesn't IoU-match any active track is checked against
    that gallery by cosine similarity before a fresh ID is allocated --
    this is what lets an ID survive fully leaving and re-entering frame,
    not just brief occlusion. Validated on physical hardware: a person
    revived with their original ID after "disappearing" and reappearing
    at a completely different position. See `scripts/qnn_spike.md`.
- **gestures/** (milestone 4, not yet built) -- per-track pose
  normalization (scale by shoulder width, center on torso, correct
  handedness so left/right is anatomical to the person rather than
  mirrored screen-left/right), a static-pose classifier (arm up/out via
  joint angles, debounced), and a dynamic-gesture recognizer for the
  overhead sweep via wrist trajectory.
- **control/** (milestone 6, not yet built) -- `MediaController`
  abstraction (play/pause/next/previous/volume). First backend emulates OS
  media keys / Linux D-Bus MPRIS against whatever player is already
  running and authenticated, which decouples the offline vision pipeline
  from Spotify's own auth/streaming requirements.
- **overlay/** -- `draw_poses` renders skeleton + track-ID label per
  person, color-keyed by `track_id` so identity is visible at a glance
  across multiple people. Pose-state/in-progress-gesture/last-action
  overlays arrive with milestone 4's gesture logic.
- **app.py + config.py** -- orchestrates the pipeline loop (capture ->
  `PoseEstimator.estimate` -> `Tracker.update` -> `draw_poses`).
  `AppConfig` loads from YAML (`configs/dev_laptop.yaml` vs
  `configs/dragon_q6a.yaml`), so switching hardware is a config change,
  not a code change.

Current status: milestones 1-3 are in place (scaffolding, capture +
pose baseline, multi-person detection + tracking). Gestures, overlay
causality UI beyond basic skeleton+ID, and media control are not yet
implemented. See `scripts/qnn_spike.md` and `docs/backlog.md` for the
detailed trail of what it took to get the QNN backend working and fast.
