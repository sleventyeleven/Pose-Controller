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
  - `qnn.py` -- `QnnPoseEstimator`, on-device via a two-model pipeline on
    the Q6A's Hexagon NPU through `onnxruntime`'s QNN execution provider
    (`_qnn_runtime.OnnxQnnRunner`): a YOLOv8n person detector
    (`_yolo_detect.py`, 640x640 input) locates people, then BlazePose's
    landmark model (`_blazepose.py`) produces keypoints per detected box
    (`_blazepose.detect_poses_from_boxes`). BlazePose's own bundled
    detector (128x128 input) was replaced for localization specifically
    because it was found to lose small/distant subjects at ordinary room
    camera-to-subject distance regardless of image quality -- see
    `scripts/qnn_spike.md`. **Genuinely multi-person**: `_yolo_detect.py`
    runs real NMS over the detector's per-anchor output (not just top-1)
    and the landmark model runs once per surviving person box, up to
    `MAX_PERSONS`. Validated on physical hardware against a known-good
    reference photo (both people detected and landmarked with high
    confidence) and against real captured frames -- still an active
    tuning area for landmark confidence on some real-world crops, see
    `docs/backlog.md`.
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
- **gestures/** -- `normalize.normalize_pose` converts raw keypoints into
  shoulder-width-scaled, shoulder-relative coordinates so gesture
  thresholds work regardless of distance from camera or position in
  frame. `static_poses.classify_arm` turns that into `ArmPose.{DOWN,
  RAISED, OUT_TO_SIDE}` per arm -- "outward" is defined relative to each
  shoulder's own offset from torso center (not a hardcoded image
  direction), so it's correct regardless of which side of the image a
  given arm appears on; MediaPipe's landmark schema already labels
  LEFT_*/RIGHT_* by the subject's own anatomical left/right (not
  mirrored), so no separate handedness-correction step is needed on top
  -- see `normalize.py`'s module docstring for the reasoning and the
  escape hatch if real-camera testing ever shows this assumption wrong.
  `dynamic_gestures.SweepDetector` tracks one wrist's rolling-window
  trajectory to detect a one-armed overhead sweep;
  `dynamic_gestures.VerticalSwipeDetector` does the same for a directional
  vertical swipe in front of the body (up or down), used for volume.
  `state_machine.GestureStateMachine` ties it together per track_id:
  debounces static poses (`DEBOUNCE_FRAMES` consecutive frames, tolerating
  momentary occlusion) and edge-triggers `ControlAction`s (right-out ->
  NEXT, left-out -> PREVIOUS, both-raised -> PLAY_PAUSE, overhead sweep ->
  SKIP, right swipe up -> VOLUME_UP, left swipe down -> VOLUME_DOWN).
  Validated against real detected keypoints (not just synthetic test
  data) and extensively unit-tested; **also validated end-to-end against
  real recorded footage** running the full `estimate -> track -> gesture`
  loop over every frame -- all four originally-required actions (NEXT,
  PREVIOUS, PLAY_PAUSE, SKIP) triggered correctly from a real person's
  arm movements, though
  reliability is still limited by detection-hit-rate/tracking-continuity
  on harder footage, not the gesture logic itself -- see
  `docs/backlog.md`.
- **control/** (milestone 6, not yet built) -- `MediaController`
  abstraction (play/pause/next/previous/volume) that would actually carry
  out a `gestures.ControlAction`. First backend emulates OS media keys /
  Linux D-Bus MPRIS against whatever player is already running and
  authenticated, which decouples the offline vision pipeline from
  Spotify's own auth/streaming requirements. For now, `app.py` just
  prints triggered actions and shows them in the overlay banner.
- **overlay/** -- `draw_poses` renders skeleton + track-ID + each
  person's current confirmed arm states (e.g. "L:- R:OUT"), color-keyed
  by `track_id`; `draw_action_banner` shows the most recently triggered
  action for a few seconds. Together these are the causality
  requirement: a viewer sees an arm state building up *before* the
  action it triggers, not just the action appearing with no visible
  cause.
- **app.py + config.py** -- orchestrates the pipeline loop (capture ->
  `PoseEstimator.estimate` -> `Tracker.update` -> `GestureStateMachine.
  update_all` -> `draw_poses` + `draw_action_banner`). `AppConfig` loads
  from YAML (`configs/dev_laptop.yaml` vs `configs/dragon_q6a.yaml`), so
  switching hardware is a config change, not a code change.

Current status: milestones 1-4 are in place (scaffolding, capture + pose
baseline, multi-person detection + tracking with re-ID, gesture
recognition). Media control (actually driving Spotify/a player) is not
yet implemented -- gestures currently only print/display what they'd
trigger. See `scripts/qnn_spike.md` and `docs/backlog.md` for the
detailed trail of what it took to get the QNN backend working and fast.
