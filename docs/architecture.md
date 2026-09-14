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
  - `yolo26_qnn.py` -- `Yolo26PoseEstimator`, an *alternate* on-device
    pipeline being evaluated alongside `QnnPoseEstimator`, not replacing
    it (`inference.backend: yolo26` vs `qnn` -- see
    `configs/dragon_q6a_yolo26.yaml`). A single end-to-end model
    (YOLO26-Pose) does person detection *and* 17-keypoint estimation in
    one forward pass, instead of two separate models with a manual
    crop/ROI step in between. Built to test whether that eliminates two
    specific, already-diagnosed weak points in the two-stage pipeline:
    the crop-margin step's own tuning fragility, and BlazePose's
    landmark model gating its entire output behind one scalar confidence
    (a real candidate for the "only registers gestures facing the
    camera" symptom found in live testing). `_yolo26_pose.py` maps
    COCO's 17-keypoint schema onto this project's `Landmark` enum, so
    `Tracker`, `GestureStateMachine`, and `overlay/` all work completely
    unchanged regardless of which pipeline is selected. The compiled
    model comes from Ultralytics' own local QNN export, not Qualcomm AI
    Hub (whose cloud compiler failed on this model -- see
    `docs/backlog.md`); its raw, non-post-processed output means
    `_yolo26_pose.py` also carries a `decode_raw_output` step (box-format
    conversion + channel splitting) ahead of the same score-filter+NMS
    logic used for the AI-Hub-style contract. Confirmed running on a
    real Hexagon v68 NPU (Q8B); not yet run against real camera footage
    or on the Q6A directly -- see `docs/backlog.md` for status.
  - `hrnet_qnn.py` -- `HrnetPoseEstimator`, a second *alternate* landmark
    stage (`inference.backend: hrnet` -- see
    `configs/dragon_q6a_hrnet.yaml`), reusing the same YOLOv8n-det first
    stage as `QnnPoseEstimator` but replacing BlazePose's landmark model
    with HRNetPose. Unlike YOLO26-Pose, this stays a two-stage design --
    evaluated specifically because it uses the same `w8a8` quantization
    path proven for every other model here, where YOLO26 (`w8a16`) hit a
    reproducible QNN compile failure on this chipset (see
    `docs/backlog.md`). `_hrnet_pose.py` decodes its heatmap output
    (64x48x17, one confidence map per COCO keypoint) via per-keypoint
    argmax and reuses `_yolo26_pose.py`'s COCO->`Landmark` mapping, so
    again every downstream consumer is unchanged. Compiled and confirmed
    running on both the Q6A's and Q8B's real NPUs; not yet run against
    real footage -- see `docs/backlog.md` for status. (See
    `docs/pipelines.md` for a flow diagram of each of these three
    pipelines -- `qnn`/`hrnet`/`yolo26` -- and every measured performance
    number, on every board tested, in one place.)
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
- **control/** (milestone 6, built 2026-09-12, **validated end-to-end on
  real hardware 2026-09-13/14**) -- `MediaController` abstraction
  (play/pause/next/previous/volume) that actually carries out a
  `gestures.ControlAction`, via `dispatch()`. The `playerctl` backend
  (`control/backends/playerctl.py`) emulates OS media keys / Linux D-Bus
  MPRIS against whatever player is already running and authenticated,
  which decouples the offline vision pipeline from Spotify's own
  auth/streaming requirements -- no Spotify-specific code exists
  anywhere in this module. Off by default (`control.enabled`); degrades
  to a no-op `NullMediaController` if `playerctl` isn't installed,
  matching `build_tracker`'s `ReidEmbedder` fallback pattern. `app.py`
  still prints triggered actions and shows them in the overlay banner
  regardless of whether control is enabled -- that causality display was
  never conditional on this. Since the official Spotify Linux client has
  no ARM64 build, real validation went through `spotifyd` (a
  Connect-compatible daemon) on the Q8B instead -- confirmed real
  gestures (`PLAY_PAUSE`/`SKIP`/`NEXT`) driving a real, playing Spotify
  track via `playerctl`, with zero code changes needed since `control/`
  never knew or cared which MPRIS player it was talking to. See
  `docs/backlog.md` for the full setup trail.
- **overlay/** -- `draw_poses` renders a bounding box, full-body skeleton
  (COCO's own skeleton -- face, arms, torso, legs), track-ID, and each
  person's current confirmed arm states (e.g. "L:- R:OUT"), color-keyed
  by `track_id`. The box costs nothing extra to draw (it's `pose.bbox`,
  already computed for tracking); how much of the skeleton actually
  renders depends on which `Landmark`s the active backend fills in --
  `yolo26`/`hrnet` cover COCO's full 17 points for free (same single
  inference call, this project's own decode just wasn't using all of it
  before), while the `qnn` backend's compiled BlazePose model has no leg
  landmarks at all (`inference/pose.py`'s `Landmark` docstring). Real
  keypoint-mapping and drawing added 2026-09-12 after the first live
  dashboard test -- see `docs/backlog.md`. `draw_action_banner` shows the
  most recently triggered action for a few seconds. Together these are
  the causality requirement: a viewer sees an arm state building up
  *before* the action it triggers, not just the action appearing with no
  visible
  cause.
- **web/** (initial draft) -- `DashboardState` + `DashboardServer`: a
  browser-viewable page showing the live overlay frame (MJPEG stream),
  a scrolling gesture-trigger queue, and (when `control.enabled`) a
  media-control status panel (connected/paused/playing, now-playing
  title/artist) polled from `/media`, on a background thread alongside
  the main capture loop. Standard-library `http.server` only, no new
  dependency. Off by default (`OverlayConfig.web_enabled`) -- for demos,
  debugging, and pipeline iteration without a display attached or a
  manual capture/scp/inspect round-trip; see `docs/backlog.md` for scope
  and what's not yet handled (slow-viewer backpressure, multi-person
  scoping, on-device validation).
- **app.py + config.py** -- orchestrates the pipeline loop (capture ->
  `PoseEstimator.estimate` -> `Tracker.update` -> `GestureStateMachine.
  update_all` -> `MediaController.dispatch` + `draw_poses` +
  `draw_action_banner` + `DashboardState.update_frame`/`add_event`/
  `update_media_status` when the dashboard is enabled). `AppConfig` loads
  from YAML (`configs/dev_laptop.yaml` vs `configs/dragon_q6a.yaml`), so
  switching hardware is a config change, not a code change.

Current status: milestones 1-6 are in place and validated on real
hardware (scaffolding, capture + pose baseline, multi-person detection +
tracking with re-ID, gesture recognition, and media control actually
driving a real Spotify Connect session end to end). See
`scripts/qnn_spike.md` and `docs/backlog.md` for the detailed trail of
what it took to get the QNN backend working and fast.
