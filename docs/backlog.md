# Backlog / deferred items

Things intentionally deferred out of the current milestone work, so they
don't get lost.

## Tooling

- **Web dashboard: live overlay feed + gesture queue (2026-09-12, not yet
  implemented).** A browser-viewable page showing the same overlay frame
  `app.py` already renders (skeleton, track IDs, arm states, action
  banner -- `overlay/renderer.py`) plus a scrolling log of triggered
  actions with timestamps (today these only go to stdout:
  `print(f"[gesture] #{track_id}: {action.name}")` in `app.py`). Purpose:
  demos (show causality to someone not standing at the board's HDMI
  output), debugging, and pipeline iteration -- this whole session's
  workflow for every finding (exposure diagnosis, orientation bug, ROI
  tuning, re-ID threshold) was capture-on-device -> `scp` down ->
  view/analyze -> iterate; a live view would remove that round-trip
  entirely for anything that doesn't need per-pixel inspection.
  - **Likely shape, given the existing architecture:** a lightweight HTTP
    server (Flask, or even `http.server` -- consistent with this
    project's preference for minimal dependencies, e.g. re-ID staying on
    plain CPU onnxruntime rather than pulling more NPU complexity in;
    see `models/osnet_x0_25/README.md`) run on a background thread
    alongside `app.py`'s main capture loop, since the loop is currently
    single-threaded around `cv2.imshow`/`cv2.waitKey`. Needs thread-safe
    shared state for "latest overlay frame" + "recent gesture events"
    (a lock around a plain variable/deque is enough at this scale, no
    need for anything heavier). An MJPEG stream (a well-worn, simple
    pattern: a generator yielding JPEG-encoded frames over one HTTP
    connection) is the lowest-effort way to get video into a browser
    with just an `<img>` tag; the gesture queue can be simple polling or
    Server-Sent Events off the same event deque.
  - Should be config-gated like `overlay.show_window` already is (e.g. a
    `web_enabled`/`web_port` pair in `OverlayConfig` or a small new
    `WebConfig`), not always-on -- running a webcam-connected board with
    an HTTP server bound by default isn't something to do silently.
  - **No auth planned or implied** -- fine for a LAN-local demo/debug
    tool the user starts deliberately, not something to expose beyond
    that without real thought given to who else could reach it.
  - Natural home: a new `web/` (or `dashboard/`) package alongside
    `capture/`, `inference/`, etc., following the project's existing
    per-concern module layout (`docs/architecture.md`).
- **Automated bring-up/install script.** `docs/setup-q6a.md` now has a
  confirmed-working manual recipe (r2 image + BIOS update + `apt install
  fastrpc fastrpc-test fastrpc-dev libcdsprpc1 radxa-firmware-qcs6490` +
  reboot + the camera overlay via `rsetup`). Turn it into an idempotent
  `scripts/setup_device.sh` that a fresh r2-imaged Q6A can run to reach a
  working state in one shot: the NPU package install + reboot, the camera
  overlay enable (still needs the exact non-interactive `rsetup
  enable_overlays <name>` invocation worked out -- see `docs/setup-q6a.md`
  section 3), the idle-suspend fix, Python venv creation, and project
  dependency install. Should detect/report cleanly rather than silently
  no-op if run on hardware/image it doesn't recognize (e.g. warn if not
  on r2+ or if BIOS is older than `20251230`).

## Performance

- ~~**QNN per-frame latency (~476ms, dominated by context-reload
  overhead).**~~ Resolved 2026-09-11: replaced the `qnn-net-run`-
  subprocess-per-call approach with `onnxruntime` + `onnxruntime-qnn`
  (`_qnn_runtime.OnnxQnnRunner`), loading each model into a persistent
  `InferenceSession` once and calling `.run()` per frame. Measured
  ~2.2ms (detector) / ~1.1ms (landmark) per warm call on the physical
  Q6A -- matches AI Hub's own NPU-only profiling, ~150x faster than the
  subprocess approach. Full `estimate()` call on a 3088x2316 test image:
  ~26ms. See `scripts/qnn_spike.md` for the two onnxruntime API details
  that weren't obvious (`register_execution_provider_library` +
  `add_provider_for_devices`, not the plain `providers=[...]` argument;
  and letting `onnxruntime_qnn` manage `ADSP_LIBRARY_PATH` itself rather
  than overriding it).
- **`onnxruntime-qnn` package size (~202MB) is mostly unused Hexagon skel
  libraries.** It bundles HTP skeleton/stub `.so` files for versions V68,
  V69, V73, V75, V79, and V81; the Q6A's QCS6490 only uses V68. Pruning
  the other 5 versions' files after confirming nothing else depends on
  them could recover a meaningful chunk of the `docs/storage-footprint.md`
  budget. Not done -- needs care to confirm onnxruntime doesn't probe for
  other versions at runtime before deleting anything.

## Tracking / multi-person (milestone 3 follow-ups)

- ~~**Appearance-based re-ID.**~~ Resolved 2026-09-11: `tracking.reid.
  ReidEmbedder` (OSNet-x0.25, plain FP32 on CPU -- see
  `models/osnet_x0_25/README.md` for why not NPU/quantized) wired into
  `Tracker` via a lost-track gallery. Validated on the physical Q6A: a
  track that fully left frame and reappeared at a different position was
  correctly revived by appearance (0.978 similarity for the same person,
  0.419 against background) -- see `scripts/qnn_spike.md`.
- **Re-ID similarity threshold (0.6) is a starting guess, not tuned.**
  Picked as a reasonable default, not validated against a labeled
  benchmark or real multi-person footage with genuinely different-looking
  people. Worth revisiting once real camera footage (not a synthetic
  same-person-twice test image) is available -- both false revivals
  (different person, same ID) and missed revivals (same person, new ID)
  are plausible failure modes at the wrong threshold.
- **Re-ID embedding cost scales with simultaneous new/lost detections.**
  ~20-50ms per `ReidEmbedder.embed()` call on the physical Q6A's CPU is
  fine for the common case (one new-or-lost person at a time), but several
  people all triggering appearance checks in the same frame would add up
  (e.g. 4 simultaneous new detections ≈ 100-200ms just for re-ID). Not
  a problem yet at this project's realistic scale (a handful of people),
  but worth knowing if a scenario with many simultaneous entries/exits
  comes up.
- **CPU dev backend is still single-person.** `mp.solutions.pose` (the
  legacy MediaPipe API `cpu.py` uses) doesn't support multi-person
  detection; MediaPipe's newer Tasks API
  (`mediapipe.tasks.python.vision.PoseLandmarker` with `num_poses`) does.
  Not migrated since the real multi-person target is the QNN backend
  (validated working, see `scripts/qnn_spike.md`) -- worth doing if
  multi-person logic needs iterating on a laptop without hardware access
  becomes a real friction point.
- **QNN landmark visibility output hovers near 0.5 for everything.**
  Noticed during the NPU spike: the landmark model's 4th output value
  (visibility) came back very close to sigmoid(0) = 0.5 for all 25 points
  in every test so far, rather than confidently separating visible from
  occluded points. The overlay's `_VISIBILITY_THRESHOLD = 0.5` sits right
  on that boundary, which risks a keypoint flickering on/off between
  frames as the value jitters across 0.5, rather than a real occlusion
  signal. Not investigated further -- may be how this particular
  BlazePose port's landmark model was trained/calibrated, or may need
  recalibrating the threshold empirically once real (non-static-photo)
  camera input is available to observe actual occlusion behavior.

## Gestures (milestone 4 follow-ups)

- **Root cause of the live-camera detection failures found (2026-09-12):
  the person is too small in frame, not a hardware/exposure problem.**
  Live-camera gesture testing on the physical Q6A went through three
  rounds before finding this. Round 1 (Nexigo webcam, dim/upward-angled)
  and round 2 (Nexigo webcam, `cheese`-boosted brightness/contrast) both
  failed pose detection with the same degenerate flat-0.5 "garbage input"
  signature seen during the original NHWC bug (see `scripts/qnn_spike.md`).
  Round 2's failure was fully diagnosed as real overexposure/desaturation
  (see `capture.exposure.assess_exposure` below); round 1's cause was not
  identified at the time -- darkness and blur were both ruled out.
  Round 3 switched to the Dragon Q6A's onboard CSI camera (Sony IMX415,
  enabled via a previously-disabled `rsetup`/device-tree overlay -- see
  `scripts/qnn_spike.md` for the full bring-up) and, after building a
  raw10-unpack + debayer + white-balance pipeline, produced a frame that
  was unambiguously well-exposed, sharp, and colorful, with a person
  clearly visible and an arm raised -- and it *still* failed detection
  with the identical flat-0.5 signature. That result ruled out exposure,
  color, sharpness, and camera hardware entirely, since this was a
  different sensor with none of the earlier defects and it failed anyway.
  Comparing the raw (pre-sigmoid) detector output against a known-good
  demo image was the key: the demo image had 4 anchors at a confident
  +4.83 logit; the failing real frame's best anchor never exceeded 0.0,
  even though the two arrays had visibly comparable image content.
  Cropping the failing frame tightly around the person and re-running it
  through the identical detector code produced a max score of 0.9921 with
  a raw logit of +4.833 -- matching the demo image's confident value
  almost exactly, and a full pose was returned. **The person detector
  (the first stage of the two-stage BlazePose pipeline) needs the subject
  to occupy a meaningfully large fraction of the frame; at ordinary room
  distance with either camera's field of view, a full-frame person is too
  small once downscaled to the fixed 128x128 detector input, regardless
  of image quality.** This is a framing/distance limitation of the chosen
  detector model against this use case's camera-to-subject distance, not
  a bug anywhere in this project's code.
  - **Net result:** gesture recognition is still not validated against a
    successful live full-frame detection, but the reason why is now fully
    understood, and the fix is known: either move the camera meaningfully
    closer to the subject, narrow the field of view (a longer-focal-length
    lens, or a fixed center-crop/digital zoom tuned to the expected
    interaction distance before feeding the detector), or have the
    subject stand closer to the camera than a typical "control the room's
    media from across the room" scenario implies. None of these have been
    implemented yet -- the crop test above was a diagnostic, not a
    real fix wired into `capture`/`inference`.
  - This also means `capture.exposure.assess_exposure`'s known gap (round
    1's failure wasn't caught by it) is now explained: that failure was
    very likely this same framing issue all along, not a distinct
    exposure-adjacent problem the checker needs to grow a rule for.
- **`capture.exposure.assess_exposure` -- detects overexposure/
  desaturation, not every bad-lighting failure mode.** Built in response
  to the finding above: a cheap, tested (`tests/test_exposure.py`),
  per-frame check for overexposure (brightness/clipping) and desaturation
  (washed-out color), wired into `app.py` as a console + on-screen
  warning. Confirmed it correctly flags the real overexposed `cheese`
  capture and correctly passes the known-good reference photo. Does
  *not* catch the framing/subject-too-small problem described above --
  that's an orthogonal failure mode this module was never meant to
  address (it's not an exposure or color problem, so a per-pixel
  brightness/saturation check has no signal to catch it on). Still
  useful for what it was built for, just not sufficient on its own to
  predict whether a frame will actually detect a person.
- **Auto-adjusting camera exposure, not just detecting bad exposure.**
  The user's original ask was "detect *and* adjust." Only detection is
  built. Adjustment would mean driving the camera's V4L2 controls
  programmatically (`cv2.VideoCapture.set` with
  `cv2.CAP_PROP_BRIGHTNESS`/`EXPOSURE`/`GAIN`/`SATURATION` for the Nexigo,
  or the sensor's `analogue_gain`/`exposure` v4l2 subdev controls for the
  CSI camera) in a feedback loop informed by `assess_exposure`'s output.
  Not attempted -- lower priority now than the framing fix above, since
  exposure was never the thing actually blocking detection in the end.
- **Framing fix implemented (2026-09-12): YOLOv8n-det replaces BlazePose's
  own detector for person localization.** `inference/backends/qnn.py`'s
  `QnnPoseEstimator` now runs a YOLOv8n person detector (640x640 input,
  compiled for QCS6490 -- see `models/yolov8n_det_qcs6490/README.md` and
  `scripts/qnn_spike.md`) as the first stage instead of BlazePose's own
  128x128 detector; BlazePose's landmark model is unchanged, just fed a
  box-derived crop (`_blazepose.detect_poses_from_boxes` /
  `_roi_corners_from_box`) instead of BlazePose's own aux-keypoint-derived
  one. Confirmed on the physical Q6A against the known-good two-person
  demo photo: both people detected and landmarked with high confidence
  (0.99 / 0.71), matching BlazePose's own detector's baseline on the same
  image.
  - **Correction to the finding above: orientation, not just subject size,
    was doing a lot of the damage.** Testing this new detector against the
    real 30-frame CSI capture sequence surfaced something the earlier
    diagnosis missed: the frames were being fed to the pipeline in the
    CSI sensor's native **portrait** orientation (a manual rotation script
    artifact, guessed wrong direction at the time and never corrected).
    Rotating to landscape took one frame's person-class detector score
    from 0.070 to **0.867** -- a far bigger swing than image quality alone
    explains. The earlier "not orientation" ruling (`scripts/qnn_spike.md`,
    round 1) tested aspect-ratio *squishing* on a demo photo, not genuine
    90-degree rotation on a real capture -- a different transformation
    that this session conflated with the real one. Orientation was likely
    a significant, previously-uncredited contributor to the original
    Nexigo failures too.
  - **Net result on the real capture sequence, correctly rotated:** the
    YOLO detector now finds a person box in 19 of 30 frames (up from 0/30
    with the old detector+orientation) -- a large, confirmed improvement.
    But the landmark model's confidence on most of those real crops is
    still poor (repeatedly landing near 0.09, well below
    `MIN_LANDMARK_SCORE`'s 0.5), with only 1 of 30 frames producing a full
    end-to-end pose. One contributing mechanism found and fixed: the
    box-derived ROI's margin (`BOX_ROI_SCALE`, tuned against the demo
    photo to 1.75) can extend past the frame's edges for a person who
    fills most of the vertical FOV, wasting crop area on black padding --
    `_roi_corners_from_box` now shifts (not resizes) the ROI to stay in
    bounds when geometrically possible. This helped one frame cross the
    threshold (0.44 -> 0.80) but most frames were unaffected, so it isn't
    the whole story. Suspected remaining factors, not yet isolated: the
    high `analogue_gain` (80/100) used during that capture session
    introducing real sensor noise the landmark model wasn't trained
    against, and/or genuine pose difficulty (the sequence was shot
    mid-motion, walking and gesturing, not standing still).
  - Still not implemented, and now lower priority given the above: moving
    the camera physically closer, or a fixed digital-zoom crop tuned to
    a specific expected interaction distance.
  - **Retested against the original overexposed `cheese` recording
    (round 2's Nexigo capture, `/home/radxa/Videos/Webcam/
    2026-09-12-024357.webm`, the exact video that first proved the
    overexposure/desaturation diagnosis) and the new detector handles it
    far better than expected.** Sampling ~1fps across the video's 34.7s
    (31 frames): `capture.exposure.assess_exposure` still flags nearly
    every frame (brightness ~170-180, saturation ~4-8 -- genuinely still
    overexposed and desaturated, nothing about the video itself changed),
    yet YOLO finds a person in **28 of 31 frames (90%)**, and the full
    detect+landmark pipeline succeeds end-to-end on **14 of 31 frames
    (45%)** -- up from 0 with the old detector on this same video. The
    new detector's higher input resolution apparently gives it enough
    headroom to work through desaturation that completely blinded the
    old 128x128 detector, independent of whatever `assess_exposure`
    measures. This is the strongest evidence yet that the YOLO swap
    meaningfully un-blocks live-camera gesture validation: real, already-
    recorded, previously-fatal footage now produces poses on nearly half
    its frames without any new capture session. The remaining ~55%
    misses are presumed to be the same landmark-confidence variability
    described above (motion/pose-dependent), not re-diagnosed separately
    for this video.
- **Gesture recognition validated end-to-end against real dynamic footage
  (2026-09-12) -- all four required actions triggered correctly.** Ran
  the full `PoseEstimator -> Tracker -> GestureStateMachine` pipeline
  (i.e. `app.py`'s actual loop, not a synthetic test) over every frame of
  the overexposed `cheese` recording above. Without re-ID
  (`Tracker(embedder=None)`, IoU/Kalman only): the person's track_id
  fragmented into **13 different IDs** across the 28.7s clip the person is
  in frame -- expected, since detection succeeds on only ~49% of frames
  (`frames_with_pose=451/923`) and misses come in bursts long enough to
  regularly exceed `max_age` (30 frames = 1s), so the Kalman coast
  couldn't hold the ID through most gaps. Even so, gesture logic still
  fired **5 real actions** (SKIP, NEXT x2, PREVIOUS, SKIP) from actual
  arm movements in the video -- the first live-camera-triggered gestures
  in this project's history. **Enabling re-ID**
  (`Tracker(embedder=ReidEmbedder(...))`, exactly the milestone-3 feature
  built for surviving exactly this kind of gap) collapsed those 13
  fragments down to **4** track_ids and increased triggers to **6**,
  now covering **all four** required actions for the first time on real
  footage: NEXT, PREVIOUS, SKIP, and PLAY_PAUSE (both arms raised).
  Re-ID isn't perfectly stable here either -- the surviving ID alternates
  between two values (#3/#4) rather than settling on one, suggesting
  `reid_similarity_threshold` (0.6) or `reid_gallery_ttl` could use
  tuning against this specific high-miss-rate regime. Performance:
  22.0fps with re-ID enabled vs 26.6fps without, on this same ~49%-hit-rate
  video -- re-ID's extra embedding cost shows up more than usual here
  because so many frames trigger a re-match attempt; a video with a
  higher underlying detection rate would pay this cost far less often.
  Static-pose thresholds and sweep parameters (`static_poses.py`,
  `dynamic_gestures.py`) are still the original reasoned starting
  points, not yet retuned against this footage, despite now having real
  gesture data to tune against -- the fact that all four actions fired
  with zero threshold changes is a good sign they're already in a
  reasonable range, not proof they're optimal.
- **`reid_similarity_threshold` retuned against this footage (2026-09-12):
  0.6 -> 0.5.** The track_id alternation above (#3/#4) had a concrete,
  measurable cause, not just bad luck: computed pairwise cosine
  similarity between every real detection of the one person in this
  video (861 embeddings, ~370k pairs) and found genuine same-person
  similarity dipping to **0.5946** for short (<2s) gaps and **0.5290**
  across the whole video -- both below the old 0.6 threshold, meaning
  the tracker was, correctly per its own logic, sometimes refusing to
  revive a track because two real photos of the same person legitimately
  scored below the cutoff (lighting/pose/motion-blur variation, likely
  worsened by this video's known desaturation). Swept actual threshold
  values (0.45/0.50/0.55/0.60/0.65) through the real
  `Tracker`+`GestureStateMachine` pipeline on this video: 0.45 and 0.50
  both collapsed fragmentation from 4 track_ids down to 2 (the remaining
  2 are most likely the person not being fully in frame yet in the
  opening ~1.8s, not a re-ID failure -- not separately confirmed).
  Gesture-trigger count and type coverage (6 actions, all 4 types) were
  identical across every threshold tested -- gesture correctness was
  never actually blocked by this, only track-ID stability (and by
  extension, overlay/causality display and anything that depends on
  track continuity, like the sweep detector's rolling window). Picked
  **0.5** over 0.45: both perform identically on this test, and 0.5 sits
  just below the measured real-world floor rather than well below it,
  a smaller, more deliberate departure from the original default.
  Updated in both `config.TrackingConfig` and `tracking.Tracker`'s own
  constructor default. **Not validated against real multi-person
  footage** -- this project has no recorded multi-person session to
  measure a genuine different-person similarity distribution against, so
  the actual false-merge risk this lower threshold trades for is a real
  unknown, not zero. If multi-person testing ever shows two different
  people getting merged into one track_id, this is the first place to
  look.
- **Handedness assumption relies on MediaPipe's documented convention,
  unverified against a live camera.** `gestures/normalize.py` assumes
  LEFT_*/RIGHT_* landmarks are the subject's own anatomical left/right
  (MediaPipe's documented behavior), not mirrored screen-left/right. The
  gesture geometry itself (`static_poses.classify_arm`'s "outward"
  computation) is robust either way since it only depends on a shoulder
  and its own wrist being labeled consistently -- but the *mapping* the
  project brief specifies ("right arm out = next, left arm out =
  previous") only means what the user intends if the anatomical-labeling
  assumption holds. If real testing shows gestures feel swapped (e.g. your left arm
  reliably triggers NEXT), that points to this assumption being wrong for
  the compiled model, not a bug in the geometry -- the fix would be
  swapping which landmark index feeds `normalize_pose`'s "left"/"right",
  not rewriting the outward-detection math.
- **Single-arm-raised and other unmapped poses.** The brief's "raising one
  arm or both arms up or to the side" only specifies semantics for
  "out to the side" (L/R) and implies "both arms up" (mapped here to
  PLAY_PAUSE as a reasonable choice, not explicitly specified). A single
  arm RAISED (not both, not swept) is detected (`ArmPose.RAISED` is a
  real classification) but has no action mapped to it -- available for a
  future gesture if one is wanted.
- **Sweep detector doesn't check direction/monotonicity.**
  `dynamic_gestures.SweepDetector` triggers on "raised + wide horizontal
  range within a time window," which would also fire for vigorous
  side-to-side waving with a raised arm, not just a clean single sweep.
  Distinguishing those needs real gesture footage to know if it's
  actually a problem in practice before adding complexity (e.g. checking
  the trajectory is roughly monotonic, not just wide-ranging).
- **Volume gestures implemented (2026-09-12): right arm swipe up ->
  `VOLUME_UP`, left arm swipe down in front of the body -> `VOLUME_DOWN`.**
  Added the `ControlAction` pair (`gestures/types.py`) and a new
  `dynamic_gestures.VerticalSwipeDetector`, parameterized by direction
  (`"up"`/`"down"`) rather than a second near-duplicate class: checks a
  minimum vertical range (`MIN_VERTICAL_SWIPE_RANGE = 0.8` shoulder-widths,
  a reasoned starting point like the horizontal sweep's own thresholds,
  not tuned against real footage) via a simple first-vs-last-sample
  comparison (not a full trajectory/monotonicity analysis -- same
  first-pass honesty as `SweepDetector`), while bounding horizontal drift
  to `SIDE_X_THRESHOLD` (reused directly from `static_poses.py` rather
  than an independently-guessed value) so "in front of the body" means
  precisely "wouldn't also register as `OUT_TO_SIDE`." Wired into
  `GestureStateMachine` as a second detector per relevant arm (right gets
  `right_sweep` + `right_volume_swipe`, left gets `left_sweep` +
  `left_volume_swipe`), firing independently of the static-pose debounce
  path exactly like the existing SKIP sweep does. Unit-tested (both the
  detector in isolation and end-to-end through `GestureStateMachine`) --
  82/82 tests passing.
  - **Re-run against the real `cheese` recording (2026-09-12): the
    originally-flagged risk didn't happen, but a different real one did.**
    Re-ran the exact same footage/pipeline used to validate the original
    four actions. Full regression check passed -- all six original
    triggers (NEXT x2, PREVIOUS, PLAY_PAUSE, SKIP x2) fired at the exact
    same timestamps as before, unaffected by the new detectors.
    `VOLUME_DOWN` never fired once, across every `RAISED`-arm-relaxing-
    to-`DOWN` transition in the video (there are several) -- the
    specific collision this was flagged for didn't materialize here.
    But `VOLUME_UP` fired **5 times**, and inspecting the actual raw
    `rel_y` trajectory around each one (not just the trigger/no-trigger
    result) showed a mixed picture, not a clean pass:
    - 2 of 5 (t=12.47s, t=15.70s) show a genuinely clean, fast,
      monotonic rise (roughly +2.7 to -2.7 shoulder-widths in ~0.3-0.6s)
      -- exactly the motion this detector is supposed to catch, even
      though the person wasn't intentionally testing this gesture (this
      recording predates the feature).
    - 2 of 5 (t=27.17s, t=30.03s) fired while the wrist stayed in its
      normal at-rest range the entire time (~2.4-3.4, nowhere near
      `RAISE_Y_THRESHOLD`) -- the detector only checks *relative* range
      and direction between the window's first and last sample, with no
      requirement that the motion actually ends up raised or starts from
      a settled baseline, so ordinary pose-estimation jitter/slow drift
      within the resting range was enough to satisfy
      `MIN_VERTICAL_SWIPE_RANGE=0.8` on its own.
    - 1 of 5 (t=19.33s) fired while the arm was already up near a
      confirmed `RAISED` position (having risen a moment earlier) --
      the rolling window still partly overlapped the original rise, so
      normal jitter while *holding* a raised pose read as "more
      swiping." (The full picture turned out to be worse than one
      isolated re-trigger -- see below.)
  - **Both fixes implemented and re-validated (2026-09-12).** Fix 1:
    `VerticalSwipeDetector._detect()` now requires the window's relevant
    endpoint to actually cross into raised territory (`up`: last sample
    `rel_y < -RAISE_Y_THRESHOLD`; `down`: first sample) rather than
    accepting any sufficiently-large relative delta regardless of where
    it starts/ends. This alone eliminated both "confined to resting
    range" false positives. Fix 2, for the held-pose case, turned out to
    need a materially different design than first planned: an initial
    attempt reset each swipe detector once at the moment its target pose
    was confirmed, but re-running against real footage showed the arm
    was *already* confirmed `RAISED` well before the observed window even
    started -- a one-time reset at the transition doesn't help partway
    into an already-long hold, and the video showed VOLUME_UP
    re-triggering roughly every 0.6s throughout the entire ~1.3s hold,
    not just once. The actual fix: gate evaluation on the *current*
    confirmed state, not just reset at the transition -- `right_volume_swipe`
    is now only fed samples while the arm ISN'T already confirmed
    `RAISED`, so a held pose simply stops accumulating any history at all
    for as long as the hold lasts, however long that is. This gate is
    safe for the "up" case specifically because the legitimate trigger
    already fires *before* `RAISED` confirms (debounce lags the raw
    crossing by a few frames), so nothing real gets cut off.
    - **The mirror-image gate for `left_volume_swipe` (evaluate only
      while confirmed `RAISED`) was tried and reverted -- it broke real
      swipe-down detection.** By the time `RAISED` actually confirms, a
      fast continuous rise-then-fall motion has often already started
      descending below the raised threshold, so gating on "still
      confirmed raised" cut off evaluation right as the descent itself
      needed measuring. There's also no real evidence this gate is
      needed for "down" in the first place: `VOLUME_DOWN` has never
      fired once across any real-footage test run, gated or not. Left
      as the fix-1-only (endpoint check, no gating) version, which was
      already working correctly.
    - **Final re-validation against the same `cheese` recording:** all
      six original actions still fire at the identical timestamps
      (zero regression, confirmed three times now across three rounds of
      changes), `VOLUME_DOWN` still never fires, and `VOLUME_UP` now
      fires exactly **once** (t=12.60s) -- the one instance already
      confirmed as a genuinely clean, fast rise. Both spurious patterns
      are gone with no new ones introduced. 85/85 unit tests passing.
  - `control/` (media control backend, milestone 6) still doesn't exist,
    so these join `NEXT`/`PREVIOUS`/`PLAY_PAUSE`/`SKIP` as actions that
    print/display but don't yet drive real volume control.

## Hardware

- **ArduCam v2 8MP compatibility.** Not yet tested on the Q6A. No matching
  overlay was found in the `radxa-overlays` DKMS package during the Radxa
  Camera 4K investigation (`docs/hardware.md`) -- likely needs its own
  overlay/driver work, or may not be supported on this board at all.
  Low priority since the Nexigo webcam and Radxa Camera 4K already cover
  the two camera interfaces (USB and CSI) this project cares about.
- **Ventuno Q port.** Revisit once the pre-order arrives -- see
  `docs/hardware.md` for why it looks like the strongest long-term fit
  (40 TOPS, 3x MIPI CSI).
- ~~**Boot media: microSD vs UFS.**~~ Resolved 2026-09-11: reflashed to
  the r2 UFS image variant, now booting from the 128GB UFS module instead
  of the microSD card (see `docs/setup-q6a.md`).

## Product scope (from the original project brief)

Deferred to keep the first milestone shippable -- see the approved plan's
"Scope confirmation" decision:

- **Controller promotion mechanic** (e.g. repeated head-taps or a
  distinctive gesture like the chicken dance) to give one person exclusive
  control and reduce bystander false positives.
- **Crowd voting system** for a shared subset of gestures (e.g. an overhead
  skip swipe available to everyone in frame, tallied and only acted on past
  a majority threshold within a time window).
- **Zone controls** to mask the pipeline to a configured region of interest,
  both to cut false positives from people outside the intended control area
  and to improve performance.
