# Backlog / deferred items

Things intentionally deferred out of the current milestone work, so they
don't get lost.

## Tooling

- **Web dashboard: live overlay feed + gesture queue -- initial draft
  implemented (2026-09-12).** `web/dashboard.py`: a browser page showing
  the live overlay frame (MJPEG stream) plus a scrolling gesture queue
  (polled JSON), built on the standard library's `http.server`
  (`ThreadingHTTPServer`) -- no Flask or other dependency added, matching
  this project's existing minimal-dependency preference (see
  `models/osnet_x0_25/README.md`). `DashboardState` is the thread-safe
  handoff point: `app.py`'s capture loop calls `update_frame()` (JPEG-
  encodes and stores the latest overlay frame) and `add_event()` (per
  triggered gesture) once per iteration; the HTTP server's request
  handlers (one thread per open connection) read from the same state.
  Config-gated via `OverlayConfig.web_enabled` (default `False`) /
  `web_port` (default 8080), exactly as planned -- doesn't bind a socket
  unless explicitly turned on. No authentication, as scoped -- a
  deliberately-started LAN-local tool.
  - Purpose unchanged from the original ask: demos (causality visible to
    someone not standing at the board's HDMI output), debugging, and
    pipeline iteration -- this whole session's workflow for every finding
    (exposure diagnosis, orientation bug, ROI tuning, re-ID threshold)
    was capture-on-device -> `scp` down -> view/analyze -> iterate; this
    removes that round-trip for anything that doesn't need per-pixel
    inspection.
  - Validated two ways: 6 unit tests (`tests/test_web_dashboard.py`)
    exercising the actual HTTP server over a real loopback socket (not
    mocked) -- index page, events JSON, MJPEG stream headers/boundary,
    event-history bounding, 404 handling -- and a manual visual check
    (synthetic frames + fake gesture events fed into a running server,
    viewed in an actual browser): confirmed the video pane updates live
    and the gesture queue populates and scrolls correctly.
  - **Not yet done:** run on the physical Q6A itself (only validated in
    this dev environment so far -- should just work, since it's pure
    Python stdlib + cv2 JPEG encoding, both already present on-device,
    but "should work" isn't "confirmed working" by this project's own
    standard). No reconnection/backpressure handling if a viewer's
    connection is slow (the stream handler just blocks on `wfile.write`,
    which could stall the one thread serving that connection --
    `ThreadingHTTPServer` gives each connection its own thread so this
    wouldn't block the capture loop or other viewers, but a genuinely
    stalled slow client would leak that one thread until it times out or
    disconnects). Single global dashboard state, not scoped per-track --
    fine for the single-person demos this was built for, would need
    rethinking for serious multi-person use.
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
- **YOLO26-Pose is noticeably slower than the default `qnn` pipeline --
  not yet root-caused.** Measured on the real-footage comparison
  (`docs/backlog.md`'s YOLO26 entry above): ~12.6fps vs ~22.1fps for the
  same recording on the Q6A. Untested hypotheses, in rough order of
  suspicion, none confirmed yet:
  - **Compute cost, not overhead.** YOLO26-Pose runs one full 640x640
    end-to-end forward pass every frame; the `qnn` pipeline's landmark
    model only processes a small per-person crop (`_blazepose.py`'s ROI),
    with the 640x640 YOLOv8n-det stage only used for coarse
    localization. Per-model NPU timing already measured this project
    (YOLO26 ~14.9ms vs BlazePose landmark ~1.1ms + YOLOv8n-det ~2.2ms
    combined) suggests this alone could explain most of the gap -- if
    so, it's an inherent architecture tradeoff, not a bug.
  - **`w8a16` vs `w8a8` quantization overhead.** Every other model this
    project runs uses `w8a8`; YOLO26 requires `w8a16` (16-bit
    activations) on this hardware. Unconfirmed whether/how much slower
    `w8a16` execution is on Hexagon v68 specifically.
  - **Python-side decode cost.** `_yolo26_pose.decode_raw_output` +
    `detect_poses`' NMS run every frame in pure numpy over 8400 raw
    anchors, unlike the `qnn` pipeline's much smaller post-NMS outputs
    from an AI-Hub export with post-processing baked into the graph.
    Not profiled separately from the NPU inference call -- worth timing
    in isolation before assuming the NPU itself is the bottleneck.
  - Whatever the cause, ~12.6fps is still usable for gesture-triggered
    control (not a video-smoothness-critical application), so this is
    an understand-it item, not necessarily a fix-it-before-using item.
- **First live (not scripted) dashboard test of `yolo26`, 2026-09-12 --
  real CPU/RAM/FPS numbers, one person, ~45 minutes.** Ran the actual
  `pose_controller.app` entry point (not a scratch script) via
  `AppConfig.from_yaml` with `inference.backend: yolo26`, re-ID, and
  `control.enabled: true` (`playerctl`, no real player running --
  correctly showed "No player found" on the dashboard panel) all on at
  once, dashboard streamed over an SSH port-forward for live interactive
  testing and feedback. User feedback: "way better" than the existing
  pipeline, consistent with the earlier scripted real-footage validation
  above; some missed detections with the person only partially in frame
  or at odd angles, "expected."
  - **Real resource numbers** (5s-interval sampling throughout the
    session, `ps`): ~9.6fps average (range 6.6-10.3, lower than the
    ~12.6fps scripted test -- this run additionally includes real webcam
    capture, re-ID appearance matching, dashboard JPEG encoding, and
    media-status polling overhead that the earlier synthetic-input
    script test didn't have), ~552% average CPU (max 640%) on this
    board's 8 cores (~69-80% of total capacity), ~251MB average RSS
    (max 253MB, stable -- no apparent leak over 45 minutes) out of
    7.4GB total system RAM. NPU utilization itself isn't visible through
    any tool used so far -- only per-model inference *latency*, not a
    load percentage, so "how much NPU headroom is left" remains unknown.
  - **Found and fixed a real, unrelated bug while setting this test up**:
    `Camera.__init__`'s plain `cv2.VideoCapture(source)` intermittently
    fails to open the Nexigo webcam on this board (OpenCV auto-selects
    GStreamer, whose pipeline sometimes fails to start) -- see the
    Hardware section below for the fix. This had never been caught
    before because every prior on-device test went through hand-written
    scratch scripts that happened to already hardcode the working
    `cv2.CAP_V4L2` backend explicitly; this was the first time the real
    `Camera` class was exercised live on this board.
  - **Follow-up requested**: extend the overlay to a full bounding box +
    full-body skeleton (matching typical YOLO26 demo visualizations) --
    the model output already supports this for free, see
    `docs/architecture.md`'s `overlay/` description. Implemented the
    same day, unit-tested, visually verified via a synthetic pose render
    (not yet re-verified live against real footage after the change).
- **Same live dashboard test repeated on the Q8B, 2026-09-13, after
  physically moving the Nexigo webcam over.** Real webcam this time was
  `/dev/video2`, not `/dev/video0` like the Q6A -- the Q8B's SoC video
  codec (`docs/setup-q8b.md`'s "Iris Decoder"/"Iris Encoder" finding)
  already occupies indices 0/1, confirmed by testing both new device
  nodes directly rather than assuming. The `Camera` fix from the Q6A test
  above applied cleanly with **no workaround needed this time** -- real
  validation that the fix generalizes, not just a same-board fluke. User
  feedback: works fine, "probably a higher frame rate than the Q6A."
  - **Real resource numbers, same methodology as the Q6A run**: ~9.9fps
    average (range 7.9-9.9) -- confirms the user's read, though the gap
    is small. The bigger difference is CPU: ~245% average (max 259%) on
    this board's 8 cores, versus the Q6A's ~552% average for
    *essentially the same throughput*. Memory: ~279MB average RSS (max
    287MB, stable). See `docs/pipelines.md`'s "Live system resource
    usage" table for both boards side by side, including a hypothesis
    for why matched FPS plus much lower CPU might mean the loop's actual
    bottleneck isn't the NPU call at all on either board -- not
    confirmed, worth profiling the loop's individual stages before
    concluding anything further.

## Tracking / multi-person (milestone 3 follow-ups)

- ~~**Appearance-based re-ID.**~~ Resolved 2026-09-11: `tracking.reid.
  ReidEmbedder` (OSNet-x0.25, plain FP32 on CPU -- see
  `models/osnet_x0_25/README.md` for why not NPU/quantized) wired into
  `Tracker` via a lost-track gallery. Validated on the physical Q6A: a
  track that fully left frame and reappeared at a different position was
  correctly revived by appearance (0.978 similarity for the same person,
  0.419 against background) -- see `scripts/qnn_spike.md`.
- ~~**Re-ID similarity threshold (0.6) is a starting guess, not tuned.**~~
  Retuned 2026-09-12 against real recorded footage (measured same-person
  embedding similarity dipping as low as 0.53) to 0.5 -- see the
  "Gestures" section below for the full measurement and the still-open
  false-merge risk this trades for, since real multi-person footage to
  measure a *different*-person distribution against still doesn't exist.
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
- **QNN landmark visibility output hovers near 0.5 for everything --
  confirmed again on live footage (2026-09-12), still unresolved.**
  Noticed during the original NPU spike (static photos only, all 25
  points ~0.5) and now reconfirmed on 30+ seconds of live camera input
  while a facing-angle diagnostic was running for an unrelated question
  (see "Gestures" below): shoulder and wrist visibility sat stable at
  ~0.50-0.51 the *entire* time, regardless of pose, motion, or which way
  the person was facing. This isn't a diagnostic bug this time (the
  earlier version of this same test *was* buggy -- it skipped the
  required sigmoid transform entirely, showing raw near-zero values
  that looked like near-zero visibility but were actually just an
  un-transformed logit; fixing that produced these stable ~0.5 numbers
  instead). A visibility channel that doesn't move from ~0.5 regardless
  of real occlusion state can't usefully separate visible from occluded
  keypoints, which matters directly for `gestures/normalize.py`'s
  `MIN_KEYPOINT_VISIBILITY = 0.3` gate and `static_poses.classify_arm`'s
  "wrist not visible" handling. Still not root-caused -- may be how this
  specific compiled BlazePose landmark model was calibrated, or a
  quantization precision-loss issue specific to this one output channel
  (the same project has already found one other channel,
  `class_idx` on the YOLOv8n-det model, with a nonsensical quantization
  scale that needed a workaround -- see `models/yolov8n_det_qcs6490/
  README.md` -- so a similarly mis-calibrated channel here wouldn't be
  unprecedented). Worth checking whether YOLO26-Pose's independent
  per-keypoint visibility (see below) has the same problem or not, once
  that pipeline is testable.
- **Overlay flicker and ghosting from track fragmentation -- two fixes
  shipped (2026-09-12), root cause still open.** Real on-device dashboard
  testing surfaced two related but distinct display problems. First,
  `Tracker.update()` only ever returns *this frame's* real detections by
  design (never a synthesized result for a track that's merely coasting,
  since `gestures.state_machine` needs to know "no observation this
  frame" as distinct from a confirmed `DOWN` -- see `_ArmDebouncer`), so
  a detection miss meant the overlay had nothing to draw for that
  person, popping their skeleton in and out on every gap. Fixed with
  `app.update_held_overlay`: holds each track's last-known pose/arm-
  states purely for display until either a fresh detection replaces them
  or `max_age` frames pass -- gesture recognition itself is untouched,
  still seeing only real per-frame detections.
  - Second, and more concerning: the *same continuously-present person*
    was observed spawning track_ids up into the teens within seconds,
    including while sitting nearly motionless just typing -- and with
    the flicker fix now holding each one's last overlay, this stacked
    several ghosted skeletons on the same physical spot, "very messy"
    per direct observation. Pulling a live frame from the dashboard
    showed the likely proximate cause: the camera was still aimed
    steeply upward (the same framing problem from the facing-camera
    investigation below), with the person only a tiny, inconsistent
    sliver of the frame -- an unstable partial-body crop plausibly
    degrades both IoU box matching and re-ID embedding quality at once.
    Added an overlap-dedup pass to `update_held_overlay` regardless
    (`OVERLAY_DEDUP_IOU = 0.5`): when two displayed poses (held or
    fresh) heavily overlap, only the fresher one survives -- this
    reduced the visual mess but the user's own follow-up testing still
    showed "some odd ghosting" afterward, and explicitly pushed back
    on camera framing being the full explanation. **Not resolved.**
    This uncertainty -- is the remaining fragmentation still framing,
    or something in the tracker/re-ID logic itself -- is a primary
    motivation for building the YOLO26-Pose alongside-pipeline below:
    if fragmentation drops substantially on the same footage/framing
    once the two-stage crop pipeline is removed, that's real evidence
    toward pipeline instability over framing, not just a plausible story.
- **YOLO26-Pose alongside-pipeline: exported and integrated
  (2026-09-12), not yet validated on hardware.** `inference.backend:
  yolo26` (`configs/dragon_q6a_yolo26.yaml`) selects a single end-to-end
  detection+17-keypoint model in place of the current YOLOv8n-det +
  BlazePose-landmark two-stage pipeline -- built to run *alongside* the
  existing `qnn` backend for direct A/B comparison, not to replace it
  outright. See `docs/journey.md` for the full reasoning and
  `models/yolo26n_pose_qcs6490/README.md` for the model's provenance and
  I/O contract once exported. Motivated by two specific, already-
  diagnosed weak points this architecture sidesteps by construction: the
  current pipeline's manual crop/ROI step (`_roi_corners_from_box`,
  already the source of real tuning bugs this project found the hard
  way) doesn't exist in a single end-to-end model, and BlazePose's
  landmark model's one scalar all-or-nothing confidence gate
  (`MIN_LANDMARK_SCORE`) is replaced by an independent per-keypoint
  visibility, which may (or may not -- untested) also sidestep the
  facing-camera symptom and/or the near-constant-0.5-visibility issue
  noted above. Code is written and unit-tested
  (`tests/test_yolo26_pose.py`); the on-device quantization I/O contract
  (`inference/backends/yolo26_qnn.py`'s `_OUTPUT_QUANT`) is still
  placeholder values pending inspection of the actual exported model,
  and nothing has been run on the physical Q6A yet. **Do not treat this
  as validated or as a decided replacement** -- the plan is to thoroughly
  test both pipelines side by side before any final call.
- **YOLO26-Detection staged for dashboard visualization (2026-09-12), not
  the pose pipeline.** A second, deliberately separate capability from
  the YOLO26-Pose alongside-pipeline above: `inference/backends/
  _yolo26_detect.py` + `yolo26_detect_qnn.Yolo26Detector` run YOLO26's
  plain detection variant (all 80 COCO classes, not filtered to person)
  purely as an optional visualization layer -- `overlay.draw_detections`
  draws every detected object (gray boxes + class + score, visually
  distinct from the pose skeleton's colored per-track boxes) on the
  dashboard feed, gated by `overlay.detect_overlay_enabled` (off by
  default; it's a full extra NPU model call per frame with no benefit to
  the core pipeline unless someone's actually looking at the dashboard).
  Not wired into `Tracker`/`GestureStateMachine` at all -- this has
  nothing to do with pose or gestures today.
  - **Why this exists, per the actual request:** immediately, to see
    what a general detector considers "an object" in frame versus what
    the pose pipeline considers "a person" -- a genuinely useful, cheap
    diagnostic overlay on its own. Longer-term (not built, explicitly
    deferred): the original project brief this whole thing is meant to
    eventually distill back into is swim-safety monitoring, and a pool
    is exactly the kind of visually cluttered/distorting scene where
    knowing *what else* is in frame (equipment, reflections
    mis-detected as something) alongside where people are could help
    explain or filter false pose/track detections. Nothing here reasons
    about that yet -- it's raw detections only, staged for whenever that
    work actually starts.
  - Code written and unit-tested (`tests/test_yolo26_detect.py`, 6
    tests); real quantization I/O contract
    (`yolo26_detect_qnn.py`'s `_OUTPUT_QUANT`) is placeholder pending the
    export, same caveat as the pose model above. Confirmed the CPU
    dev-loop backend never imports `onnxruntime_qnn` when this is
    disabled (the default) -- `Yolo26Detector` is imported lazily inside
    `app.build_detect_overlay`, not at module load time.
- ~~**YOLO26 (both variants) fails to compile for QCS6490 via Qualcomm AI
  Hub -- reproducible, not a fluke (2026-09-12).**~~ **Resolved the same
  day via a different export toolchain -- see the entry below.** Both
  `yolo26n_pose_qcs6490` and `yolo26_det_qcs6490` AI-Hub exports failed:
  the QDQ-to-context-binary compile job failed identically for both with
  "Conversion to context binary failed with exit code 14," confirmed via
  `qai_hub.get_job(job_id).get_status()` directly (the AI Hub CLI's own
  console output is misleading -- it shows a "stuck at 0/3 CREATED"
  spinner even after the job has actually failed, so always check job
  status via the API directly, not the CLI progress display). YOLO26
  only supports `w8a16` quantization via AI Hub for this chipset (older
  models here use the proven `w8a8` path). Tried `--precision float` as
  a fallback: failed immediately with `ValueError: ... requires FP16
  support, but the selected device does not support FP16` -- QCS6490/
  Hexagon v68 genuinely lacks FP16 hardware support, confirmed by AI
  Hub's own device-capability check. **The FP16 finding is still real and
  worth a GitHub issue against `qualcomm/ai-hub-models`** (not yet
  filed) -- but the original conclusion drawn from the exit-code-14
  failure ("genuine YOLO26 + w8a16 + QCS6490 incompatibility, not fixable
  from this project's side") was **wrong**: it was specific to AI Hub's
  own compile pipeline, not the hardware or the model architecture. See
  below.
- **YOLO26 actually works on this hardware -- via Ultralytics' own local
  QNN export, not Qualcomm AI Hub (resolved 2026-09-12).** The user
  pointed out that Ultralytics documents official Hexagon V68 support
  for its own QNN export path (https://docs.ultralytics.com/integrations/qnn),
  entirely separate from AI Hub's cloud compile service (fully local, no
  Qualcomm account, `model.export(format="qnn", name="68")`). Tried it
  directly for both `yolo26n.pt` (detect) and `yolo26n-pose.pt`: **both
  exported successfully in ~10 seconds each**, producing a
  self-contained `.onnx` file with a genuine embedded `EPContext` QNN
  context binary (confirmed via `onnx.load()` -- real node, not a
  fallback). Copied both files unmodified to the Q8B and ran them
  through `onnxruntime-qnn`'s QNN execution provider: **both execute
  successfully on the real Hexagon v68 NPU** -- `yolo26n_qnn.onnx`
  ~7.1ms median, `yolo26n-pose_qnn.onnx` ~6.7ms median (20 warm calls
  each), matching the timing profile of every other on-NPU model this
  project has confirmed. This directly contradicts the AI-Hub-only
  conclusion above -- YOLO26 + Hexagon v68 is not incompatible, AI Hub's
  specific compiler/toolchain version has a bug or limitation with this
  model+precision combination that Ultralytics' own (likely newer or
  differently-configured) QNN toolchain doesn't hit.
  - **I/O contract is different from what was originally planned for
    (and different from AI-Hub-compiled models generally)**: plain
    float32 in/out (the graph's own `QuantizeLinear`/`DequantizeLinear`
    nodes handle quantization internally, confirmed via the input
    tensor's scale ~2^-16 matching 16-bit activation quantization), and
    **not post-processed** -- Ultralytics' export stops after per-anchor
    box/keypoint decode, leaving NMS as the caller's job (unlike AI Hub's
    `include_postprocessing=True` convention this project's other YOLO
    models were built around). `_yolo26_pose.py`/`_yolo26_detect.py`
    gained a `decode_raw_output` function each (cxcywh->xyxy + channel
    splitting -- confirmed exactly from `ultralytics.nn.modules.head`
    source, not guessed) to bridge into the existing, unchanged
    score-filter+NMS+mapping logic. `yolo26_qnn.py`/`yolo26_detect_qnn.py`
    were rewritten to use the new `_qnn_runtime.create_qnn_session` (a
    plain session, no manual `QuantSpec`) instead of `OnnxQnnRunner`.
    Placeholder `QuantSpec` values are gone -- this model's real I/O
    contract was verified directly, not guessed at.
  - Real models now committed to `models/yolo26n_pose_qcs6490/` and
    `models/yolo26_det_qcs6490/` (see each dir's `README.md` for full
    provenance) -- these were empty placeholders before. `ultralytics`
    added to `pyproject.toml`'s `hub` extra as a documented dev-machine
    export dependency.
  - **Correction: the "no reason to expect a different result on the
    Q6A" assumption above was wrong -- tested directly and it failed.**
    Copying the Q8B-verified `model.onnx` files to the Q6A and loading
    them produced a real, reproducible error:
    `QNN_CONTEXT_ERROR_CREATE_FROM_BINARY: Failure to create context
    from binary` -- not a version mismatch (`onnxruntime-qnn` 2.6.0 on
    both boards) and not QNN loading being broken in general (the
    existing AI-Hub-compiled `yolov8n_det_qcs6490` loaded fine on the
    same Q6A install, sanity-checked directly). Root cause, found by
    checking `qai_hub.get_devices()`'s own device attributes rather than
    guessing: **`soc_model` in the QNN EP is a per-exact-chip identifier,
    not per-Hexagon-architecture-generation** -- `hexagon:v68` alone
    covers at least four different `soc-model` values across Qualcomm's
    real device catalog (30, 35, 39, 93). Ultralytics' generic
    `name="68"` export option only sets `htp_arch=68` (the architecture
    generation), which apparently finalizes against a *different* v68
    variant than QCS6490's own -- compatible with the Q8B's SC8280XP by
    coincidence/overlap, not with the Q6A's QCS6490.
  - **Fix**: QCS6490's real `soc_model` value is `93`, read directly off
    AI Hub's own `Dragonwing RB3 Gen 2 Vision Kit` device entry (a real
    QCS6490 board AI Hub does support) -- not a guess or a public-docs
    lookup (Qualcomm's own QNN SDK docs don't publish this mapping
    anywhere findable). Ultralytics has no built-in name for it, but its
    `QNN_HTP_TARGETS` dict is a plain importable module-level dict, so
    registering one is a one-line patch:
    `QNN_HTP_TARGETS["qcs6490"] = ("soc_model", "93")` before calling
    `.export(format="qnn", name="qcs6490")`. Re-exported both variants
    this way and **confirmed running on both boards**: detect ~14.7ms
    (Q6A) / ~9.6ms (Q8B), pose ~14.9ms (Q6A) / ~9.9ms (Q8B). The
    `soc_model=93` binary is strictly better than the generic one -- it
    replaced it in `models/yolo26n_pose_qcs6490/` and
    `models/yolo26_det_qcs6490/` (see each `README.md` for the full
    story), not a separate per-board artifact.
  - **Validated against real footage (2026-09-12): a real, positive
    result.** Ran the full pipeline (`Yolo26PoseEstimator` +
    `Tracker`/`ReidEmbedder` + `GestureStateMachine`, identical config to
    the existing `test_cheese_gestures_reid.py`) against the same
    recorded cheese-webcam footage (`~/Videos/Webcam/2026-09-12-024357
    .webm` on the Q6A, one real person walking in/out of frame) already
    used to characterize the existing `qnn` pipeline's flicker/ghosting
    behavior, for a direct, same-footage comparison -- see
    `docs/pipelines.md`'s "Full-pipeline real-footage results" table for
    the numbers (detection rate nearly doubled, 49%->96%; the person who
    fragmented into two tracked identities under `qnn` held one
    continuous identity throughout under `yolo26`, aside from a ~4-frame
    flicker in the very last moment as they left frame; 7 vs. 14 gestures
    triggered; ~22.1fps vs ~12.6fps).

    This is the exact question this alongside-pipeline was built to
    answer: does removing BlazePose's two-stage crop/ROI step and its
    single all-or-nothing confidence gate reduce identity fragmentation
    and improve detection reliability? On this real recording, yes on
    both counts -- detection rate nearly doubled and the sustained
    fragmentation essentially disappeared, at the cost of roughly half
    the frame rate (still comfortably usable for gesture-triggered
    control; this isn't a video-smoothness-critical application).
    Confirmed via raw per-frame diagnostics, not just the tracker's
    output, that YOLO26 never detects more than 1 person simultaneously
    in this footage (there's only ever been 1 real person in frame for
    any test so far) -- multi-person behavior specifically is still
    completely unvalidated and shouldn't be assumed from this result.
    **One real, honest tradeoff, not yet investigated further**: this
    export's INT8 calibration used Ultralytics' default 4-image
    `coco8`/`coco8-pose` dataset, well under the "300+ recommended"
    warning it printed at export time -- accuracy on harder real-world
    cases (smaller/farther/partially-occluded people, multi-person
    scenes) is unproven and could plausibly be improved with a properly-
    sized, this-project-specific calibration set later.
- **HRNetPose alongside-pipeline: exported, compiled, and integrated
  (2026-09-12), not yet validated on hardware.** Same motivation as the YOLO26-Pose alongside-pipeline
  above (test whether removing the two-stage crop/ROI pipeline reduces
  fragmentation/ghosting) but via a model that uses the same proven
  `w8a8` quantization path as the existing `qnn` backend, rather than
  YOLO26's blocked `w8a16` path. Architecturally this is a top-down,
  single-person, pre-cropped 256x192 model -- the same role as
  BlazePose's landmark stage, not a whole-frame detector -- so it would
  slot in alongside the existing YOLOv8n-det first stage, not replace
  the two-stage design outright.
  - Getting the export CLI to even run required working around a broken
    dev-machine dependency chain: `hrnet_pose`'s package `__init__`
    unconditionally imports real `mmpose`/`mmdet`/`mmcv` (only needed for
    an interactive demo feature this project never uses). `mmengine`
    installed cleanly; `mmcv` failed to build on Python 3.13 (`pkg_resources`
    removal, then a `KeyError: '__version__'` in its own legacy setup.py)
    -- worked around with the prebuilt `mmcv-lite` wheel, pinned to
    `2.1.0` specifically (mmdet asserts `mmcv<2.2.0`). `mmpose`/`mmdet`
    themselves are pure-Python wheels and installed fine once pulled in
    with `--no-deps` and their real dependencies installed individually,
    skipping the three that are broken/unneeded (`chumpy`, `munkres`,
    `xtcocotools>=1.12` -- `qai_hub_models.extern.mmpose
    .patch_mmpose_no_build_deps()` auto-stubs the latter two with
    `MagicMock` when absent; `chumpy` never turned out to be needed on
    this import path at all). Also needed `yacs` (clean) and
    `QAIHM_CI=1` env var to auto-confirm export.py's interactive "ok to
    clone external repo" prompt non-interactively.
  - **Caught and fixed a real mistake while doing this work:** the
    `torchvision`/mmpose install pulled in `torch==2.14.0` as a
    transitive dependency, silently upgrading past `qai_hub_models`'
    own pin (`torch<=2.11.0,>=2.4`) -- would have risked breaking the
    export toolchain for every model, not just HRNetPose. Caught before
    running the actual export and pinned back down to `torch==2.11.0` /
    `torchvision==0.26.0` (both `+cpu`, matching what was already
    working). Also caught that the very first install attempt landed in
    the machine's global Python, not the project `.venv` -- redone
    correctly against `.venv/Scripts/python.exe` before anything that
    mattered ran.
  - Once the import chain was fixed, `export.py --chipset qcs6490
    --target-runtime precompiled_qnn_onnx --precision w8a8` actually
    **scheduled real jobs** (float-model compile, quantize, QDQ compile,
    profile) -- notably further than YOLO26 got. (The CLI itself crashed
    at the very end with an unrelated `UnicodeEncodeError` trying to
    print a Unicode spinner character to a `cp1252` Windows console --
    cosmetic, the jobs were already scheduled server-side by that point;
    check real status via `qai_hub.get_job(job_id).get_status()`.)
  - **Result: float compile, quantize, QDQ-to-context-binary compile, and
    profiling all SUCCEEDED.** This is the exact step that failed with
    "exit code 14" for both YOLO26 variants -- HRNetPose compiles cleanly
    for QCS6490 on the proven `w8a8` path where YOLO26's `w8a16` path did
    not. See `docs/pipelines.md`'s performance table for real measured
    inference time on both boards (well within a 30fps budget alongside
    the shared YOLOv8n-det first stage). Model
    downloaded and committed to `models/hrnet_pose_qcs6490/` (real
    `input_spec`/`output_spec` read directly off the compiled model, no
    placeholder quantization values needed this time -- see that
    directory's `README.md`).
  - **Integration code written**: `inference/backends/_hrnet_pose.py`
    (heatmap decode: per-keypoint argmax + peak-value confidence, no
    sub-pixel refinement yet; reuses `_blazepose.py`'s box-to-ROI crop
    geometry and `_yolo26_pose.py`'s COCO->`Landmark` mapping table) and
    `hrnet_qnn.HrnetPoseEstimator` (reuses the same YOLOv8n-det first
    stage as the `qnn` backend). Selected via `inference.backend: hrnet`
    (`configs/dragon_q6a_hrnet.yaml`). Unit-tested
    (`tests/test_hrnet_pose.py`, 5 tests -- heatmap decode math,
    confidence clamping, box-to-frame coordinate mapping via a
    centered-box/centered-peak identity check, COCO keypoint identity
    mapping, one-result-per-box). Full suite (127 tests) still green.
  - **Still not validated end-to-end or tried on the physical Q6A** --
    everything above is dev-machine unit testing plus AI Hub's own
    device-farm profiling, not this project's actual gesture/tracking
    pipeline on real footage. That's the next step whenever hardware time
    is available, alongside the already-running `qnn` and `yolo26`
    configs for a real three-way comparison.

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

## Control (milestone 6)

- **`control/` built (2026-09-12): media control now drives a real
  player via D-Bus MPRIS, not yet validated on real hardware.** Per the
  original project plan's own stated approach ("emulates OS media keys /
  Linux D-Bus MPRIS against whatever player is already running and
  authenticated"), `control.media_controller.MediaController` is an
  abstract interface (`play_pause`/`next`/`previous`/`volume_up`/
  `volume_down`/`get_status`) with `dispatch(ControlAction)` owning the
  action->method mapping (`SKIP` and `NEXT` -- two different gestures --
  both map to `next()`, since there's no separate "skip forward N
  seconds" concept anywhere in this project's gesture vocabulary).
  - **Backend: `control.backends.playerctl.PlayerctlController`**, shelling
    out to the `playerctl` CLI (`sudo apt install playerctl`) rather than
    a D-Bus client library directly -- `playerctl` already handles
    picking "the" active player when several run at once, and avoids
    adding `dbus-python` (needs system dev headers to build) as a
    dependency for a first implementation. Works against Spotify's
    official Linux client (which exposes MPRIS) with zero Spotify-
    specific code; any other MPRIS-compliant player works identically.
    A background thread polls `playerctl status`/`metadata` every 2s for
    `get_status()` (display only, e.g. the dashboard's media panel) --
    every actual gesture-triggered action is a separate, immediate,
    short-timeout (2s) subprocess call so a hung D-Bus call can never
    block a gesture trigger.
  - Off by default (`control.enabled: false`) and degrades to a
    `NullMediaController` no-op if `playerctl` isn't installed or fails
    to start, matching this project's existing degrade-don't-crash
    pattern (`app.build_tracker`'s `ReidEmbedder` fallback) -- an
    optional feature failing to initialize should never take the whole
    app down.
  - Web dashboard gained a "Media Control" panel (connected/paused/
    playing status dot, now-playing title/artist when available) polled
    via a new `/media` JSON endpoint, `DashboardState.update_media_status`.
  - ~~**Not yet tested against a real player or real hardware.**~~
    Resolved 2026-09-13/14 on the Q8B, against a real Spotify Connect
    session, not just `playerctl` against a stub. Since the official
    Spotify Linux client has **no ARM64/aarch64 build at all** (verified:
    amd64/i386 only, a longstanding unaddressed community request), and
    `spotifyd` has **no Bluetooth/BLE support** (confirmed via its own
    GitHub README), the path validated here is `spotifyd` v0.4.2
    (`linux-aarch64-full` prebuilt, chosen over `default`/`slim` for
    MPRIS) as the local Spotify Connect endpoint, with `playerctl`
    talking to it over D-Bus exactly as it would to the official client --
    zero Spotify-specific code anywhere in `control/`.
    - The prebuilt binary linked against OpenSSL 1.1, which Q8B's Ubuntu
      26.04 doesn't ship (`libssl.so.1.1: cannot open shared object
      file`, only OpenSSL 3.5.5 available and no `libssl1.1` package in
      its own apt repos). Fixed without a system-wide install or
      touching the real OpenSSL 3 packages: downloaded
      `libssl1.1_1.1.1f-1ubuntu2.24_arm64.deb` from
      `ports.ubuntu.com/pool/main/o/openssl/` (the ARM64-specific
      archive host -- `security.ubuntu.com`/`archive.ubuntu.com` don't
      carry arm64 packages) and extracted it locally with `dpkg -x` into
      `~/libssl1.1_compat/`, then ran `spotifyd` with
      `LD_LIBRARY_PATH` pointed at the extracted `.so`s.
    - Modern `spotifyd` has no `--username`/`--password` flags at all --
      auth is OAuth-only via `spotifyd authenticate --oauth-port
      <PORT>`, which prints an authorization URL for the user to open
      and complete themselves in their own browser (SSH-port-forwarded
      back to the board); credentials then cache under
      `~/.cache/spotifyd/` and survive a reboot, so re-authentication
      isn't needed on every restart.
    - Launched with `--backend alsa --use-mpris=true --dbus-type
      session --device-name "Pose-Controller-Q8B"`, advertised over
      zeroconf/Spotify Connect. One real quirk worth knowing: its MPRIS
      interface only registers on D-Bus once it becomes the **active**
      playback device (i.e. after a phone/laptop picks it from the
      Connect device list and starts playing) -- `playerctl -l` shows
      nothing beforehand, which looks like a failure but isn't.
    - **End-to-end validated live** through the actual dashboard/gesture
      pipeline against a real playing track ("Pink Pony Club" by
      Chappell Roan): `PLAY_PAUSE`, `SKIP`, and `NEXT` gestures all
      dispatched real commands that changed real Spotify Connect
      playback state, confirmed via both `playerctl -p spotifyd
      status`/`metadata` and the dashboard's `/media` endpoint tracking
      the live title/artist. Ran for roughly 30 minutes across 40+
      track IDs with no crashes or hangs.
    - **Bluetooth/BLE bridge remains a deferred, separate idea** (`spotifyd`
      itself has no Bluetooth support at all) -- noted here for later,
      not scheduled.

## Hardware

- **Q8B webcam device index is not stable across reboots.** First bring-up
  (2026-09-13) found the Nexigo webcam at `/dev/video2`, with the SoC's
  Iris video codec occupying `/dev/video0`/`/dev/video1` -- all three
  `configs/dragon_q8b*.yaml` were set to `source: "2"` accordingly. After
  a reboot the next day, the enumeration order flipped: the webcam came
  up as `/dev/video0`/`/dev/video1` and the Iris codec landed on
  `/dev/video2`/`/dev/video3` instead, confirmed by reading
  `/sys/class/video4linux/videoN/name` directly rather than assuming
  index continuity -- the deployed dashboard config had to be hand-edited
  back to `source: "0"` to recover. Root cause is USB vs. platform-device
  probe-order timing at boot, which isn't guaranteed stable on this
  board. **Not yet fixed properly** -- the real fix is matching by device
  name/`udev` rule (e.g. a persistent symlink keyed on the Nexigo's
  USB vendor/product ID) instead of a bare index, so the committed
  configs stop silently going stale across reboots. Low priority since
  it only affects the Q8B (the Q6A has a single camera and no competing
  codec device), but worth doing before this board is used unattended.
- **Q8B onboard fan not spinning** (noticed 2026-09-13 after a manual
  attempt to fix HDMI audio required a reboot). No PWM/tach control is
  exposed anywhere under `/sys/class/hwmon/*` on this board (checked all
  65 `hwmon` entries), so there's no software angle to investigate this
  further -- it's a physical connector/wiring check. Thermal zones read
  normal (mid-50s C) at idle; not yet observed under sustained inference
  load with the fan confirmed absent. Worth watching for thermal
  throttling on longer test sessions until the fan itself is checked.
- ~~**`Camera.__init__` intermittently fails to open the Nexigo webcam on
  the Q6A.**~~ Resolved 2026-09-12, found live during the first real
  dashboard test run through the actual `app.py`/`AppConfig` entry point
  (prior on-device testing all went through hand-written scratch scripts
  that happened to already hardcode `cv2.CAP_V4L2`, which is presumably
  why this was never caught before). Plain `cv2.VideoCapture(source)`
  lets OpenCV auto-select a backend, which on this board sometimes picks
  GStreamer -- and that GStreamer pipeline intermittently fails to start
  ("Internal data stream error", "unable to start pipeline") even though
  the exact same device index opens reliably via the explicit V4L2
  backend every time it was tried (confirmed with repeated manual
  `cv2.VideoCapture(0, cv2.CAP_V4L2)` calls -- 100% success -- versus
  plain `cv2.VideoCapture(0)`, which failed, then succeeded, then failed
  again across three consecutive attempts seconds apart). Fixed in
  `capture/camera.py`: force `cv2.CAP_V4L2` for a plain integer device
  index on Linux specifically (leaving Windows/dev-laptop and GStreamer-
  pipeline-string sources -- a `source` string is itself a way to pick a
  backend -- on the previous auto-select behavior, `cv2.CAP_ANY`), plus a
  one-retry-after-a-short-pause safety net in case even V4L2 hits a
  transient failure. Unit-tested with `cv2.VideoCapture` mocked
  (`tests/test_camera.py`, 8 tests) -- real validation is confirming this
  eliminates the intermittent failure across many real restarts on the
  physical Q6A, not yet done (the live dashboard test that found this bug
  is still running on a launcher-level workaround from before the fix
  landed, to avoid disrupting an in-progress test session).
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
- ~~**Radxa Dragon Q8B viability.**~~ Resolved 2026-09-12, and the initial
  answer was wrong: first check (AI Hub's own `qai_hub.get_devices()`
  device catalog has no `sc8280xp` entry) was read as "this board can't
  run anything," but that only rules out *new* AI-Hub cloud compiles
  targeting it by name. Per Radxa's own docs and confirmed on the
  physical board, SC8280XP and QCS6490 share the same Hexagon V68
  architecture (`fastrpc_test -a v68` passes identically to the Q6A), and
  this project's already-compiled QCS6490 models (`yolov8n_det_qcs6490`,
  `hrnet_pose_qcs6490`) load and run on the Q8B's real NPU via
  `onnxruntime-qnn` completely unmodified, at timings matching the Q6A.
  **The Q8B is a viable second deployment target for every model this
  project has already compiled, including YOLO26** (see the YOLO26 entry
  above -- its Ultralytics-exported models were verified running on this
  exact board) -- see `docs/setup-q8b.md` for the full verification. No
  camera attached yet (the Nexigo webcam needs to move over from the
  Q6A) and no end-to-end pipeline test has been run there -- that's the
  actual remaining step, not toolchain support.

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
