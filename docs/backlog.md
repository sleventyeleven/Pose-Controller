# Backlog / deferred items

Things intentionally deferred out of the current milestone work, so they
don't get lost.

## Tooling

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
