# Storage footprint (stretch goal)

Stretch goal: keep the whole system -- OS, project code, dependencies, and
model weights -- small enough to comfortably fit on a **32GB eMMC/UFS**
module, since that's a realistic storage tier for a cheaper/more mainstream
version of this device. This is explicitly a stretch goal: correctness and
functionality come first, but every milestone's dependency/model choices
should default to the smaller option when it doesn't cost meaningful
capability.

## Measured baseline (Q6A, r2 image on UFS, 2026-09-11)

The dev Q6A has no onboard eMMC. It initially booted off a 64GB microSD
card as a quick way to get a fresh image running, but has since been
reflashed to Radxa's r2 stable image on the 128GB UFS module (`sda`,
117.9GB usable) -- see `docs/setup-q6a.md`. Neither is the 32GB target
this stretch goal is budgeting for, so treat the numbers below as a floor
measurement, not a target-hardware match.

```
Filesystem      Size  Used Avail
/dev/sda3       116G  4.7G  107G   (UFS, current boot media)
```

A stock, unmodified image already spends **4.7GB** before any project code,
Python deps, or models are added -- that's the floor to budget around, not
zero. It includes a full desktop-adjacent package set (X11-capable libs,
full `libopencv-*` C++ dev packages including modules this project will
never use like `viz`, `stitching`, `superres`, GStreamer plugins, etc.).

## Budget

Target: OS + project + deps + models under **~10GB total**, leaving >20GB
of headroom on a 32GB module for OS updates, logs, and margin. Concretely:

| Component | Budget | Notes |
|---|---|---|
| Base OS image | ~4-6GB | Use a minimal/server image variant (no desktop environment) if Radxa offers one for the Q6A -- a desktop image carries GUI/X11 packages this headless pipeline never uses. |
| Python + project deps | <500MB | See package choices below. |
| Model weights (QNN) | <200MB | Quantized (INT8) context binaries for the detector + pose model; AI Hub typically offers quantized variants alongside FP32. |
| Logs/data/margin | remainder | Deliberately left unbudgeted rather than assumed away. |

## `onnxruntime-qnn` footprint (2026-09-11) -- over budget, needed anyway

`onnxruntime` (59MB) + `onnxruntime-qnn` (202MB) = **~261MB** installed via
pip on-device -- already over half the entire `<500MB` Python+deps budget
above, from two packages. This isn't optional: it's what replaced a
subprocess-per-frame `qnn-net-run` approach that took ~476ms/frame with a
persistent-session approach that takes ~3ms/frame (see
`scripts/qnn_spike.md`) -- correctness *and* the real-time requirement
both depend on it.

`onnxruntime-qnn` bundles Hexagon skeleton libraries for HTP versions V68,
V69, V73, V75, V79, and V81; the Q6A's QCS6490 only needs V68. Pruning the
unused skel/stub `.so` files (`libQnnHtpV69*`, `V73*`, `V75*`, `V79*`,
`V81*`) after confirming which files are actually load-bearing could
recover a meaningful chunk of that 202MB -- not done yet, tracked in
`docs/backlog.md`. The `<500MB` budget above should be treated as
aspirational until this is resolved, not revised upward reflexively.

## Concrete package choices to keep footprint down

- **Don't pip-install `opencv-python`/`opencv-contrib-python` on-device.**
  Each bundles a full static OpenCV build (tens of MB) that duplicates the
  `libopencv-*` C++ libraries the base image already ships. Confirmed on
  the Q6A: `libopencv-core406t64`, `-imgproc406t64`, `-videoio406t64`, etc.
  are already present system-wide. Prefer `sudo apt-get install
  python3-opencv`, which dynamically links against those existing shared
  libraries instead of shipping its own copy.
- **Don't install the `dev` extra (mediapipe, matplotlib, pytest) on the
  Q6A.** `mediapipe` is the laptop-only CPU dev backend (see
  `inference/backends/cpu.py`) -- the device build should only ever pull in
  what `inference/backends/qnn.py` actually needs at runtime (a thin QNN/
  onnxruntime-qnn runtime, not a full ML framework). `pyproject.toml`'s
  `qnn` extra should stay minimal for exactly this reason as it's filled in.
- **Never install PyTorch/TensorFlow on-device.** Those are only needed on
  a dev machine to export/quantize models through Qualcomm AI Hub's compile
  job; the Q6A itself only needs the resulting compiled QNN artifact plus a
  thin runtime.
- **Use `pip install --no-cache-dir` and `apt-get clean` after installs** on
  the device image so package caches don't sit on the boot media permanently.
- **Prefer quantized (INT8) model exports over FP32** once accuracy is
  validated -- typically a 4x size reduction with the Hexagon NPU's own
  quantized execution path, not just a storage saving.

## Tracking

Re-run `df -h /` on the device after each milestone that adds a new
dependency or model, and note the delta here or in the milestone's PR
description, so growth is visible incrementally rather than discovered as a
surprise at the end.
