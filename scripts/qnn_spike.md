# NPU spike plan (milestone 1)

Goal: get *any* model running through the Q6A's Hexagon NPU from Python,
before implementing `inference/backends/qnn.py` for real. This is the
highest-risk unknown in the project (per `docs/hardware.md`) and needs to
happen on the physical board.

Suggested steps (to run on the Q6A itself, e.g. over SSH):

1. Confirm the board's OS image and Python version (`cat /etc/os-release`,
   `python3 --version`).
2. Install `qai-hub-models` and check whether it can target QCS6490 directly,
   or whether models need to be compiled via the Qualcomm AI Hub cloud
   service first (`pip install qai-hub qai-hub-models`, then
   `qai-hub configure` and check `qai_hub.get_devices()` for a QCS6490
   entry).
3. Export/compile a small model (start with a person detector, e.g.
   `qai_hub_models.models.yolov8_det`) targeting QCS6490 and download the
   resulting QNN context binary / `.so`.
4. Determine the runtime: try `onnxruntime` with the QNN execution provider
   first (`pip install onnxruntime-qnn` if it exists for this platform);
   fall back to AI Hub's own on-device inference harness
   (`qai_hub_models.models.<model>.demo` or `App` classes) if not.
5. Run inference on a static test image and confirm output shape/values are
   sane and that the NPU (not the CPU fallback) is actually being used --
   check `qcom` diagnostics or process/power monitoring during inference to
   distinguish NPU from CPU execution.
6. Repeat for a pose model (`qai_hub_models.models.mediapipe_pose` or
   similar) once the detector round-trip works.

Once this works end to end, port the working incantation into
`inference/backends/qnn.py`, matching the `PoseEstimator` interface in
`inference/pose.py`.

This step requires the physical Q6A and is not something that can be done
from a plain dev machine -- run it and report back what worked so the
`qnn.py` stub can be filled in.

## Progress log (2026-09-11)

**DSP/NPU hardware bring-up: done, confirmed working.** `fastrpc_test
-a v68` passes all 3 test modules on the physical Q6A after reflashing to
the r2 stable image, updating BIOS firmware to build `260120`+, and
installing `fastrpc fastrpc-test fastrpc-dev libcdsprpc1
radxa-firmware-qcs6490`. Full trail (including the `noble-test` image
dead ends that led to the reflash) is in `docs/setup-q6a.md`.

This answers steps 1-2 above: no manual QNN SDK download needed --
Radxa's stable apt channel ships everything (`fastrpc*`,
`radxa-firmware-qcs6490`), and separately Qualcomm's `qairt-*` packages
provide the QNN CLI tools (`qnn-net-run`, `qnn-context-binary-generator`)
and `libQnnHtp.so`. `docs/setup-q6a.md` section 4 also has the exact
Qualcomm AI Hub Models workflow (steps 3-4 above) with the confirmed
model list for this chipset, including `mediapipe_pose`.

**Steps 3-6: done, spike complete.** Ran the full `qai_hub_models` export
for `mediapipe_pose` (chipset `qcs6490`, `--target-runtime
qnn_context_binary --quantize w8a8`) via Qualcomm AI Hub's cloud pipeline.
Both sub-models (`pose_detector`, `pose_landmark_detector`) went through
compile -> quantize -> compile -> link -> profile -> inference, all
`SUCCESS`, profiled on a real QCS6490 device in Qualcomm's fleet:

| Model | Inference time | Peak memory |
|---|---|---|
| pose_detector | 1.62 ms | ~4.4 MB |
| pose_landmark_detector | 1.20 ms | ~4.7 MB |

Combined ~2.8ms/person/frame -- enormous headroom for a 15-30fps target.
Downloaded the compiled QNN context binaries (`pose_detector.bin` 3.54MB,
`pose_landmark_detector.bin` 6.80MB -- comfortably within the
`docs/storage-footprint.md` model-weight budget), copied them to the
physical Q6A, and ran the landmark model locally with `qnn-net-run
--retrieve_context=pose_landmark_detector.bin --backend=/usr/lib/libQnnHtp.so`
(with `ADSP_LIBRARY_PATH` pointed at the RB3gen2 skel dir, same as
`docs/setup-q6a.md`). It loaded the context and began graph execution on
the real NPU -- the only error was "Graph input info vector is empty"
(expected: no `--input_list` was supplied). **This confirms the compiled
model runs on this physical board's Hexagon NPU end to end.**

## `inference/backends/qnn.py` implementation (2026-09-11)

Fully wired up and validated end-to-end on the physical Q6A:

- `_qnn_runtime.py`: `QnnContextRunner` shells out to `qnn-net-run
--retrieve_context=... --backend=/usr/lib/libQnnHtp.so` per call, writing
  input as a raw float32 file and an `--input_list` referencing it, and
  reading back the named output `.raw` files from `Result_0/`. Confirmed
  empirically that qnn-net-run reads/writes plain float32 regardless of
  the model's underlying quantized dtype (no manual dequant needed).
- `_blazepose.py`: a from-scratch pure numpy/opencv port of the BlazePose
  two-stage detector+landmark pipeline (resize/pad, anchor decode, top-1
  detection, ROI computation, affine crop, inverse-transform landmarks
  back to frame space) -- ported from `qai_hub_models`' reference
  implementation rather than re-derived, and validated against it
  directly: fed the real on-device detector/landmark outputs into
  `qai_hub_models`' own reference decode functions and compared against
  this port's output on the bundled demo photo (`mediapipe_pose`'s
  `pose.jpeg`), landing within ~2.5px mean / ~7px max on a 3088x2316
  image once two bugs were found and fixed:
  1. **Wrong ROI formula** -- initially ported the generic
     `MediaPipeApp._compute_object_roi` (derives ROI from the detector's
     box), missing that `MediaPipePoseApp` *overrides* it to derive the
     ROI purely from two of the detector's auxiliary keypoints instead.
     Caused ~300px average landmark error.
  2. **Wrong input tensor layout** -- assumed NCHW (matching the original
     PyTorch model's `get_input_spec()`), but the actual AI-Hub-compiled
     context binaries expect **NHWC** (`[1,128,128,3]` /
     `[1,256,256,3]`), confirmed via `qnn-context-binary-utility
     --context_binary=... --json_file=...`. Fed with the wrong layout,
     the detector produced a suspicious flat max score of exactly 0.5 (a
     tell -- multiple anchors tied at exactly sigmoid(0)) regardless of
     input image; fixing the layout took the top score from 0.5 to 0.99
     on a real photo. **Lesson: don't trust the original model's
     `get_input_spec()` for a compiled/exported artifact's actual tensor
     layout -- verify against the compiled binary directly.**

Full end-to-end test (`QnnPoseEstimator.estimate()` on `mediapipe_pose`'s
own demo photo, run on the physical Q6A): 25 keypoints returned with
plausible, correctly-proportioned positions (shoulders/wrists/hips all
symmetric and in sensible relative positions), closely matching the fp32
reference pipeline's positions for the same image.

**Known caveat -- per-frame latency, not per-model latency.** AI Hub's
1.2-1.6ms/model profiling number is the NPU execution time alone.
`qnn-net-run` reloads the full context binary from scratch on every
invocation (matches the ~170-190ms warm-load time also seen in AI Hub's
profiling); since `QnnContextRunner.run()` shells out fresh per call, one
`estimate()` call (detector + landmark, 2 subprocess launches) measured
**~476ms** end-to-end on-device -- ~2fps, not the 15-30fps target.
Resolved below.

## Latency fix: onnxruntime + onnxruntime-qnn (2026-09-11)

Replaced the `qnn-net-run`-subprocess-per-call runtime entirely with
`onnxruntime`'s QNN execution provider, loading each model once into a
persistent `InferenceSession` (`_qnn_runtime.OnnxQnnRunner`) instead of
reloading the context binary on every call.

**Getting a QNN-capable ONNX model.** The `qnn_context_binary` export
target used for the initial spike produces a raw `.bin` context binary
with no ONNX wrapper -- not loadable by onnxruntime directly. Re-exported
with `--target-runtime precompiled_qnn_onnx` instead:
```bash
python -m qai_hub_models.models.mediapipe_pose.export \
  --chipset qcs6490 --target-runtime precompiled_qnn_onnx --precision w8a8 \
  --components pose_detector pose_landmark_detector
```
(Passing `--quantized-model-id <job_ids>` to reuse the earlier
quantization and skip re-downloading the 8GB calibration dataset failed
with "Model ID could not be found" -- those IDs are *quantize job* IDs,
not the resulting *model* IDs; not worth chasing further since the
dataset was already cached locally by the first export, making the
re-run fast anyway.) Output is a `.zip` per component containing
`model.onnx` (a thin graph wrapping one `EPContext` node) + `model.bin`
(the actual QNN context binary as external data) -- both files must ship
together in the same directory; see `models/mediapipe_pose_qcs6490/`.

**Finding `onnxruntime-qnn`.** Plain `pip index versions onnxruntime`
exists for this platform but only ships `CPUExecutionProvider`/
`AzureExecutionProvider` -- QNN support ships in a **separate** PyPI
package, `onnxruntime-qnn` (confirmed via `pip3 index versions
onnxruntime-qnn`), which bundles `libonnxruntime_providers_qnn.so` plus
the QNN backend libraries themselves (`libQnnHtp.so`, per-HTP-version
skel/stub libraries) inside the Python package directory --
`onnxruntime_qnn.get_library_path()` / `get_qnn_htp_path()` give their
paths.

**Two non-obvious onnxruntime API details** (this is a newer "plugin
execution provider" mechanism, not the older statically-compiled-provider
API most onnxruntime docs describe):
1. `InferenceSession(path, providers=[("QNNExecutionProvider", opts)])`
   does **not** work for a dynamically-registered plugin EP -- it only
   recognizes providers compiled into the base `onnxruntime` build,
   failing with "EPContext node ... not compatible with any execution
   provider added to the session. Available: [CPUExecutionProvider]".
   The working pattern:
   ```python
   ort.register_execution_provider_library("QNNExecutionProvider", oq.get_library_path())
   qnn_devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
   so = ort.SessionOptions()
   so.add_provider_for_devices(qnn_devices, {"backend_path": oq.get_qnn_htp_path()})
   session = ort.InferenceSession(path, sess_options=so, providers=[])
   ```
2. Don't set `ADSP_LIBRARY_PATH` externally. Doing so (e.g. reusing the
   `qairt-tools`-based value from the `qnn-net-run` days) produced
   `QNN_DEVICE_ERROR_INVALID_CONFIG: Invalid config values` and a warning:
   *"Using existing ADSP_LIBRARY_PATH setting ..., which may cause the
   HTP backend to fail."* `onnxruntime_qnn` wants to point this at its
   **own bundled** skel libraries (inside the `onnxruntime_qnn` package
   dir, not `qairt-tools`') and sets it itself when the env var isn't
   already set -- leave it unset for this runtime path.

**I/O is raw quantized `uint8`, not auto-handled float.** Unlike
`qnn-net-run` (which transparently quantizes/dequantizes raw float32 I/O
files), the ONNX graph's declared tensor types are literally `uint8` --
`session.run()` requires quantized input and returns quantized output.
ONNX has no standard place to carry scale/offset for a bare tensor
outside explicit QuantizeLinear/DequantizeLinear nodes (this whole model
is one opaque `EPContext` node), so those values were extracted once via
`qnn-context-binary-utility --context_binary=model.bin
--json_file=info.json` and hardcoded as `QuantSpec` constants in
`inference/backends/qnn.py` (`real = (raw + offset) * scale`).

**Result:** ~2.2ms (detector) / ~1.1ms (landmark) per warm `session.run()`
call on the physical Q6A -- matching AI Hub's own NPU-only profiling
almost exactly. Full `estimate()` on the 3088x2316 demo photo: **~26ms**
(down from ~476ms), with session load (~160ms + ~55ms) paid once at
`QnnPoseEstimator.__init__()`, not per frame. Verified the resulting
keypoints are unchanged (correctness carried over from the `qnn-net-run`
validation) -- same shoulder/wrist/hip positions within quantization
noise.

**Trade-off, not free:** `onnxruntime` + `onnxruntime-qnn` together are
~261MB installed -- a real cost against `docs/storage-footprint.md`'s
budget, with a known pruning opportunity (unused Hexagon HTP versions
bundled in) tracked in `docs/backlog.md`.
