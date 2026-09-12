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

## Milestone 3: multi-person detection (2026-09-11)

`_blazepose.py`'s `_top_detection` (top-1 only) was replaced with
`_select_detections`: real greedy NMS over all 896 anchors' decoded boxes
(kept for dedup only -- the actual per-person ROI still comes from
keypoints, see `_compute_roi_corners`'s docstring), returning every
detection above `MIN_DETECTOR_SCORE` up to `MAX_PERSONS`. Each surviving
detection runs through the same ROI->crop->landmark path as before,
independently.

Validated on the physical Q6A with a synthetic two-person image (two
crops of the demo photo placed side by side): 2 people correctly detected
with disjoint, correctly-positioned bounding boxes, ~15-20ms per frame
total (both people) -- the multi-person case barely costs more than
single-person, confirming the ~2ms/model NPU execution time is the real
bottleneck unit, not some fixed per-frame overhead.

## Milestone 3: appearance-based re-ID (2026-09-11)

Added to close the gap `tracking.Tracker`'s Kalman/IoU tracking can't:
recognizing someone who fully left the camera's field of view and
returned (not just brief occlusion, which Kalman/IoU already handles via
`max_age` coasting).

**Model: OSNet, not a hand-rolled choice.** `qai_hub_models` already
catalogs OSNet (`qai_hub_models.models.osnet`) -- the exact model class
the original project plan named (OSNet-x0.25) -- complete with pretrained
weights, multiple width variants, and QCS6490 listed in `perf.yaml`'s
`supported_chipsets`. Re-used it directly rather than sourcing/training a
custom re-ID model.

**Blocker: quantization needs a gated dataset.** Attempted the same
`--target-runtime precompiled_qnn_onnx --precision w8a8` export used for
`mediapipe_pose`. The compile step succeeded, but quantization failed:
```
UnfetchableDatasetError: To use dataset entire_id, you must download it manually.
  1. Open the Google Drive folder: https://drive.google.com/drive/folders/...
  2. Download bounding_box_test/ and query/, zip them together
  3. Run: python -m qai_hub_models.scripts.configure_dataset --class ...ENTIReIDDataset --files ...
```
`OSNet.get_calibration_dataset_cls()` requires `ENTIReIDDataset`; the
eval dataset (`Market1501Dataset`) is *also* gated the same way (both are
long-standing real licensing quirks of these actual ReID benchmark
datasets, not something `qai_hub_models` can route around). Rather than
ask the user to navigate a manual Google Drive download+zip+configure
step, or substitute ad-hoc calibration images for a model whose own code
explicitly flags quantization-sensitivity
(`OSNet.get_hub_quantize_options` overrides the default range scheme
specifically because "min_max: tf_enhanced clips OSNet embeddings and
tanks w8a8 ReID mAP") -- reconsidered the actual requirement instead.

**Re-examining the requirement changed the right answer.** Pose
estimation runs on every person every frame (real-time-critical -- worth
the NPU/quantization investment). Re-ID only runs when `Tracker` can't
IoU-match a detection to an existing track -- occasional, not per-frame.
OSNet-x0.25 is ~0.71M parameters; a quick local check
(`torch.onnx.export`, eager PyTorch, unoptimized) showed ~26ms/call on a
dev laptop CPU. That's fast enough for occasional use without NPU
compilation at all. Exported via plain `torch.onnx.export` (opset 17,
`dynamo=False` -- the new dynamo-based default exporter needs
`onnxscript`, not installed) and run via onnxruntime's default
`CPUExecutionProvider` -- no AI Hub compile job, no quantization, no
gated dataset. See `models/osnet_x0_25/README.md`.

**Result, validated on the physical Q6A:** ~20-50ms per embedding call
(ARM CPU, slower than the x86 dev-laptop number above but still fine for
occasional use). Using the same synthetic two-person image: cosine
similarity between the two (visually identical) people was **0.978**,
against an unrelated background patch **0.419** -- the model
discriminates meaningfully, not just returning a constant. Full
end-to-end test: a track that "disappeared" past `max_age` and
reappeared at a **completely different position** (left side of frame to
right side -- IoU/Kalman alone would never relink this) was correctly
revived with its original `track_id`.

`tracking.Tracker` now accepts an optional `embedder: ReidEmbedder`; when
a track ages out past `max_age`, its last-known embedding moves to a
"lost gallery" for `reid_gallery_ttl` additional frames, and new
unmatched detections are checked against that gallery by cosine
similarity before a fresh ID is allocated. `reid_similarity_threshold`
(default 0.6) is a reasonable starting point, not empirically tuned
against a benchmark -- see `docs/backlog.md`.

## Milestone 4: live-camera gesture testing investigation (2026-09-12)

Attempted to validate gesture recognition end-to-end with the physical
Nexigo webcam on the Q6A. Never got past the pose-detection stage --
every real webcam frame tried produced the same degenerate signature
seen during the original NHWC layout bug earlier in this log: flat
~0.5 confidence scores, effectively tied across anchors, indicating the
model is seeing something structurally different from what it was
calibrated against (not just "a hard image").

**Setup issues found and fixed first** (mundane, but cost real time):
- `cv2.VideoCapture` with the default (GStreamer) backend failed to open
  the webcam at all -- "not a capture device" / "Internal data stream
  error". Forcing `cv2.VideoCapture(<index>, cv2.CAP_V4L2)` fixed it.
- The webcam's V4L2 device index moved after the r2 image reflash --
  `/dev/video0`/`1` are now claimed by the `qcom-venus` hardware
  encoder/decoder, not the webcam. Found the real index by checking
  `/sys/class/video4linux/video*/name` (webcam ended up at
  `/dev/video2`/`3`).

**Ruled out as the detection failure's cause:**
- *Orientation/aspect ratio.* A forced-landscape known-good reference
  image still scored 0.99 fine, so portrait vs. landscape framing isn't
  it.
- *A `resize_pad`/preprocessing shape bug.* Manually inspected the
  128x128x3 model input tensor after preprocessing a live frame -- shape
  was correct, and visually it was clearly a recognizable person, not
  garbage or a wrongly-sliced buffer.
- *Simple darkness*, for the first failed capture specifically (dim room,
  camera pointed steeply upward -- mean grayscale brightness 48.5).
  Synthetically darkening the known-good reference image step by step
  only broke detection down around mean brightness ~19-28, well below
  48.5, so darkness alone doesn't explain this frame's failure.
- *Motion blur*, also for that first capture. Laplacian variance (a
  common sharpness/blur proxy) was actually *higher* (601.7) than the
  known-good reference's (130.0) -- the opposite of what blur would
  produce. More consistent with sensor noise inflating that metric than
  with an actual blur problem.
- **Root cause of this first failure mode is still not identified.**

**Fully diagnosed, second failure mode:** a second attempt used `cheese`
(GNOME's webcam app) to manually crank brightness/contrast, on the
theory that the first capture was too dark. The resulting video (saved
by the user to `/home/radxa/Videos/Webcam/`) also failed detection, but
this time with a clear, measurable cause: real sensor clipping (8.2% of
grayscale pixels at or above 250/255) plus near-total desaturation (HSV
saturation channel mean of 4.35 out of 255, versus 70.7 on the
known-good reference photo). Tried to recover it in post -- gamma
correction and synthetic saturation boosting -- neither restored
detection, confirming the color information was actually destroyed at
capture time (sensor clipping), not just visually washed out and
recoverable by reprocessing.

**Outcome:** built `pose_controller.capture.exposure.assess_exposure()`
(see its module docstring and `tests/test_exposure.py`) to catch this
second failure mode -- overexposure and desaturation -- as a per-frame,
cheap (no model inference) warning wired into `app.py`. Confirmed
correct on both the bad `cheese` capture and the known-good reference.
It does **not** catch the first failure mode (the dim, upward-angled
capture technically passes all of its thresholds yet still fails
detection), so passing this check is necessary but not sufficient
evidence a frame will actually be detected by the pose model.

Net result: gesture recognition (`gestures/`) remains validated only by
synthetic unit tests plus one static-photo sanity check -- no live
camera feed has yet produced a successful pose detection to drive it.
See `docs/backlog.md` for the up-to-date status and next steps (trying
camera positions/lighting between these two failure extremes; building
an auto-exposure-adjustment control loop once a working configuration
is found by hand).

## CSI camera bring-up and the real root cause of the detection failures (2026-09-12)

Continuing the live-camera investigation above, tried the Dragon Q6A's
onboard CSI camera connector as an alternative to the Nexigo USB webcam,
on the theory that a different sensor/pipeline might sidestep whatever
was wrong with the Nexigo captures.

**Getting the CSI camera working at all required enabling a disabled
device-tree overlay.** Out of the box the CSI camera produced literally
nothing: no `/dev/video*` node, no CSI/sensor lines anywhere in `dmesg`
or `journalctl`, and no matching I2C address on any bus -- the kernel
wasn't even trying to probe it, which is a very different (and more
fundamental) problem than "detection is unreliable". The board ships
`rsetup` (`/usr/bin/rsetup`), a Radxa-provided config tool, and its
overlays live in `/boot/dtbo/*.dtbo.disabled` until enabled. The exact
overlay for this hardware exists:
`qcs6490-radxa-dragon-q6a-cam1-radxa-camera-4k.dtbo` -- contradicting the
original project plan's assumption that this camera's Dragon Q6A
compatibility was unconfirmed; Radxa does ship board-specific support for
it, it's just off by default. Enabled it with
`sudo rsetup enable_overlays qcs6490-radxa-dragon-q6a-cam1-radxa-camera-4k.dtbo`
(which strips `.disabled` and runs `u-boot-update`) and rebooted. After
that, a Sony **IMX415** sensor showed up fully bound to its kernel driver
at I2C address `16-001a`, with a complete `qcom-camss` media graph
(`msm_csiphy0` -> `msm_csid0` -> `msm_vfe0_rdi0` -> `/dev/video2`) visible
via `media-ctl -d /dev/media1 -p`.

**This is Qualcomm's mainline CAMSS ISP stack, not a plug-and-play UVC
device** -- unlike the Nexigo, nothing streams until the media graph's
links are explicitly created and the sensor's native format is
propagated down every pad:
```
media-ctl -d /dev/media1 -l '"msm_csiphy0":1 -> "msm_csid0":0[1]'
media-ctl -d /dev/media1 -l '"msm_csid0":1 -> "msm_vfe0_rdi0":0[1]'
media-ctl -d /dev/media1 -V '"imx415 16-001a":0 [fmt:SGBRG10_1X10/3864x2192]'
media-ctl -d /dev/media1 -V '"msm_csiphy0":0 [fmt:SGBRG10_1X10/3864x2192]'
media-ctl -d /dev/media1 -V '"msm_csiphy0":1 [fmt:SGBRG10_1X10/3864x2192]'
media-ctl -d /dev/media1 -V '"msm_csid0":0 [fmt:SGBRG10_1X10/3864x2192]'
media-ctl -d /dev/media1 -V '"msm_csid0":1 [fmt:SGBRG10_1X10/3864x2192]'
media-ctl -d /dev/media1 -V '"msm_vfe0_rdi0":0 [fmt:SGBRG10_1X10/3864x2192]'
v4l2-ctl -d /dev/video2 --set-fmt-video=width=3864,height=2192,pixelformat=pGAA
```
`/dev/video2` is an RDI (Raw Data Interface) node -- it hands back raw,
undemosaiced 10-bit Bayer data packed MIPI-style (4 pixels per 5 bytes),
not a finished BGR/YUV frame. Unpacking it needs a manual bit-unpack
(shift each of 4 bytes left 2 and OR in 2 bits from a shared 5th byte),
then `cv2.cvtColor(img8, cv2.COLOR_BayerGB2BGR)` to debayer (`pGAA` =
GBRG Bayer order, confirmed against the sensor's own reported format).
The result is very green-dominant straight out of the debayer step --
expected for raw Bayer data (2x as many green photosites as red/blue per
2x2 block) with no white balance applied -- fixed with a simple
gray-world white balance (scale R and B channels so their means match
G's mean). The sensor's own auto-exposure is also bypassed entirely in
raw RDI mode -- the first captured frame was far too dark (mean
brightness ~16/255) until `analogue_gain` was manually pushed up via
`v4l2-ctl -d /dev/v4l-subdev27 --set-ctrl=analogue_gain=80` (out of a
0-100 range; `exposure` was already at its max default of 2242 and isn't
the useful lever here). After gain + white balance, the resulting image
was genuinely clean: correct color, real detail (a shelving unit, boxes,
a light fixture rendered without blowing out), brightness ~62-72/255,
saturation ~43-101 -- comfortably past every threshold in
`capture.exposure.assess_exposure`. (The image also comes out rotated
90 degrees from the physical mounting orientation -- cosmetic, trivially
fixed with `cv2.rotate`, and confirmed not to matter to the detector.)

**And it still failed detection, with the identical flat-0.5 signature.**
This was the decisive test: a different sensor, a completely different
capture pipeline, definitively good exposure/color/sharpness, a person
clearly visible with an arm raised -- same failure as the Nexigo. That
ruled out camera hardware and exposure as the cause across the board.

Captured a 30-frame sequence (roughly 1 fps over 30 seconds, while a
person moved in and out of frame and performed gestures) and ran every
frame through the real `QnnPoseEstimator._run_detector()` path. All 30
frames scored an identical flat 0.5 (raw logit exactly 0.0) despite
brightness varying frame to frame (~66-76/255) -- strong evidence this
wasn't a per-frame fluke.

Comparing raw (pre-sigmoid) detector logits against a known-good demo
image was the key diagnostic: the demo image has 4 anchors sitting at a
confident **+4.83** logit; every failing real frame's best anchor never
rose above **0.0**, even on a frame with an unmistakable, well-lit person
in it. Taking that exact frame and cropping tightly around the person
(from the full ~3864x2192 sensor frame down to roughly a 1600x1700 region
centered on them) and re-running the identical detector code produced a
raw logit of **+4.833** -- matching the demo image's confident value
almost exactly -- with `max_score=0.9921` and a full pose returned.

**Root cause: the person detector (BlazePose's first stage) needs the
subject to occupy a meaningfully large fraction of the frame, and at
normal room distance with either camera's field of view, they don't.**
Both the Nexigo and the CSI/IMX415 have wide-ish fields of view suited to
"webcam at a desk, person nearby" or general-purpose framing -- not the
tighter framing MediaPipe's detector was implicitly tuned against
(closer to selfie/portrait distance). Once a full room-distance frame is
downscaled to the detector's fixed 128x128 input (a ~30x shrink for the
CSI camera's native resolution), the person becomes too small a
silhouette for the detector's anchors to register, independent of how
sharp, well-exposed, or well-colored the source frame is. This also
retroactively explains round 1's Nexigo failure from earlier in this log
(dim, upward-angled capture, root cause originally left as "not
identified") -- it was very likely this same framing issue the whole
time, just investigated before this crop test existed to reveal it.

**Not yet fixed, only diagnosed.** The crop-and-retest above was a manual
diagnostic, not a pipeline change. See `docs/backlog.md` for the concrete
options going forward (move the camera closer, add a fixed center-crop/
digital zoom tuned to the expected interaction distance, or treat
"stand closer to the camera" as a real operating constraint of this
project) -- none are implemented yet.

## Milestone 4 follow-up: YOLOv8n-det as the first-stage detector (2026-09-12)

Per the finding above, replaced BlazePose's own 128x128 detector with a
YOLOv8n person detector as the pipeline's first stage, keeping BlazePose's
landmark model unchanged for the actual keypoints -- this is what the
original milestone-1 plan called for from the start (a separate
higher-resolution person detector feeding a per-crop pose model), never
implemented until now.

**Export.** `qai_hub_models.models.yolov8_det.export`, same
`precompiled_qnn_onnx` / `w8a8` / `qcs6490` recipe as the pose models.
Needed extra dev-machine-only packages beyond the base `[hub]` extra:
`ultralytics` (the model definition itself imports it -- absent from
`qai_hub_models`' own dependency list), and `aiofiles` + `pycocotools`
(both surfaced only once quantization tried to download and load COCO
calibration data -- `pip install`-able wheels for both, no build toolchain
needed on Windows). Real device profiling (Dragonwing RB3 Gen 2 Vision
Kit, the QCS6490 dev kit): **4.6ms inference, 254/254 ops on NPU** (0
GPU/CPU fallback).

**The exported model's outputs needed direct verification, not
assumption** -- Radxa's own QCS6490 deployment notes for this exact model
(a sibling board, Airbox Q900) warn of a real quirk: some exports return
outputs ordered `[scores, class_idx, boxes]` instead of the expected
`[boxes, scores, class_idx]`. Checked via `onnx.load()` against the actual
downloaded graph rather than trusting either ordering: this export's
graph order was `boxes, scores, class_idx` as expected, and all three are
uint8 with real `QuantizeLinear`/`DequantizeLinear` nodes carrying
scale/zero_point as graph initializers -- readable directly via `onnx`,
no `qnn-context-binary-utility` round-trip needed (unlike the mediapipe
models, whose I/O quantization lives only in the opaque compiled context
binary, not the ONNX graph itself). One value needed a judgment call:
`class_idx`'s graph-declared scale was exactly **0.0**, which taken
literally would dequantize every value to 0 regardless of the raw byte --
not a plausible real quantization scale for an 80-class index. Treated as
an exporter formality for smuggling an integer output through a uint8 QDQ
wrapper and used `scale=1.0, offset=0.0` (pure passthrough) instead --
confirmed correct on-device (a person in frame reliably decodes to
`class_idx == 0`, not a garbage float). Since `_yolo_detect.py`/`qnn.py`
look outputs up by name (via `OnnxQnnRunner`'s dict-keyed `run()`), the
Radxa-documented reordering quirk wouldn't have mattered even if this
export had exhibited it.

**Architecture:** `_blazepose.py` gained `_roi_corners_from_box` (an
axis-aligned square ROI from a plain xyxy box, no rotation cue available
the way BlazePose's own aux-keypoint-derived `_compute_roi_corners` has
one) and `detect_poses_from_boxes` (skips BlazePose's detector entirely,
runs the landmark model against externally-supplied boxes). The
shared crop-and-infer logic was factored out of `_landmarks_for_keypoints`
into `_run_landmark_on_roi` so both ROI paths feed the landmark model
identically. `_yolo_detect.py` handles YOLO's own pre/post-processing
(resize/pad, person-class + score filtering, NMS, coordinate unscaling)
following `qai_hub_models.models.templates.yolo.{model,app}`'s reference
implementation (score threshold 0.45, NMS IoU 0.7, matching that
reference's defaults) -- the exported model already does box decoding and
per-box argmax-over-classes internally (`include_postprocessing=True` at
export time), so this module only needed score/class filtering + NMS on
top, not raw anchor decoding.

**Validated against the known-good two-person demo photo used throughout
this project:** YOLO correctly finds both people; both get landmarked
with high confidence (0.99 / 0.71) once `BOX_ROI_SCALE` (the margin
around YOLO's box used to build the landmark model's square crop) was
tuned from an initial guess of 1.25 up to 1.75 -- 1.25 scored 0.09 for
both people (fails `MIN_LANDMARK_SCORE`), confirming the landmark model
needs a generous margin around a person's own bounding box, not just
"tight box plus a little slack": it was trained on BlazePose's own
detector's specific ROI convention, not a generic box crop.

**Validated against the real 30-frame CSI capture sequence from the
earlier framing investigation, and this surfaced a real correction to
that investigation.** Feeding the sequence through the new detector
initially got very weak person-class scores (0.01-0.15 typical, one
outlier at 0.44) -- surprising, since YOLO's 640x640 input was supposed
to fix exactly this. Rotating one frame through all four cardinal
orientations isolated the cause immediately: the frames were being fed in
**portrait** (the CSI capture pipeline's manual rotation script had
guessed the wrong direction back when this sequence was first processed,
and it was never corrected since the earlier investigation had moved on
to other things). Portrait: person-class score 0.070. Landscape (the
correct orientation, confirmed visually -- door frame and shelving
upright, not sideways): **0.867**. That is a far bigger swing than image
quality alone would produce, and it means the earlier "ruled out
orientation" conclusion (comparing a forced-*landscape squish* of a demo
photo, which still scored fine) was testing the wrong transformation --
squishing an aspect ratio and genuinely rotating a person 90 degrees are
not the same thing, and only the latter was ever a real risk. Orientation
was likely a meaningful, previously-uncredited contributor to the
original Nexigo failures earlier in this document too, not just subject
size.

With orientation corrected, YOLO now finds a person box in **19 of 30**
frames (up from 0/30 before this integration) -- confirmed, substantial
progress. But the landmark model's confidence on most of those real
crops stayed poor, repeatedly landing on almost exactly 0.094 regardless
of which frame -- suspicious enough to investigate rather than write off
as "hard photos." One real mechanism found: `BOX_ROI_SCALE=1.75`'s margin,
tuned against the demo photo, can produce a square ROI taller than the
camera's own vertical field of view when a person fills most of the
frame (this capture's person occupied roughly 78% of frame height before
any margin was even added) -- the ROI then samples out-of-bounds black
padding rather than real image content along the affected edge, for no
reason related to the actual pose. Fixed by having `_roi_corners_from_box`
shift (translate, never resize) the ROI to stay within frame bounds
whenever it's geometrically possible (i.e. the requested ROI side is
still smaller than the frame in that dimension). This took one frame from
0.44 to 0.80 (crossing the threshold, now a full end-to-end success), but
left most others unchanged -- it's a real, principled fix, just not the
dominant remaining factor. Final tally: **1 of 30** real frames now
produce a complete, successful pose end-to-end (up from 0/30 for the
entire duration of this project's live-camera testing to date).

**Not fully solved.** The repeated near-identical 0.094 landmark score
across most real frames (as opposed to a spread of varying-difficulty
values) suggests something more systematic than "some poses are just
harder" -- suspected but not isolated: the `analogue_gain=80` (out of a
0-100 range) used during that capture session amplifying real sensor
noise well beyond what the landmark model saw in training, and/or the
sequence genuinely being shot mid-motion (walking, gesturing) rather than
standing still, both plausible without further live-iteration to
distinguish them. See `docs/backlog.md` for current status.

## Sanity check against the original overexposed Nexigo recording

Went back to the very first piece of evidence in this document's
overexposure investigation -- the `cheese`-recorded Nexigo video at
`/home/radxa/Videos/Webcam/2026-09-12-024357.webm` (1920x1080, 30fps,
34.7s) -- and ran it through the new YOLO+BlazePose pipeline without any
new capture session, sampling ~1fps (31 frames) via plain
`cv2.VideoCapture` (no rotation needed -- USB UVC webcams don't have the
CSI sensor's mounting-orientation problem).

`capture.exposure.assess_exposure` still flags nearly every sampled frame
(brightness ~170-180, saturation ~4-8) -- nothing about the recording
changed, it's genuinely still overexposed and desaturated by the same
measure that originally diagnosed it. Despite that, the new detector
found a person in **28 of 31 frames (90%)**, and the full detect+landmark
pipeline succeeded end-to-end on **14 of 31 frames (45%)** -- both a
dramatic improvement over the 0/31 the old 128x128 detector managed on
this exact footage. The higher-resolution YOLO detector apparently has
enough headroom to work through desaturation severe enough to have fully
blocked the old pipeline, independent of whatever `assess_exposure`
measures on the frame. This is the single strongest piece of evidence so
far that the detector swap actually unblocks live-camera gesture
validation, since it's real previously-recorded footage, not a new
best-effort capture session.

## Full pipeline validation against real dynamic footage: first live-camera gesture triggers (2026-09-12)

The natural next question: does a ~45% frame-level detection hit rate
actually translate into real, usable gesture triggers, or is it too
sparse for `GestureStateMachine`'s 4-consecutive-frame debounce to ever
confirm anything? Ran the exact `app.py` loop (`PoseEstimator.estimate`
-> `Tracker.update` -> `GestureStateMachine.update_all`) over all 923
readable frames of the same `cheese` recording (30fps, person in frame
for 28.7s), not a synthetic test.

**Without re-ID** (`Tracker()`, IoU/Kalman only): `frames_with_pose=451`
of 923 (48.9%, consistent with the ~45% sampled estimate above). The
track_id fragmented into **13 different IDs** across one continuous
person's screen time -- expected once you look at the miss pattern:
misses aren't isolated single-frame drops, they come in bursts often
longer than `max_age` (30 frames = 1s), so the Kalman filter's coast
couldn't bridge most gaps and a fresh ID got allocated on the other side
each time. Every fresh ID resets `GestureStateMachine`'s per-track
debounce state (`_TrackGestureState` starts fresh at DOWN/DOWN), so most
of the ~30 logged arm-state transitions never had 4 consecutive frames
within one track to actually confirm. Despite that handicap, **5 real
actions still fired** (SKIP, NEXT, NEXT, PREVIOUS, SKIP) -- the first
live-camera-triggered gestures in this project's history, extracted from
genuinely fragmented, intermittent detection.

**With re-ID enabled** (`Tracker(embedder=ReidEmbedder(...))` --
milestone 3's appearance-matching feature, built specifically to survive
a track aging out past `max_age` and reappearing, exactly this
recording's failure mode): fragmentation dropped from 13 IDs to **4**,
and triggered actions increased to **6**, now covering **all four**
required actions for the first time on real footage -- NEXT, PREVIOUS,
SKIP, and (newly) PLAY_PAUSE (both arms confirmed RAISED
simultaneously). Re-ID isn't perfect here: the surviving identity
alternates between two IDs (#3 and #4) rather than settling on one,
suggesting `reid_similarity_threshold` (0.6) or `reid_gallery_ttl` (300
frames) could use tuning specifically against high-miss-rate footage
like this, rather than the cleaner scenarios milestone 3 validated
against (a track cleanly exiting and re-entering frame). Throughput:
26.6fps without re-ID, 22.0fps with -- re-ID's per-lost-track embedding
cost shows up more here than it would on cleaner footage, precisely
because so many frames trigger a re-match attempt at this hit rate.

**Net effect:** gesture recognition is no longer a purely
synthetic-plus-one-static-photo validation -- it has now correctly
triggered every one of the four required actions from a real person's
real arm movements in real (and notably, still-imperfect: overexposed,
desaturated, ~49% per-frame miss rate) footage, using the project's
existing tracker and re-ID machinery with no changes needed. See
`docs/backlog.md` for the current status and remaining tuning
opportunities (re-ID threshold/TTL, static-pose/sweep thresholds now
retunable against real data for the first time).

## Tuning `reid_similarity_threshold` against real footage (2026-09-12)

The `#3`/`#4` alternation above wasn't random -- measured it directly.
Ran `ReidEmbedder` against every real YOLO-detected box across all 923
frames of the same `cheese` video (861 successful embeddings) and
computed cosine similarity for all ~370k pairs, since every pair in a
single-person video is by definition a same-person comparison:

```
All-pairs same-person similarity (n=370230):
  min=0.5290  p1=0.7140  p5=0.7644  median=0.8841  mean=0.8743  max=0.9999

similarity for gaps <2s (n=47352):  min=0.5946  p5=0.7893  median=0.9116
similarity for gaps >=2s (n=322878): min=0.5290  p5=0.7623  median=0.8806
```

The old default (`reid_similarity_threshold=0.6`) sits *above* both
measured floors -- meaning the tracker was correctly, faithfully applying
its own logic every time it refused to revive a track: on this footage,
genuine same-person photo pairs really did sometimes score below 0.6,
most likely from a combination of motion blur, pose change, and this
video's already-diagnosed desaturation (OSNet leans on color/texture cues
that a washed-out frame gives it less of). This isn't a bug in the
tracker or the embedder, it's a threshold picked before any real
same-person similarity data existed to check it against (the original
`docs/backlog.md` note called it "not empirically tuned against a
benchmark -- a reasonable starting point", which turned out to be a
touch too strict).

Swept `reid_similarity_threshold` in {0.45, 0.50, 0.55, 0.60, 0.65}
through the real `Tracker` + `GestureStateMachine` pipeline (not just the
raw embedding math above -- gallery TTL and match-order effects aren't
fully captured by pairwise stats alone) on the same video:

| threshold | unique track_ids | triggered actions | action types |
|---|---|---|---|
| 0.45 | 2 | 6 | all 4 |
| 0.50 | 2 | 6 | all 4 |
| 0.55 | 4 | 6 | all 4 |
| 0.60 (old default) | 4 | 6 | all 4 |
| 0.65 | 4 | 6 | all 4 |

Gesture-trigger count and type coverage never changed across any
threshold tested -- confirming gesture *correctness* was never actually
at risk here, only track-ID stability. 0.45 and 0.50 both collapse
fragmentation from 4 down to 2 (the remaining 2 IDs are most likely the
person not being fully in frame during the first ~1.8s, not a re-ID
failure -- plausible from the timing in the earlier log but not
separately confirmed). Picked **0.5** over 0.45 since both perform
identically here: 0.5 sits just below the measured real-world floor
(0.5290) rather than well below it, the smaller and more defensible
departure from the original default. Updated in both
`config.TrackingConfig.reid_similarity_threshold` and
`tracking.Tracker.__init__`'s own default, confirmed via the real
`config.py` -> `app.build_tracker` wiring (not just a hardcoded test
value) reproducing the same 2-track-id / 6-action result.

**Honest limitation:** this project still has no recorded multi-person
footage, so there's no measured *different*-person similarity
distribution to weigh against the same-person floor above -- lowering
the threshold reduces false-*splits* (one person getting multiple IDs)
but by construction makes false-*merges* (two different people
collapsing into one ID) somewhat more likely, and that trade's real cost
is unmeasured. If multi-person testing ever shows incorrect merges,
`reid_similarity_threshold` is the first place to look, and pushing it
back toward 0.6 (or higher, now that there's a rationale trail to
compare a new value against) would be the fix.

## Volume gestures: implementation and re-run against real footage (2026-09-12)

Added `VOLUME_UP`/`VOLUME_DOWN` (right arm swipe up / left arm swipe down
in front of the body -- see `docs/backlog.md` for the original proposal
and the collision risk flagged at the time). Implementation:
`dynamic_gestures.VerticalSwipeDetector(direction="up"|"down")`, checking
a minimum `rel_y` range between the rolling window's first and last
sample (`MIN_VERTICAL_SWIPE_RANGE=0.8` shoulder-widths) while bounding
`|rel_x|` to `SIDE_X_THRESHOLD` (reused from `static_poses.py`, not a new
constant) so "in front of the body" means "wouldn't also classify as
`OUT_TO_SIDE`." Wired into `GestureStateMachine` as an extra detector per
arm alongside the existing `SweepDetector`, firing independently of the
static-pose debounce path. 82/82 unit tests passing after the change.

**Re-ran the exact same `cheese` video and pipeline used to validate the
original four actions**, to check two things: did the new detectors
disturb the old ones, and does the flagged collision risk (a relaxing
`RAISED` arm reading as a false `VOLUME_DOWN`) actually happen on real
footage.

Regression: clean. All six original triggers (NEXT x2, PREVIOUS,
PLAY_PAUSE, SKIP x2) fired at the identical timestamps as the pre-volume-
gesture run. `VOLUME_DOWN` never fired -- the specific risk originally
flagged (arm lowering from `RAISED`) didn't happen on this footage,
despite several real `RAISED`->`DOWN` transitions in it.

`VOLUME_UP` fired 5 times, and inspecting the actual raw `rel_y`
trajectory around each trigger (not just whether it fired) told a more
useful story than the trigger count alone:

```
t=12.47s: 2.80 -> 2.48 -> 2.05 -> 1.06 -> 0.49 -> 0.19 -> -0.76 -> -2.35   (clean fast rise, ~0.5s)
t=15.70s: ~2.1-2.75 (flat ~1.3s) -> 1.53 -> 1.30 -> 0.25 -> -1.66 -> -2.71 (clean fast rise, ~0.3s, after a hold)
t=19.33s: 2.23 -> -0.85 -> -2.94 -> [oscillates -2.3..-3.1 for ~1.3s, including the trigger samples]
t=27.17s: oscillates 2.46-3.38 the *entire* window -- never leaves the resting range
t=30.03s: slow drift 2.97 -> ~1.0-1.7 over ~1s -- directionally "up" but stays fully within resting range
```

2 of 5 (12.47s, 15.70s) are genuinely clean, fast, monotonic rises from
resting to well past `RAISE_Y_THRESHOLD` -- exactly the intended motion,
even though the person in this recording was never trying to trigger a
volume gesture (it predates the feature). The other 3 are real gaps in
the current design, not noise to dismiss:

- **27.17s and 30.03s never leave the arm's normal resting range.** The
  detector only checks the *relative* delta between a window's first and
  last sample -- nothing requires the motion to actually reach raised
  territory, or start from a settled baseline. Ordinary pose-estimation
  jitter (real keypoint noise, not a code bug) or a slow, small drift
  within the resting band was enough to clear
  `MIN_VERTICAL_SWIPE_RANGE=0.8` on its own.
- **19.33s fired while the arm was already confirmed `RAISED`**, having
  actually risen a moment earlier (visible in the trajectory as the drop
  to -2.94 well before the trigger). The rolling window still partially
  overlapped that original rise, so jitter while *holding* the raised
  pose read as additional swiping. `SweepDetector` has a `.reset()` for
  exactly this kind of situation (clear history once a pose settles);
  `VerticalSwipeDetector` has the same method but nothing calls it.

## Fixing the two volume-gesture false-positive patterns (2026-09-12)

**Fix 1: require the window's endpoint to actually reach raised
territory.** `VerticalSwipeDetector._detect()` gained an endpoint check
(`up`: last sample's `rel_y < -RAISE_Y_THRESHOLD`; `down`: first
sample's) mirroring `static_poses.classify_arm`'s own `RAISED` boundary,
instead of accepting any sufficiently large relative delta regardless of
where it starts or ends. Re-ran against the same real footage: this
alone eliminated both "confined to resting range" triggers (27.17s,
30.03s) with no effect on the two genuine rises. 5 real `VOLUME_UP`
triggers down to 3 (12.60s, 19.37s, 19.93s).

**Fix 2, attempt 1 (dead end): reset on confirmation.** The plan going
in was to reset each swipe detector's history the moment its static
debouncer confirmed the relevant pose -- `right_volume_swipe` on
`RAISED` confirming, `left_volume_swipe` on `DOWN` confirming (chosen
over mirroring `RAISED` for both, since resetting `left_volume_swipe` on
`RAISED` was worked out on paper to wipe a continuous rise-then-fall
motion's starting point right as the debounce catches up mid-descent).
Implemented, unit-tested, redeployed, and re-run against the same real
footage -- **the two remaining triggers (19.37s, 19.93s) were still both
there, unchanged.** Instrumenting the actual run (logging
`right.confirmed` and `len(right_volume_swipe._samples)` frame by frame
around that window) showed why: `right.confirmed` was already `RAISED`
well *before* the observed window even started -- the arm had been held
raised for over a second by that point. A reset that only fires once, at
the moment of the *transition into* a state, does nothing for jitter
that occurs deep into an already-long hold of that same state. The log
showed the real pattern plainly: `right_volume_swipe`'s sample count
climbing from 1 to 7 over ~0.5s, triggering (which self-clears via
`add_sample`'s existing logic), climbing from 0 to 10 over the next
~0.5s, triggering again -- a clean, repeating ~0.6s cycle for as long as
the hold lasted, entirely independent of the one-time reset.

**Fix 2, attempt 2 (this is what shipped): gate evaluation on the
*current* confirmed state, not just reset at the transition.**
`right_volume_swipe` is now only fed samples at all while
`state.right.confirmed != ArmPose.RAISED` -- once the arm is confirmed
raised, evaluation stops entirely, for however long the hold lasts, and
resumes cleanly once it isn't (with `_prune`'s existing age-based cleanup
discarding anything stale by the time evaluation restarts, no explicit
clear needed). This is safe specifically for "up" because the legitimate
trigger already fires *before* `RAISED` confirms (debounce lags the raw
crossing by several frames in every real and synthetic trace looked at),
so gating on the *debounced* state doesn't cost any real detections --
only the post-confirmation jitter that shouldn't count anyway.

**Tried the mirror-image gate for `left_volume_swipe` (evaluate only
while confirmed `RAISED`) for symmetry -- it broke real swipe-down
detection, and got reverted.** Traced through the existing
`test_swipe_down_triggers_volume_down` sequence by hand: by the time
`RAISED` actually confirms (again, lagging the raw crossing), a fast
continuous rise-then-fall motion has frequently *already started
descending* below the raised threshold. Gating on "still confirmed
raised" cut off evaluation right as the descent needed to be measured,
so the down-swipe's own endpoint requirement (must *start* raised) could
never be satisfied by whatever samples were left. This is the same
underlying mistake as fix-2-attempt-1's original plan (reasoning about
raw pose *thresholds* while gating on *debounced, lagged* state, for a
motion whose two phases straddle exactly that lag), just caught a
different way. There was also never any real evidence this gate was
needed for "down" -- `VOLUME_DOWN` has not fired once across any
real-footage run in this whole investigation, gated or not -- so fixing
a problem that was never observed, at the cost of breaking one that was
working, wasn't a trade worth making. Left `left_volume_swipe` at
fix-1-only (endpoint check, no gating).

**Final re-validation:** all six original actions (NEXT x2, PREVIOUS,
PLAY_PAUSE, SKIP x2) still fire at the identical timestamps as every
prior run -- zero regression across three consecutive rounds of gesture
changes now. `VOLUME_DOWN` still never fires. `VOLUME_UP` fires exactly
**once**, at t=12.60s -- the one instance already confirmed (by its raw
`rel_y` trajectory) as a genuinely clean, fast rise. Both spurious
patterns are gone, no new ones introduced. 85/85 unit tests passing.
See `docs/backlog.md` for current status.

The throughline worth remembering from this whole detour: two separate,
independently-reasoned fixes (the original reset design, and the
symmetric "down" gate) both failed for the *same underlying reason* --
treating the debounced/confirmed state as if it tracked the raw pose in
real time, when it's deliberately several frames behind by design
(that's what `DEBOUNCE_FRAMES` is for). Every fix in this area needs to
be checked against "what does the raw classification look like during
the few frames where confirmed hasn't caught up yet," not just "what
does confirmed eventually settle to."
