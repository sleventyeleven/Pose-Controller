# hrnet_pose -- compiled for QCS6490 (landmark stage, alongside evaluation)

Compiled via Qualcomm AI Hub (`qai_hub_models.models.hrnet_pose.export`) on
2026-09-12, targeting chipset `qcs6490`, `--target-runtime
precompiled_qnn_onnx`, `--precision w8a8` (default 256x192 input).

- `model.onnx` + `model.bin` (external context-binary data; both files
  required, must stay together) -- AI Hub compile job
  [jp3zvz3z5](https://workbench.aihub.qualcomm.com/jobs/jp3zvz3z5/),
  profiling job
  [jgolkl0dg](https://workbench.aihub.qualcomm.com/jobs/jgolkl0dg/).
  Already named `model.onnx`/`model.bin` on download for this export --
  unlike `models/yolov8n_det_qcs6490/`, no `EPContext` node patching was
  needed (`ep_cache_context` already points at `./model.bin`).

## Why this model exists

Evaluated as a second alternative *landmark stage* (same role as
BlazePose's landmark model in `models/mediapipe_pose_qcs6490/`, not a
whole-frame detector) after both YOLO26-Pose and YOLO26-Detection hit a
reproducible "exit code 14" QNN context-binary compile failure on this
same chipset (see `docs/backlog.md`) -- HRNetPose uses the same `w8a8`
quantization path already proven to work for every other model in this
project, where YOLO26 requires `w8a16`. It's a genuinely different
alternative to YOLO26-Pose too: a top-down, single-person, pre-cropped
model reusing the existing YOLOv8n-det first stage
(`models/yolov8n_det_qcs6490/`), not a fully end-to-end one -- see
`src/pose_controller/inference/backends/_hrnet_pose.py`'s module
docstring for the full reasoning, including why its heatmap-based,
independent per-keypoint confidence is a candidate fix for the same
"gestures only register facing the camera" symptom YOLO26-Pose was
evaluated for.

Getting this far required working around a broken dev-machine dependency
chain (`mmcv`/`mmpose`/`mmdet` on Python 3.13) -- see `docs/backlog.md`
for the full list of workarounds.

## I/O

Confirmed directly from the compiled model's own `input_spec`/
`output_spec` (`qai_hub.get_job(job_id).get_target_model()`), not
assumed:

- Input `image`: NHWC uint8 `[1, 256, 192, 3]`, `scale=0.003917243331670761,
  zero_point=0` -- close to but not exactly the standard `1/255` used by
  this project's other models (a marginally different value from this
  model's own calibration), so it gets its own `QuantSpec` constant in
  `inference/backends/hrnet_qnn.py` rather than reusing the shared one.
- Output `heatmaps`: NHWC uint8 `[1, 64, 48, 17]`, `scale=
  0.003737130668014288, zero_point=10.0` -- 17 COCO keypoints, each a
  64x48 (H, W) confidence heatmap at stride 4 relative to the 256x192
  input (standard HRNet convention: 256/4=64, 192/4=48). Decoded by
  simple per-channel argmax (peak location + peak value as confidence) in
  `_hrnet_pose._decode_heatmaps` -- no sub-pixel refinement yet, see
  `docs/backlog.md`.

## Profiling (AI Hub device farm, 100 iterations)

Median inference time ~5.9ms per call (range ~5.6-7.0ms) for the landmark
stage alone on a real QCS6490 device -- about 5x BlazePose's landmark
model (~1.1ms, see `scripts/qnn_spike.md`) but still comfortably within a
30fps frame budget alongside the ~2.2ms YOLOv8n-det first stage. **Not
yet measured end-to-end on the physical Q6A** (camera capture + tracking
+ gestures + overlay all add their own overhead) -- treat this as a
per-model NPU number only, not a validated full-pipeline framerate.
