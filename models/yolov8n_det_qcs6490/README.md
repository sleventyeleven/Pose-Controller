# yolov8n_det -- compiled for QCS6490 (first-stage person detector)

Compiled via Qualcomm AI Hub (`qai_hub_models.models.yolov8_det.export`) on
2026-09-12, targeting chipset `qcs6490`, `--target-runtime
precompiled_qnn_onnx`, `--precision w8a8` (default 640x640 input).

- `model.onnx` + `model.bin` (external context-binary data; both files
  required, must stay together) -- AI Hub compile job
  [j5wlelw6p](https://workbench.aihub.qualcomm.com/jobs/j5wlelw6p/),
  profiling job
  [jg9zlz0lp](https://workbench.aihub.qualcomm.com/jobs/jg9zlz0lp/).
  Renamed from the export's own `yolov8_det.onnx` /
  `yolov8_det_qairt_context.bin` to match this project's `model.onnx` /
  `model.bin` convention -- the rename requires patching the ONNX graph's
  `EPContext` node (`ep_cache_context` attribute hardcodes the bin
  filename as a relative path), not just renaming files on disk.

## Why this model exists

Replaces BlazePose's own bundled detector (the first stage of
`models/mediapipe_pose_qcs6490/`) for person *localization* only --
BlazePose's landmark model is still used for the actual keypoints. That
detector's fixed 128x128 input was found to lose small/distant subjects
at ordinary room camera-to-subject distance, regardless of image quality:
a person who reads clearly in a full-resolution, well-exposed photo
becomes too small a silhouette once the whole frame is squeezed down to
128x128 for the detector, and its anchors simply never fire. See
`scripts/qnn_spike.md` for the full investigation (comparing raw detector
logits against a known-good demo image was the key diagnostic: a cropped-
tight version of an otherwise-failing frame produced the same confident
logit as the demo image, while the full frame never exceeded 0.0).
YOLOv8n's 640x640 input (5x the linear resolution, ~25x the pixel budget)
preserves far more detail before any localization step.

## I/O

Input `image`: NHWC uint8 `[1, 640, 640, 3]`, standard [0,1] float ->
uint8 scaling (same convention as the mediapipe models, `scale=1/255,
zero_point=0`).

Exported with `include_postprocessing=True` (the default) -- box decoding
and per-box argmax-over-classes are already done inside the graph, so
`inference/backends/_yolo_detect.py` only has to do score/class filtering
and NMS on top, not raw anchor decoding. Outputs, confirmed via
`onnx.load` against the actual graph (not assumed -- Radxa's own QCS6490
deployment notes for this model warn of an output-ordering quirk on this
chipset, see below):

| Output | Shape | Meaning |
|---|---|---|
| `boxes` | `[1, 8400, 4]` | xyxy, in the padded/scaled 640x640 input space |
| `scores` | `[1, 8400]` | max-class probability per box |
| `class_idx` | `[1, 8400]` | COCO class index (0 = person) per box |

All three are uint8, quantized -- but unlike the mediapipe models, this
graph carries its quantization as explicit `QuantizeLinear`/
`DequantizeLinear` nodes with real initializers, so the scale/zero_point
came straight from `onnx.load()` (see the node named identically to each
output), not `qnn-context-binary-utility`:

| Output | scale | zero_point |
|---|---|---|
| `boxes` | 3.1010673 | 25 |
| `scores` | 0.00390625 (= 1/256) | 0 |
| `class_idx` | **0.0** | 0 |

`class_idx`'s graph-declared scale of exactly 0.0 is not a real
quantization scale (it would dequantize every value to exactly 0
regardless of the raw byte) -- treated as an exporter formality for
smuggling an integer-valued output (0-79) through a uint8 QDQ wrapper.
`inference/backends/qnn.py` uses `QuantSpec(scale=1.0, offset=0.0)` (pure
passthrough) for it instead, confirmed correct on-device: a person in
frame reliably decodes to `class_idx == 0` (COCO's person class), not a
garbage float.

**Radxa's own docs for this exact model on QCS6490** (a sibling board,
Airbox Q900) flag a real quirk: some exports return outputs ordered
`[scores, class_idx, boxes]` instead of `[boxes, scores, class_idx]`.
This project's own export did NOT exhibit that reordering (confirmed via
`onnx.load` -- graph order is `boxes, scores, class_idx` as documented
above) -- but `_yolo_detect.py`/`qnn.py` look up outputs by name via
`OnnxQnnRunner`'s dict-keyed `run()`, not positionally, so this wouldn't
matter even if a future re-export did reorder them.

## Profiling (AI Hub, on a real QCS6490 device -- Dragonwing RB3 Gen 2 Vision Kit)

| Metric | Value |
|---|---|
| Inference time | 4.6 ms |
| Compute units | 254/254 ops on NPU (0 GPU, 0 CPU) |
| boxes output PSNR (on-device vs local fp32) | 38.06 dB |
| scores output PSNR (on-device vs local fp32) | 33.54 dB |

To reproduce or update: `pip install -e ".[hub]"` on a dev machine (never
on the Q6A -- pulls in PyTorch), plus `ultralytics`, `aiofiles`, and
`pycocotools` (all dev-machine-only extras this particular model's export
needs beyond the base `[hub]` extra -- COCO calibration data download and
the YOLO model definition itself depend on them), `qai-hub configure
--api_token <token>` (free account at https://aihub.qualcomm.com/), then:
```bash
QAIHM_CI=1 python -m qai_hub_models.models.yolov8_det.export \
  --chipset qcs6490 --target-runtime precompiled_qnn_onnx --precision w8a8 \
  --output-dir export_assets/yolov8_det
```
Then rename+patch as described above (see the export command's own output
for the exact source filenames, which vary run to run).
