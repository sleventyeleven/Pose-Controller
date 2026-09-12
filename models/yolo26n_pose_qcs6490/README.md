# yolo26n-pose -- Ultralytics local QNN export, Hexagon v68

**Not from Qualcomm AI Hub.** The AI Hub export
(`qai_hub_models.models.yolo26_pose`) was attempted first and failed
reproducibly with a QNN context-binary compile error ("exit code 14")
for both this model and YOLO26-Detection -- see `docs/backlog.md`. This
model instead comes from **Ultralytics' own local QNN export path**
(https://docs.ultralytics.com/integrations/qnn), which runs entirely
offline (no Qualcomm account, no cloud upload) and succeeded where AI
Hub's toolchain did not:

```python
from ultralytics.utils import QNN_HTP_TARGETS
QNN_HTP_TARGETS["qcs6490"] = ("soc_model", "93")  # see "Why soc_model, not htp_arch" below

from ultralytics import YOLO
model = YOLO("yolo26n-pose.pt")
model.export(format="qnn", name="qcs6490", imgsz=640)
```

`model.onnx` is the entire artifact -- self-contained, the QNN context
binary is embedded inline in an `EPContext` node (no separate `.bin`
file, unlike this project's AI-Hub-compiled models).

## Why `soc_model=93`, not the generic `htp_arch=68` target

Ultralytics' QNN export has a built-in `name="68"` option targeting
Hexagon **architecture version** v68 generically -- since both this
project's boards use Hexagon V68 (confirmed via `fastrpc_test -a v68` on
both), that looked sufficient. **It wasn't**: a `htp_arch=68`-targeted
export loaded and ran fine on the Q8B but failed to load on the Q6A with
`QNN_CONTEXT_ERROR_CREATE_FROM_BINARY`. Checking `qai_hub.get_devices()`'s
own device attributes revealed why: `soc-model` is a per-*exact-chip*
identifier, not per-architecture-generation -- Hexagon v68 alone covers
at least four different `soc-model` values across Qualcomm's real device
catalog (30, 35, 39, 93), and the generic `htp_arch=68` finalization
apparently defaults to a different one of those than QCS6490's own (`93`,
read directly from AI Hub's `Dragonwing RB3 Gen 2 Vision Kit` device
entry -- a real QCS6490 board AI Hub does support, not a guess).
Re-exporting with `soc_model=93` explicitly (there's no built-in
Ultralytics name for it, so `QNN_HTP_TARGETS` needs a one-line patch as
shown above) fixed the Q6A load failure -- and, tested directly, the
resulting binary **still loads and runs fine on the Q8B too** (see
below), so it's a strictly better, more broadly-compatible target than
the generic one, not a per-board tradeoff.

## Verified on real hardware -- both boards

Copied `model.onnx` unmodified to both the Radxa Dragon Q6A and Q8B and
loaded it via `onnxruntime-qnn`'s QNN execution provider on each --
genuine on-NPU execution (not a CPU fallback) on both:

| Board | Warm inference (median, 20 calls) |
|---|---|
| Q6A | ~14.9ms |
| Q8B | ~9.9ms |

The Q6A being slower here (the reverse of `yolov8n_det_qcs6490`, where
Q8B was slower) is plausible clock/binning/driver-stack variance between
two different physical boards -- not yet root-caused, and not a red flag
on its own (`docs/backlog.md`).

## I/O

Plain float32, unlike this project's AI-Hub-compiled models -- the
graph's own `QuantizeLinear`/`DequantizeLinear` nodes handle the uint8/
int16 quantization internally (confirmed via `onnx.load()`: input scale
`1.5259022e-05` ~= 2^-16, zero_point 0 -- consistent with `w8a16`'s
16-bit activations), so `inference/backends/yolo26_qnn.py` needs no
manual `QuantSpec` math, unlike `qnn.py`/`hrnet_qnn.py`.

- Input `images`: NHWC float32 `[1, 640, 640, 3]`, same `/255.0`
  normalization convention as every other model in this project.
- Output `output0`: float32 `[1, 56, 8400]` -- **raw**, not
  post-processed the way an AI-Hub `include_postprocessing=True` export
  would be. Confirmed from `ultralytics.nn.modules.head.Detect._inference`
  / `Pose26.kpts_decode` source (not assumed): per anchor (8400 total,
  640x640 at strides 8/16/32), channels are `[cx, cy, w, h, person_conf,
  17x(kp_x, kp_y, kp_visibility)]` -- box already stride/anchor-decoded
  to pixel space in (cx, cy, w, h) form, confidence and keypoint
  visibility already sigmoid-activated, keypoint x/y already decoded to
  640x640 pixel space, all in COCO keypoint order. Still needs NMS across
  anchors -- `inference/backends/_yolo26_pose.py`'s `decode_raw_output`
  does the box-format conversion, and `detect_poses` (unchanged from the
  originally-planned AI-Hub contract) does the score filtering + NMS +
  `Landmark` mapping.
