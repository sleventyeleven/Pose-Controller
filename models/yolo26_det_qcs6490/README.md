# yolo26n -- Ultralytics local QNN export, QCS6490-targeted (soc_model=93)

Same provenance and rationale as `models/yolo26n_pose_qcs6490/README.md`
-- AI Hub's export failed ("exit code 14"), Ultralytics' own local QNN
export succeeded. **Read that README's "Why `soc_model=93`, not the
generic `htp_arch=68` target" section** -- the first export attempt
(generic `name="68"`) loaded fine on the Q8B but failed to load on the
Q6A; targeting QCS6490's real `soc_model` value fixed it and still works
on both boards.

```python
from ultralytics.utils import QNN_HTP_TARGETS
QNN_HTP_TARGETS["qcs6490"] = ("soc_model", "93")

from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.export(format="qnn", name="qcs6490", imgsz=640)
```

`model.onnx` is self-contained (QNN context binary embedded inline).
Verified running on both boards' real Hexagon NPU via `onnxruntime-qnn`:

| Board | Warm inference (median, 20 calls) |
|---|---|
| Q6A | ~14.7ms |
| Q8B | ~9.6ms |

## I/O

Plain float32 (see the pose model's README for why -- same internal
`QuantizeLinear`/`DequantizeLinear` handling).

- Input `images`: NHWC float32 `[1, 640, 640, 3]`, `/255.0` normalized.
- Output `output0`: float32 `[1, 84, 8400]` -- raw, per anchor: `[cx, cy,
  w, h, 80x class_score]`, box already decoded to pixel-space (cx, cy, w,
  h), class scores already sigmoid-activated but **not** argmax'd over
  classes (unlike an AI-Hub `include_postprocessing=True` export).
  `inference/backends/_yolo26_detect.py`'s `decode_raw_output` does the
  box-format conversion + per-anchor argmax, then `detect_objects`
  (unchanged) does score filtering + NMS + class-name resolution.
