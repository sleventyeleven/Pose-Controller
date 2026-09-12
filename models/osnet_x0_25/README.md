# osnet_x0_25 -- person re-identification embeddings

`model.onnx` is a plain `torch.onnx.export` (opset 17, legacy TorchScript-
based exporter -- the newer dynamo-based default needs `onnxscript`, not
installed) of
`qai_hub_models.models.osnet.model.OSNet.from_pretrained("osnet_x0_25")`,
exported 2026-09-11. NCHW input `[1, 3, 256, 128]` float32 RGB in [0, 1]
(ImageNet normalization is baked into the traced graph -- callers don't
normalize separately); output `[1, 512]` L2-normalized embedding.

## Why this model runs differently from the pose models

Unlike `models/mediapipe_pose_qcs6490/` (compiled via Qualcomm AI Hub for
the Q6A's Hexagon NPU, INT8-quantized), this model runs in **plain FP32 on
the CPU** via onnxruntime's default `CPUExecutionProvider` -- no AI Hub
compile job, no NPU, no quantization. Two reasons:

1. **Usage pattern is different.** Pose estimation runs on every person on
   every frame (real-time, latency-critical -- worth the NPU/quantization
   investment). Re-ID (`tracking.reid.ReidEmbedder`, used by
   `tracking.Tracker`) only runs occasionally: when a detection can't be
   IoU-matched to an existing track. At ~0.71M parameters, plain FP32 CPU
   inference (~20-50ms measured on the physical Q6A) is fast enough for
   that occasional use without the NPU pipeline's complexity.
2. **The official quantization path is blocked without a manual step.**
   OSNet's `get_calibration_dataset_cls()` requires `ENTIReIDDataset`,
   which needs a manual gated Google Drive download (see
   `scripts/qnn_spike.md` for the exact blocker). The model's own code
   comments flag that w8a8 quantization is sensitive enough to tank ReID
   accuracy if calibrated poorly (`get_hub_quantize_options` overrides the
   range scheme specifically to guard against this) -- not worth
   substituting ad-hoc calibration images for a model this sensitive to
   calibration quality, especially when plain FP32 CPU is fast enough
   anyway.

## Validation (2026-09-11, on the physical Q6A)

Using a synthetic two-person test image (two crops of the
`mediapipe_pose` demo photo placed side by side -- visually identical
content, different positions):

- Cosine similarity between the two (same-appearance) crops: **0.978**
- Cosine similarity against an unrelated background patch: **0.419**
- Full re-ID flow: a track that "disappeared" for several frames (past
  `Tracker`'s `max_age`) and reappeared at a **completely different
  position** (left side of frame -> right side) was correctly revived
  with its original `track_id` -- something IoU/Kalman tracking alone
  cannot do, confirming the actual purpose of this model in the pipeline.

To reproduce or update the export:
```bash
pip install -e ".[hub]"   # dev machine only, see docs/storage-footprint.md
python -c "
import torch
from qai_hub_models.models.osnet.model import OSNet
m = OSNet.from_pretrained('osnet_x0_25')
m.eval()
torch.onnx.export(m, torch.rand(1, 3, 256, 128), 'model.onnx',
                   input_names=['image'], output_names=['embedding'],
                   opset_version=17, dynamo=False)
"
```
If a different OSNet variant is desired (`osnet_x0_5`, `osnet_x1_0`, etc.
-- see `qai_hub_models.models.osnet.model.VARIANTS`), pass it to
`from_pretrained` and update `tracking/reid.py`'s `EMBED_INPUT_SIZE` if
the input shape changes (it doesn't for the OSNet family -- all variants
share the 256x128 input, only channel widths differ).
