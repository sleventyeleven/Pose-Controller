# mediapipe_pose -- compiled for QCS6490

Compiled via Qualcomm AI Hub (`qai_hub_models.models.mediapipe_pose.export`)
on 2026-09-11, targeting chipset `qcs6490` (matches the Radxa Dragon Q6A's
Qualcomm QCS6490 SoC), `--target-runtime precompiled_qnn_onnx`. See
`scripts/qnn_spike.md` and `docs/setup-q6a.md` for the full export/
verification process, including why `precompiled_qnn_onnx` (not
`qnn_context_binary`) is the runtime target: it produces an ONNX file
loadable by `onnxruntime`'s QNN execution provider, which keeps the model
loaded across calls (~2ms/frame) instead of reloading it every call
(~200ms/frame) the way `qnn-net-run` does.

- `pose_detector/` -- `model.onnx` + `model.bin` (external context-binary
  data; both files required, must stay together) -- AI Hub job
  [jp2rk7wxg](https://workbench.aihub.qualcomm.com/jobs/jp2rk7wxg/)
- `pose_landmark_detector/` -- same structure -- AI Hub job
  [jp8eq3xzp](https://workbench.aihub.qualcomm.com/jobs/jp8eq3xzp/)
- `anchors_pose.npy` -- BlazePose detector anchors (896x4), needed by
  `inference/backends/_blazepose.py`'s anchor decode; copied from the
  `MediaPipePyTorch` external repo `qai_hub_models` clones during export.

Both models are quantized to INT8 (`--precision w8a8`). Profiled on a
real QCS6490 device via AI Hub:

| Model | Inference time | Peak memory |
|---|---|---|
| pose_detector | 1.62 ms | ~4.4 MB |
| pose_landmark_detector | 1.20 ms | ~4.7 MB |

Verified end-to-end on the physical Q6A via `inference/backends/qnn.py`
(`onnxruntime` + `onnxruntime-qnn`, see `scripts/qnn_spike.md`): ~2.2ms /
~1.1ms per warm inference call, full `estimate()` on a 3088x2316 test
photo in ~26ms, producing 25 correctly-proportioned keypoints matching
the fp32 reference pipeline.

**I/O is raw quantized uint8**, not auto-handled float -- callers must
quantize input and dequantize output using each tensor's scale/offset
(hardcoded as `QuantSpec` constants in `inference/backends/qnn.py`,
extracted via `qnn-context-binary-utility --context_binary=model.bin
--json_file=info.json`; see `scripts/qnn_spike.md` for why ONNX has no
standard place to carry this metadata for an opaque `EPContext` node).

To reproduce or update: `pip install -e ".[hub]"` on a dev machine (never
on the Q6A -- pulls in PyTorch), `qai-hub configure --api_token <token>`
(free account at https://aihub.qualcomm.com/), then:
```bash
QAIHM_CI=1 python -m qai_hub_models.models.mediapipe_pose.export \
  --chipset qcs6490 --target-runtime precompiled_qnn_onnx --precision w8a8 \
  --components pose_detector pose_landmark_detector
```
Each output `.zip` contains `model.onnx` + `model.bin` for one component
-- extract both into `pose_detector/` / `pose_landmark_detector/` here.
If quantization/precision changes, re-extract the new scale/offset values
per output tensor and update the `QuantSpec` constants in
`inference/backends/qnn.py` to match.
