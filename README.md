# Pose-Controller

Offline, on-device pose/position-triggered media control. A distilled
proof-of-concept that isolates pose estimation, entity tracking, and
gesture-triggered control -- in a dry, controlled setting -- to be later
folded back into the [swim safety monitoring](https://hackersvanguard.com/ai-swim-safety-monitoring/)
project.

MVP goal: control Spotify playback using upper-body pose/gesture triggers
(e.g. raise your right arm for next, left for previous, an overhead sweep to
skip), entirely offline, running on a Radxa Dragon Q6A's Qualcomm Hexagon
NPU. See `docs/architecture.md` for the pipeline design and
`docs/hardware.md` for hardware notes and constraints.

**Stretch goal:** keep the whole system (OS, code, deps, models) small
enough to fit comfortably on a 32GB eMMC/UFS module -- see
`docs/storage-footprint.md`.

## Status

Milestone 1 (scaffolding) and the start of milestone 2 (capture +
single-pose baseline) are in place for both backends. The QNN backend
(`inference/backends/qnn.py`) is implemented and validated end-to-end on
the physical Q6A, and fast enough for real-time use: a full `estimate()`
call takes ~26ms (~2.2ms detector + ~1.1ms landmark on the Hexagon NPU,
via `onnxruntime`'s QNN execution provider, plus preprocessing) -- see
`scripts/qnn_spike.md` for the full trail, including three real bugs
found and fixed along the way (a mis-ported ROI formula, a wrong NCHW/
NHWC input layout assumption, and an initial ~476ms-per-frame subprocess
approach later replaced by a persistent-session one).

Not yet implemented: multi-person tracking, gesture recognition, the
overlay causality UI, and media control.

## Setup (dev loop, laptop with a webcam)

```bash
pip install -e ".[dev]"
python -m pose_controller.app --config configs/dev_laptop.yaml
```

Press `q` in the preview window to quit.

## Setup (on-device, Radxa Dragon Q6A)

See `docs/setup-q6a.md` for full board bring-up (image, BIOS, NPU
packages, camera). Once done:

```bash
sudo apt-get install -y python3-opencv python3-numpy fastrpc fastrpc-test fastrpc-dev libcdsprpc1 radxa-firmware-qcs6490
pip install --break-system-packages -e ".[device]"   # onnxruntime-qnn -- ~261MB, see docs/storage-footprint.md
python3 -m pose_controller.app --config configs/dragon_q6a.yaml
```

Don't set `ADSP_LIBRARY_PATH` yourself for this runtime path --
`onnxruntime-qnn` manages it internally (pointing at its own bundled
Hexagon libraries); overriding it causes a QNN device config error. See
`scripts/qnn_spike.md` for why.

## Exporting/updating on-device models

Model export (via Qualcomm AI Hub) needs PyTorch and should run on a dev
machine, never on the Q6A itself (see `docs/storage-footprint.md`):

```bash
pip install -e ".[hub]"
qai-hub configure --api_token <token>   # free account at https://aihub.qualcomm.com/
QAIHM_CI=1 python -m qai_hub_models.models.mediapipe_pose.export \
  --chipset qcs6490 --target-runtime precompiled_qnn_onnx --precision w8a8 \
  --components pose_detector pose_landmark_detector
```

See `models/mediapipe_pose_qcs6490/README.md` for the currently-checked-in
export's provenance and profiling results.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
