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

Milestones 1-3 are in place: scaffolding, capture + pose baseline, and
multi-person detection + tracking with appearance-based re-ID. The QNN
backend (`inference/backends/qnn.py`) is validated end-to-end on the
physical Q6A and fast enough for real-time use, including with multiple
people: ~15-20ms per `estimate()` call with 2 people detected
simultaneously (~2.2ms detector + ~1.1ms landmark per person on the
Hexagon NPU via `onnxruntime`'s QNN execution provider, plus
preprocessing). `tracking.Tracker` gives each person a stable `track_id`
across frames -- Kalman + IoU + Hungarian assignment survives brief
occlusion, and an appearance embedding (OSNet-x0.25, plain FP32 on CPU)
lets an ID survive fully leaving and re-entering frame, validated on
physical hardware (0.978 similarity for the same person reappearing at a
different position, 0.419 against background). See `scripts/qnn_spike.md`
for the full trail, including real bugs found and fixed along the way (a
mis-ported ROI formula, a wrong NCHW/NHWC input layout assumption, an
initial ~476ms-per-frame subprocess approach later replaced by a
persistent-session one, and a gated-dataset blocker on OSNet's official
quantization path that led to running it unquantized on CPU instead).

Gesture recognition (`gestures/`) is implemented: per-track pose
normalization, static arm-pose classification (down/raised/out-to-side,
debounced), a one-armed overhead sweep detector, and a state machine
mapping to `ControlAction`s (right-out -> NEXT, left-out -> PREVIOUS,
both-raised -> PLAY_PAUSE, sweep -> SKIP), all edge-triggered so holding
a pose doesn't repeat-fire. The overlay shows each person's current arm
states and a banner for the most recently triggered action, satisfying
the project's causality requirement. Validated with extensive unit tests
and a sanity check against real (not synthetic) detected keypoints, but
**not yet with a live camera actually performing the gestures** -- see
`docs/backlog.md`.

Not yet implemented: media control (`gestures.ControlAction`s currently
just print and show in the overlay; nothing drives Spotify/a player
yet). Other known gaps tracked in `docs/backlog.md`, notably: the CPU dev
backend is still single-person, the re-ID similarity threshold is a
starting guess not tuned against real footage, and the handedness
assumption in `gestures/normalize.py` (MediaPipe's L/R landmarks are
anatomical, not mirrored) is unverified against a live camera.

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
sudo apt-get install -y python3-opencv python3-numpy python3-scipy fastrpc fastrpc-test fastrpc-dev libcdsprpc1 radxa-firmware-qcs6490
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

The re-ID model (`models/osnet_x0_25/`) is exported differently -- plain
`torch.onnx.export`, no AI Hub compile job, no quantization (see
`models/osnet_x0_25/README.md` for why):

```bash
pip install -e ".[hub]"
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

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
