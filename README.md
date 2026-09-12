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

**New here, or want the story behind the code?** `docs/journey.md` is a
narrative retrospective of how this project got built -- what broke, what
each break taught the next decision, and where the real (non-obvious)
findings came from. It's the readable version of the detailed logs in
`scripts/qnn_spike.md` and `docs/backlog.md`.

**Stretch goal:** keep the whole system (OS, code, deps, models) small
enough to fit comfortably on a 32GB eMMC/UFS module -- see
`docs/storage-footprint.md`.

## Status

Milestones 1-4 are in place and validated end-to-end on the physical
Q6A, including with a live camera and real gestures -- not just
synthetic tests. The current on-device pipeline
(`inference/backends/qnn.py`) is a two-stage detector: YOLOv8n-det
(640x640, first-stage person localization) feeding BlazePose's landmark
model (25 keypoints per person), both running on the Hexagon NPU via
`onnxruntime`'s QNN execution provider. This replaced BlazePose's own
bundled 128x128 detector after real-camera testing found it loses
small/distant subjects at ordinary room distance regardless of image
quality -- see `docs/journey.md` for the story and `scripts/qnn_spike.md`
for the numbers.

`tracking.Tracker` gives each person a stable `track_id` across frames --
Kalman + IoU + Hungarian assignment survives brief occlusion, and an
appearance embedding (OSNet-x0.25, plain FP32 on CPU) lets an ID survive
fully leaving and re-entering frame. `reid_similarity_threshold` (0.5)
is tuned against real measured same-person embedding similarity from
recorded footage, not a starting guess.

Gesture recognition (`gestures/`) covers all six `ControlAction`s from
the original brief plus volume control: right-out -> `NEXT`, left-out ->
`PREVIOUS`, both-raised -> `PLAY_PAUSE`, overhead sweep -> `SKIP`, right
swipe up -> `VOLUME_UP`, left swipe down -> `VOLUME_DOWN`, all
edge-triggered. **Validated end-to-end against real recorded footage**
running the full `estimate -> track -> gesture` loop -- every action has
been triggered correctly from a real person's real arm movements, though
reliability still varies with detection hit-rate and camera framing on
harder footage (see `docs/backlog.md`).

A local browser dashboard (`web/`) streams the live overlay feed plus a
gesture-trigger queue -- off by default (`OverlayConfig.web_enabled`),
for demos and debugging without a manual capture/scp/inspect cycle.

Not yet implemented: media control (`gestures.ControlAction`s currently
print and show in the overlay/dashboard; nothing drives Spotify/a player
yet). Known open gaps, tracked in `docs/backlog.md`: the CPU dev backend
is still single-person, the re-ID threshold's false-merge risk is
unvalidated against real multi-person footage, some tracking-identity
instability may still trace back to camera framing (two alternate
landmark pipelines, YOLO26-Pose and HRNetPose, are being built alongside
the current one to help isolate this -- both compile and run on real
Hexagon NPU hardware, YOLO26-Pose via Ultralytics' own QNN export after
Qualcomm AI Hub's failed to compile it, but neither is validated on real
footage yet), and the
handedness assumption in `gestures/normalize.py` has held up in every
real test so far but is still worth watching.

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

Alternate landmark-stage pipelines (`configs/dragon_q6a_yolo26.yaml`,
`configs/dragon_q6a_hrnet.yaml`) are being evaluated alongside the
default above, not replacing it -- see `docs/backlog.md` and
`docs/journey.md` for status and real-footage results.

## Setup (on-device, Radxa Dragon Q8B)

A second deployment target sharing the Q6A's Hexagon V68 NPU -- every
model this project has compiled runs there unmodified, no separate
export needed. See `docs/setup-q8b.md` for full board bring-up. Once
done, the same install steps as the Q6A apply, just pointed at the Q8B's
configs (`configs/dragon_q8b.yaml`, or the `_yolo26`/`_hrnet` variants).

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

The `yolo26` backend's models are exported differently again -- via
Ultralytics' own local QNN export, not AI Hub (whose compiler fails on
these models -- see `docs/backlog.md`). Note `soc_model`, not the
generic `htp_arch`, is required for QCS6490/SC8280XP specifically (see
`models/yolo26n_pose_qcs6490/README.md` for why):

```bash
pip install -e ".[hub]"   # includes ultralytics
python -c "
from ultralytics.utils import QNN_HTP_TARGETS
QNN_HTP_TARGETS['qcs6490'] = ('soc_model', '93')
from ultralytics import YOLO
YOLO('yolo26n-pose.pt').export(format='qnn', name='qcs6490', imgsz=640)
"
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
