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

## Status

Milestone 1 (scaffolding) and the start of milestone 2 (capture +
single-pose baseline, dev/CPU backend only) are in place. Multi-person
tracking, gesture recognition, the overlay causality UI, media control, and
the on-device QNN backend are not yet implemented.

## Setup (dev loop, laptop with a webcam)

```bash
pip install -e ".[dev]"
python -m pose_controller.app --config configs/dev_laptop.yaml
```

Press `q` in the preview window to quit.

## Setup (on-device, Radxa Dragon Q6A)

The QNN backend (`inference/backends/qnn.py`) is a stub pending the NPU
spike described in `scripts/qnn_spike.md`. Once implemented:

```bash
pip install -e ".[qnn]"
python -m pose_controller.app --config configs/dragon_q6a.yaml
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
