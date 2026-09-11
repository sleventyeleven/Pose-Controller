# Hardware notes

## Boards on hand

| Board | SoC | NPU | Camera I/O | Role in this project |
|---|---|---|---|---|
| Radxa Dragon Q6A | Qualcomm QCS6490 | Hexagon, ~12 TOPS | 3x MIPI CSI | **Primary target.** Vision/IoT-class SoC with a real camera ISP pipeline and the best Qualcomm AI Hub model coverage of what's on hand. |
| Radxa Dragon Q8B | Snapdragon 8cx Gen3 | Hexagon, ~29+ TOPS | No documented MIPI CSI/ISP pipeline | Higher raw compute but PC/laptop-class chip, weaker fit for a camera-in vision pipeline. Not targeted for this milestone. |
| Arduino Uno Q | Qualcomm Dragonwing QRB2210 | Integrated AI engine (GPU+DSP+ISP) | Camera via headers | Acknowledged by the user as likely too weak for this vision workload; not targeted. |
| Arduino Ventuno Q | Qualcomm Dragonwing IQ-8275 | Hexagon, ~40 dense TOPS | 3x MIPI CSI | Looks like the strongest long-term fit, but is still on pre-order -- no hardware in hand yet. Revisit once it arrives. |

## Cameras available

| Camera | Interface | Status |
|---|---|---|
| Nexigo 1080p webcam | USB (UVC) | **Used for the first working pipeline.** Standard V4L2/OpenCV capture, no driver risk on an unfamiliar board. |
| Radxa Camera 4K (Sony IMX415) | MIPI CSI | Radxa's own product page lists compatibility with ROCK-series boards only (ROCK 3A/5A/5B/5C/5T, CM4/CM5 IO boards, etc.) -- the Dragon Q6A/Q8B are **not** in that list. CSI connector/driver compatibility with the Dragon series is unconfirmed. Defer until the USB pipeline is proven, then verify the physical connector and required kernel driver before relying on it. |
| ArduCam v2 8MP | MIPI CSI | Same unconfirmed-compatibility caveat as the Radxa Camera 4K. |

## NPU / QNN spike (milestone 1, not yet done)

Qualcomm AI Hub (`qai-hub-models`) publishes pre-optimized models -- including
pose estimation and person detection -- for QCS6490 specifically, exportable
to run via QNN. The open question is the exact runtime path on the Q6A's
Ubuntu image: whether that's `onnxruntime` with the QNN execution provider,
or AI Hub's own on-device inference wrapper. This needs to be validated on
the physical board before `inference/backends/qnn.py` can be implemented.
See `scripts/qnn_spike.md` for the plan.

## Offline constraint and Spotify

The "offline" requirement applies to the vision/AI decision pipeline (camera
capture through gesture recognition must need zero network access). Spotify
itself still requires network connectivity to authenticate and stream audio
-- that's outside this project's control. `control/` is built as a
`MediaController` abstraction specifically so the gesture pipeline never
depends on Spotify directly: the first implementation drives whatever
player is already running via OS media keys / Linux D-Bus MPRIS, which
requires no network access for the control path itself.
