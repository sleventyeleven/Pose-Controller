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
| Nexigo 1080p webcam | USB (UVC) | **Working.** Confirmed via `v4l2-ctl --list-devices` on the Q6A: enumerates as `/dev/video0`/`/dev/video1` with no extra setup. Used for the first working pipeline. |
| Radxa Camera 4K (Sony IMX415) | MIPI CSI | **Supported, but needs an overlay enabled -- not on by default.** Radxa's product page only lists ROCK-series boards, but the Q6A image ships a `radxa-overlays` DKMS package with device-tree overlays specifically for it: `qcs6490-radxa-dragon-q6a-cam2-radxa-camera-8m-219.dtbo` and the `cam3` equivalent (an unrelated `cam1-imx577` overlay exists too, for a different sensor -- not this camera). On a fresh image `dmesg` shows zero CSI/CamSS probing at all until the matching overlay is enabled. Enable via `sudo rsetup` (Hardware > Overlays, or whatever the current menu path is) for the CSI port the camera is physically plugged into, then reboot -- `rsetup`'s overlay pipeline is interactive/menu-driven and not safe to script blindly over SSH. Re-check with `v4l2-ctl --list-devices` and `dmesg | grep -i camss` after reboot. |
| ArduCam v2 8MP | MIPI CSI | Not yet tested; no matching overlay found in the `radxa-overlays` package during the Radxa Camera 4K investigation above -- likely needs its own overlay/driver, unconfirmed. |

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
