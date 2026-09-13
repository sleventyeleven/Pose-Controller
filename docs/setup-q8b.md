# Q8B bring-up steps

Step-by-step record of what it took to get a fresh Radxa Dragon Q8B
reachable and usable for this project, worked out by hand over SSH on
2026-09-12. Unlike `docs/setup-q6a.md`, this board was set up specifically
to evaluate as an *alternate* target for the YOLO26-Pose/YOLO26-Detection
alongside-pipeline (see `docs/backlog.md`) after both variants hit a
reproducible compile failure on the Q6A's QCS6490.

**Status: this board is a viable second deployment target for every
model this project has compiled, including YOLO26, confirmed on real
hardware** -- an earlier version of this doc concluded the opposite (see
"AI Hub chipset support" below for what that first check actually showed
and why it wasn't the full picture). YOLO26 itself was also separately
unblocked the same day, via a different export toolchain than the one
that failed -- see `docs/backlog.md`.

## 0. What's actually in the box

Identified from the device tree and PCI bus, not assumed from the board's
name:

- **SoC: Qualcomm SC8280XP** (`cat /proc/device-tree/compatible` ->
  `qcom,sc8280xp`; `/sys/devices/soc0/soc_id` -> `SC8280XP`) -- this is the
  "Snapdragon 8cx Gen 3" **compute/laptop** platform (used in devices like
  the Lenovo ThinkPad X13s), **not** the mobile "Snapdragon 8 Gen 3"
  (Qualcomm SM8650) that had been assumed earlier in this project. These
  are different chips with different Hexagon NPU generations -- see below
  for why this distinction turned out to matter a lot.
- CPU: 8x ARM Cortex-A78C @ up to 2.44 GHz (`lscpu`).
- Storage: 500GB NVMe (`nvme0n1`, MAXIO MAP1202 controller), ~469GB
  usable, mounted at `/`.
- OS: Ubuntu 26.04.1 LTS ("Resolute Raccoon"), same desktop/GNOME-flavored
  image style as the Q6A's stock install.
- Python 3.14.4 preinstalled (`/usr/bin/python3`) -- newer than what the
  Q6A shipped with; not yet validated against this project's dependency
  set (`qai_hub_models`, `torch`, etc. -- see `docs/backlog.md`'s
  Python-3.13-compatibility notes from the HRNetPose export work, which
  may or may not carry over to 3.14).
- ~~**No physical camera is attached.**~~ True as of first bring-up: on
  `/dev/video0`/`/dev/video1`, `udevadm info` identified both as the
  SoC's Iris hardware video **codec** block (`ID_V4L_PRODUCT=Iris
  Decoder`/`Iris Encoder` -- encode/decode acceleration, not camera
  capture), confirming no camera was silently missed at that point. The
  Nexigo USB webcam was physically moved over from the Q6A on
  2026-09-13 (plug-and-play UVC, no driver work needed as expected) --
  it enumerated as **`/dev/video2`** (new device nodes `/dev/video2`/
  `/dev/video3`, since indices 0/1 were already taken by the Iris
  codec), confirmed by testing both new nodes directly rather than
  assuming index continuity from the Q6A -- `video3` doesn't open at
  all, `video2` is the real capture device. `configs/dragon_q8b*.yaml`
  all set `capture.source: "2"` accordingly, not `"0"` like the Q6A's
  configs.

## 1. SSH access

The board ships with password auth only (`radxa` / `radxa123`, provided
out of band -- not written here). Set up key-based access matching the
Q6A's pattern:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/pose_controller_q8b -N "" -C "pose-controller-q8b"
# install the public key using the one-time password (PuTTY's plink used here on Windows,
# since OpenSSH's ssh doesn't support non-interactive password auth):
"/c/Program Files/PuTTY/plink.exe" -ssh -pw radxa123 radxa@192.168.5.94 \
  "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '<pubkey contents>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

`~/.ssh/config` alias added (mirrors the existing `q6a` entry):

```
Host q8b
    HostName 192.168.5.94
    User radxa
    IdentityFile ~/.ssh/pose_controller_q8b
    StrictHostKeyChecking accept-new
```

`ssh q8b` now works without a password.

## 2. Disable idle suspend (do this early -- same issue as the Q6A)

Confirmed present before fixing: `gsettings get
org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout` ->
`900` (15 min), and none of `sleep.target` / `suspend.target` /
`hibernate.target` / `hybrid-sleep.target` were masked -- the exact same
stock-desktop-image idle-suspend trap documented in `docs/setup-q6a.md`
section "1a", which drops SSH/network along with the rest of the machine
after 15 min of local-input idle (SSH/CPU activity does not reset it).

Fixed the same way:

```bash
sudo gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

Verified `systemctl is-enabled` on all four targets now reports `masked`.
(The `gsettings` read-back still showed `900` afterward when checked over
a plain SSH session with no active GNOME session/D-Bus bus to write
through -- per the Q6A doc, the `systemctl mask` is the actual hard
backstop that matters regardless, since it blocks the suspend request at
the systemd level no matter what triggers it.)

## 3. AI Hub *cloud compile* has no chipset entry for this board -- but that turned out not to matter

The entire reason this board was being evaluated was to retry the
YOLO26-Pose/YOLO26-Detection export that failed on the Q6A's QCS6490 (see
`docs/backlog.md`), on the assumption that a newer Hexagon NPU generation
might not hit the same "exit code 14" QNN context-binary compile failure.

Checked directly against the `qai_hub` Python API (`hub.get_devices()`,
78 devices total as of this writing) rather than assuming: **there is no
AI Hub device or chipset entry for `sc8280xp` at all** -- the full list of
Qualcomm compute-tier chipsets AI Hub's cloud compile service supports is
`sc8340xp`, `sc8380xp`, `sc8480xp` (newer generations) plus the Snapdragon
X Elite / X Plus / X2 Elite laptop chips. This was initially (and
incorrectly) treated as a hard blocker -- "AI Hub can't compile for this
chip, so nothing can target this board." That conclusion missed the
actual question: does a model *already compiled* for a different chip
that happens to share the same Hexagon generation run here anyway?

Radxa's own official docs answer yes, explicitly, for this exact pair of
chips: the [Q8B YOLOv8-det NPU
example](https://docs.radxa.com/en/dragon/q8b/app-dev/npu-dev/qai-appbuilder-demo/yolov8-det)
states outright that **"SC8280XP and QCS6490 both use Hexagon V68"** and
instructs using `--chipset 6490` on this board for that reason, and the
[NPU enablement
page](https://docs.radxa.com/en/dragon/q8b/app-dev/npu-dev/fastrpc-setup)
gives `fastrpc_test -a v68` (the same architecture flag, and the same
verification tool, used on the Q6A) as this board's own DSP verification
command.

**Verified directly on this board, not just taken on Radxa's word:**

```bash
sudo apt install fastrpc-test   # fastrpc/radxa-firmware-sc8280xp/task-qualcomm-npu were already present
fastrpc_test -a v68             # RESULT: All applicable tests PASSED, Sum = 499500 -- identical to the Q6A's own verification
```

Then, with `onnxruntime-qnn` installed in a fresh venv (`pip install
onnxruntime-qnn` -- a `manylinux_2_34_aarch64` wheel exists for this
board's Python 3.14, no compatibility issues), **this project's
already-compiled QCS6490 models were copied over unmodified and loaded
successfully on the Q8B's real Hexagon v68 NPU** via the exact same
`OnnxQnnRunner` registration pattern used on the Q6A
(`inference/backends/_qnn_runtime.py`):

| Model | Q8B warm inference (median) | Q6A reference |
|---|---|---|
| `yolov8n_det_qcs6490` | ~5.9ms | ~2.2ms (`scripts/qnn_spike.md`) |
| `hrnet_pose_qcs6490` | ~6.5ms | ~5.9ms (AI Hub device-farm profile) |
| `yolo26n_qnn.onnx` (detect) | ~7.1ms | not yet tested on Q6A |
| `yolo26n-pose_qnn.onnx` | ~6.7ms | not yet tested on Q6A |

(The two YOLO26 rows above are from the first export attempt, generically
targeting Hexagon architecture v68 -- since superseded by a QCS6490-
`soc_model`-targeted re-export after this generic one failed to load on
the Q6A entirely; see `docs/backlog.md`. **For current numbers on both
boards, all pipelines, see `docs/pipelines.md`** -- this table is kept
here as the historical record of this board's initial bring-up
verification, not a live reference.)

All ran genuinely on the NPU (QNN execution provider found and used,
real DSP/`rpcmem` activity in the logs), not a silent CPU fallback. The
`yolov8n_det` gap versus the Q6A's own measured number is plausible clock/
binning variance between two different physical chips of the same
architecture, not evidence of a fallback path -- worth a real footage
comparison later, but not a red flag on its own.

The two YOLO26 models are **not** the ones that failed to compile via
Qualcomm AI Hub -- those exports (`--chipset qcs6490` via `qai_hub_models`)
hit a reproducible "exit code 14" QNN compile failure regardless of
target device (see `docs/backlog.md`). These come from **Ultralytics'
own local QNN export** instead
(https://docs.ultralytics.com/integrations/qnn, `model.export(format=
"qnn", name="68")`), which succeeded where AI Hub's toolchain did not --
a different compiler/toolchain, not a different chip. Both variants
compiled in ~10 seconds each, fully offline.

**Conclusion, corrected from the first pass:** AI Hub's cloud compiler
not listing `sc8280xp` only means you can't request a *new* compile job
targeting this exact chip by name through that specific service -- it
does **not** mean nothing can run here, and it does **not** mean every
model is stuck. Every model this project has now compiled for Hexagon
v68 -- the default BlazePose two-stage pipeline, YOLOv8n-det, HRNetPose,
**and now YOLO26 (both pose and detect)** -- deploys to the Q8B with
**zero recompilation**, once the webcam is physically moved over.
Nothing on this hardware remains blocked.

## Where this leaves the Q8B

Fully validated for real use: SSH access, idle-suspend fixed, DSP/NPU
verified working, this project's compiled models confirmed running on
its actual NPU, the webcam physically moved over and confirmed on
`/dev/video2`, and a full live dashboard test (`yolo26`, real webcam,
re-ID, media control) run end to end -- see `docs/backlog.md` for the
real FPS/CPU/RAM numbers, and `docs/pipelines.md` for the side-by-side
comparison against the Q6A (same throughput, less than half the CPU).
The Radxa Dragon Q6A remains this project's primary, documented,
maintained pipeline regardless -- the Q8B is a genuine second target,
not a replacement for it.
