# Q6A bring-up steps

Step-by-step record of what it actually took to get a fresh Radxa Dragon
Q6A ready for this project, worked out by hand over SSH on 2026-09-11.
`docs/backlog.md` tracks turning this into an automated
`scripts/setup_device.sh` once it's fully validated -- for now, follow it
manually.

**Status as of 2026-09-11: NPU confirmed working.** The board's original
OS was on Radxa's `qcs6490-noble-test` apt channel (a rolling dev channel,
not the stable release), which was the root cause of a long chain of NPU
bring-up failures. Fixed by:

1. Reflashing to the **r2** stable image
   ([GitHub release](https://github.com/radxa-build/radxa-dragon-q6a/releases/tag/rsdk-r2)),
   UFS variant (`radxa-dragon-q6a_noble_gnome_r2.output_4096.img.xz`) --
   this also resolved the microSD-vs-UFS boot-media question in
   `docs/backlog.md`.
2. Updating the board's BIOS/UEFI firmware to build `260120` or later
   (r2 requires `20251230`+) via EDL mode + `edl-ng` -- see "BIOS firmware
   update" below.
3. `sudo apt install -y fastrpc fastrpc-test fastrpc-dev libcdsprpc1 radxa-firmware-qcs6490`
   then reboot. (The critical package we were missing in earlier attempts
   was `radxa-firmware-qcs6490` specifically.)

Confirmed with `fastrpc_test -a v68`: all 3 test modules PASS, both
`adsp`/`cdsp` remoteprocs report `running`, DSP-computed result verified
(`Sum = 499500` for 0..999 computed on-device). Sections 1-3 below (SSH
access, idle-suspend fix, both cameras) applied cleanly after reflashing
with no changes needed. The long NPU debugging trail from the
`noble-test` image is kept at the bottom of this file as a "how we got
here" record, since it explains *why* the fix is what it is -- start with
"4. Qualcomm AI Runtime (QNN/NPU) -- confirmed working recipe" below for
the actual steps to reproduce.

## 1. Network + SSH access

The board was flaky on Wi-Fi (IP kept changing/dropping) -- a wired
connection with a DHCP reservation was far more reliable for development.

1. Connect the board via Ethernet and set a DHCP reservation for it on your
   router so its IP stays stable.
2. Default credentials on a fresh image: `radxa` / `radxa`.
3. Install a dedicated SSH key for this project rather than typing the
   password repeatedly:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/pose_controller_q6a -N "" -C "pose-controller-dev@q6a"
   # copy the pubkey into the board's authorized_keys (one-time, password auth):
   ssh-copy-id -i ~/.ssh/pose_controller_q6a.pub radxa@<board-ip>
   ```

4. Add an SSH config alias so it's just `ssh q6a`:

   ```
   Host q6a
       HostName <board-ip>
       User radxa
       IdentityFile ~/.ssh/pose_controller_q6a
   ```

Confirmed board identity: Ubuntu 24.04.3 LTS, kernel `6.17.1-2-qcom`,
Python 3.12.3, aarch64. Storage: no onboard eMMC -- boots off a 64GB
microSD card (`mmcblk1`, ~56GB usable); a 128GB UFS module is also attached
as `sda` but unused as boot media (see `docs/storage-footprint.md` and
`docs/backlog.md`).

## 1a. Disable idle suspend (do this early)

The stock image is a desktop-flavored install with GNOME's idle-suspend
enabled (`sleep-inactive-ac-timeout` = 900s / 15 min, type `suspend`) --
this device needs to run unattended and headless, so a 15-minute idle
timer that suspends the whole machine (dropping the network along with it)
is actively harmful. It bit us once: mid-troubleshooting over SSH, the
board went completely unreachable (ping and SSH both dead) for several
minutes. SSH/CPU activity does **not** reset GNOME's local-input idle
timer, so purely remote work never prevents this -- it looks identical to
a hang or reboot from the network side, and looks like a frozen screen if
a monitor's attached. (Confirmed this was suspend, not a crash or thermal
shutdown: all thermal zones read 37-44degC at the time, nowhere near this
chip's throttling range.)

Fix (do this right after first SSH access, before anything else):

```bash
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

The `gsettings` line disables GNOME's idle-suspend trigger; the
`systemctl mask` is a hard backstop at the systemd level so *no* suspend
request can succeed regardless of what triggers it (GNOME, a power
button, ACPI, etc.) -- appropriate since this device should behave as an
always-on embedded appliance, never a sleeping desktop.

## 2. Nexigo USB webcam

No setup needed. It's a standard UVC device:

```bash
sudo apt-get install -y v4l-utils   # not installed by default
v4l2-ctl --list-devices
```

Confirmed output:

```
NexiGo N60 FHD Webcam : NexiGo  (usb-xhci-hcd.1.auto-1.1):
	/dev/video0
	/dev/video1
	/dev/media0
```

`/dev/video0` is the one to point `capture.source` at in
`configs/dragon_q6a.yaml`.

## 3. Radxa Camera 4K (MIPI CSI)

Not enabled by default -- on a fresh image `dmesg | grep -iE
'camss|csi|imx415'` returns nothing at all, meaning the CSI capture
subsystem isn't even attempting to probe.

The fix is a device-tree overlay. The `radxa-overlays` DKMS package (already
installed on the stock image; check with `dkms status | grep radxa-overlay`)
ships the exact overlay for this camera:

- `qcs6490-radxa-dragon-q6a-cam2-radxa-camera-8m-219.dtbo`
- `qcs6490-radxa-dragon-q6a-cam3-radxa-camera-8m-219.dtbo`

(`cam1-imx577` is a different sensor -- not this camera. `cam2`/`cam3`
correspond to which physical MIPI CSI port the camera is plugged into.)

Enable it with Radxa's `rsetup` tool:

```bash
sudo rsetup
```

`rsetup` is an interactive (whiptail/ncurses) menu -- run it from a real
terminal session (SSH is fine as long as it's an interactive `ssh`, not a
piped/non-interactive command), not scripted blindly. Navigate to the
overlay/hardware configuration section and enable the `cam2-radxa-camera-8m`
or `cam3-radxa-camera-8m` overlay matching the CSI port the camera is
physically connected to, then let it apply and **reboot**.

`rsetup` also has a non-TUI CLI form (`rsetup <function> <args>`, e.g.
`rsetup enable_overlays <name>`) that in principle can be scripted, but its
overlay-name resolution expects the target `.dtbo`/`.dtbo.disabled` file to
already exist in the boot-managed overlay directory (default
`/boot/dtbo/`), which isn't populated by just installing the DKMS package --
some rebuild step populates it and wasn't fully traced down. Use the
interactive TUI for now; this is exactly the gap the backlog's automated
setup script needs to close.

After reboot, verify:

```bash
dmesg | grep -iE 'camss|imx415'
v4l2-ctl --list-devices
```

## 3a. BIOS firmware update (do this before the NPU steps)

r2 requires BIOS/UEFI firmware build `20251230` or later. Check the
current version:

```bash
cat /sys/class/dmi/id/bios_version
```

If it's older, update it -- this needs physical access and a host PC, not
something doable purely over SSH (the board isn't running its OS during
this):

1. Download the latest BIOS firmware zip (e.g.
   `dragon-q6a_flat_build_wp_260120.zip`) and the `edl-ng` tool from
   https://docs.radxa.com/en/dragon/q6a/download
2. Enter EDL mode: hold the **EDL button**, power on the board, release
   the EDL button once powered.
3. Verify EDL mode from the host PC: `lsusb` (Linux) or Device Manager
   (Windows) should show a Qualcomm EDL/download-mode device.
4. Unzip the firmware package (contains `prog_firehose_ddr.elf`,
   `rawprogram0.xml`, `patch0.xml`), then from that directory:
   ```bash
   sudo edl-ng --memory=spinor rawprogram rawprogram0.xml patch0.xml --loader=prog_firehose_ddr.elf
   ```
   (Windows: `.\edl-ng.exe --memory=spinor --loader prog_firehose_ddr.elf rawprogram rawprogram0.xml patch0.xml`)
5. Power cycle. Re-check `cat /sys/class/dmi/id/bios_version`.

Confirmed working: flashed `dragon-q6a_flat_build_wp_260120.zip`
successfully (`'rawprogram' command finished successfully`, all
partitions written), verified post-flash as
`6.0.260120.BOOT.MXF.1.0.1-00549-KODIAKWP-1`.

**Note:** SSH host keys and any previously-installed SSH keys are
board/OS-state, not BIOS-state, but a full OS reflash (section above)
wipes them -- expect a "REMOTE HOST IDENTIFICATION HAS CHANGED" warning
and `ssh-keygen -R <ip>` plus re-running the SSH key setup in section 1
after any reflash.

## 4. Qualcomm AI Runtime (QNN/NPU) -- confirmed working recipe

On the r2 image with up-to-date BIOS firmware:

```bash
sudo apt update
sudo apt install -y fastrpc fastrpc-test fastrpc-dev libcdsprpc1 radxa-firmware-qcs6490
sudo reboot
```

The critical package that's easy to miss is **`radxa-firmware-qcs6490`**
-- without it, `fastrpc_test` fails even on r2 with BIOS updated (this bit
us once; `apt install fastrpc libcdsprpc1` alone, as an earlier reading of
Radxa's docs suggested, was not sufficient in practice). After rebooting:

```bash
fastrpc_test -a v68
```

Confirmed working output: all 3 test modules (`libcalculator.so`,
`libhap_example.so`, `libmultithreading.so`) PASS, `RESULT: All applicable
tests PASSED`, both `adsp`/`cdsp` remoteprocs (`cat
/sys/class/remoteproc/remoteproc*/state`) report `running`.

### Qualcomm AI Hub Models -- model list and workflow

Per [Radxa's QAI Hub Models docs](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qai-hub-models),
these pose estimation models are confirmed to work on this chipset via
Qualcomm AI Hub (this is the source of truth for `inference/backends/qnn.py`
-- pick from this list rather than assuming a model export will work):

- `qai_hub_models.models.mediapipe_pose` (MediaPipe Pose -- what
  `inference/backends/cpu.py` already uses for the dev backend, so this is
  the natural first target to keep both backends producing the same
  landmark schema)
- `qai_hub_models.models.movenet`
- `qai_hub_models.models.hrnet_pose`
- `qai_hub_models.models.litehrnet`
- `qai_hub_models.models.posenet_mobilenet`
- `qai_hub_models.models.rtmpose_body2d`
- `qai_hub_models.models.facemap_3dmm` (facial landmarks, not relevant here)

Workflow (compilation happens via Qualcomm's cloud AI Hub service, not
on-device):

```bash
pip3 install qai_hub_models
# register at https://aihub.qualcomm.com/ first, then:
qai-hub configure --api_token API_TOKEN

export PRODUCT_CHIP=qualcomm-qcs6490-proxy
python3 -m qai_hub_models.models.mediapipe_pose.export \
  --chipset ${PRODUCT_CHIP} \
  --target-runtime qnn_context_binary \
  --quantize w8a8

python3 -m qai_hub_models.models.mediapipe_pose.demo \
  --eval-mode on-device \
  --hub-model-id <ID-from-export-step> \
  --chipset ${PRODUCT_CHIP}
```

`--quantize w8a8` (INT8 weights+activations) lines up with the
`docs/storage-footprint.md` stretch goal -- smaller and faster than FP32
on the Hexagon NPU, not just a storage saving. Requires a (free) Qualcomm
AI Hub account for the `qai-hub configure` API token.

## Appendix: debugging trail on the original `noble-test` image

Kept for context on *why* the confirmed recipe above is what it is --
not needed to reproduce the working setup, skip unless curious or unless
something regresses.

The board's original image had a Qualcomm-maintained apt PPA configured
(`ubuntu-qcom-iot/qcom-ppa`), which carries the modern "QAIRT" (Qualcomm AI
Runtime -- the unified successor to separate QNN/SNPE SDKs) packages
directly, no manual SDK download needed:

```bash
sudo apt-get install -y qairt-libs qairt-tools qairt-headers qairt-dsp-binaries
```

- `qairt-libs` / `qairt-headers` -- runtime libraries and dev headers.
- `qairt-tools` -- binary tools (`qnn-net-run`/`snpe-net-run`-equivalent,
  per the package's `Provides:` on the legacy `qnn-tools`/`snpe-tools`
  names).
- `qairt-dsp-binaries` -- the actual Hexagon DSP execution binaries; this is
  what makes inference actually run on the NPU rather than falling back to
  CPU.

There's also an empty Radxa meta-package, `radxa-qcom-qairt`, suggesting
Radxa intends to formalize this as the blessed install path -- worth
watching for updates.

**Note:** installing `qairt-dsp-binaries` can leave the install
half-finished (`dpkg -l` shows `iU`/`iHR` status) if anything interrupts it
mid-way -- it pulls in a large `linux-firmware-*` chain and regenerates the
initramfs/bootloader entry, which takes a while. If that happens, finish it
non-interactively rather than re-running plain `apt-get install`:

```bash
sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a dpkg --configure -a
sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y qairt-libs qairt-tools qairt-headers qairt-dsp-binaries
```

`qairt-tools` includes real QNN CLI binaries once installed: `qnn-net-run`,
`qnn-context-binary-generator`, `qnn-platform-validator`, plus the
`snpe-*`/`genie-*` equivalents. `qairt-libs` includes `libQnnHtp.so` (the
Hexagon NPU/HTP backend). DSP skeleton binaries land under
`/usr/share/qcom/qcm6490/<vendor>/<board>/dsp/cdsp/` -- there's no
Radxa-specific directory, only `Thundercomm/RB3gen2` and
`Thundercomm/RubikPi3` (both QCS6490-family reference boards); these are
silicon/DSP-firmware-version specific rather than truly board-branded, so
the RB3gen2 set is the one to point `ADSP_LIBRARY_PATH` at:

```bash
export ADSP_LIBRARY_PATH=/usr/share/qcom/qcm6490/Thundercomm/RB3gen2/dsp/cdsp
```

### DSP bring-up attempts on `noble-test`

`qnn-platform-validator --backend dsp --testBackend` initially failed with
`Backend Hardware: Supported / Backend Libraries: Found / Unit Test: Failed`
and `Please use testsig if using unsigned images` -- that error text is
misleading (it's QNN's generic fallback message for basically any FastRPC
failure, not specifically a signing problem). The real chain of issues,
found by digging into `dmesg` and cross-referencing community reports of
the same board (search "Radxa Dragon Q6A FastRPC" -- notably a Radxa forum
thread titled "Non-functional NPU Support on Dragon Q6A (QCS6490) under
Ubuntu 24.04 (Noble)" and a couple of detailed GitHub gists by "Foadsf"):

1. **`/dev/fastrpc-*` device nodes didn't exist at all.** `ls
   /sys/class/remoteproc/*/state` showed both `adsp` (remoteproc0) and
   `cdsp` (remoteproc1) as `offline` -- the DSP firmware never auto-boots
   on this image. Fixed for the current boot with:
   ```bash
   sudo modprobe fastrpc
   echo start | sudo tee /sys/class/remoteproc/remoteproc0/state  # adsp
   echo start | sudo tee /sys/class/remoteproc/remoteproc1/state  # cdsp
   ```
   This alone failed the first time with `dmesg` showing `Direct firmware
   load for qcom/qcs6490/radxa/dragon-q6a/adsp.mbn failed with error -2`
   (ENOENT) -- **root cause: the actual firmware blobs are installed by
   `linux-firmware-dragonwing` at the generic path
   `/lib/firmware/updates/qcom/qcs6490/{adsp,cdsp}.mbn`, but the Q6A's
   device tree requests them at the board-specific path
   `qcom/qcs6490/radxa/dragon-q6a/{adsp,cdsp}.mbn`, which Radxa's packaging
   hasn't caught up to yet.** Fixed by symlinking (the base ADSP/CDSP
   firmware is SoC-specific, not really board-specific -- consistent with
   the userspace DSP skel libraries also being reused generically across
   QCS6490 boards, see above):
   ```bash
   sudo mkdir -p /lib/firmware/qcom/qcs6490/radxa/dragon-q6a
   sudo ln -s /lib/firmware/updates/qcom/qcs6490/adsp.mbn /lib/firmware/qcom/qcs6490/radxa/dragon-q6a/adsp.mbn
   sudo ln -s /lib/firmware/updates/qcom/qcs6490/cdsp.mbn /lib/firmware/qcom/qcs6490/radxa/dragon-q6a/cdsp.mbn
   ```
   After this, both remoteprocs reach `state: running` and
   `/dev/fastrpc-adsp`, `/dev/fastrpc-cdsp`, `/dev/fastrpc-cdsp-secure`
   exist (owned by a new `fastrpc` group -- add your user to it:
   `sudo usermod -aG fastrpc $USER`, then re-login or `sg fastrpc -c ...`).

2. **Still blocked after that.** `qnn-platform-validator` still failed the
   same way (`error -6` executing the DSP calculator stub's sum function).
   `dmesg` showed `qcom,fastrpc ...: no reserved DMA memory for FASTRPC`
   during remoteproc start.

**Root cause, confirmed:** `apt-cache policy fastrpc` showed the package
source as `https://radxa-repo.github.io/qcs6490-noble-test` -- a rolling
test/dev channel, not the `r2` stable release. Reflashing to r2 (UFS
variant) plus the BIOS firmware update plus the confirmed-working package
set in section 4 above resolved it completely -- `fastrpc_test -a v68`
now passes cleanly. This whole appendix is historical only.

<!-- TODO: run the qai_hub_models workflow (section 4 above) against
     mediapipe_pose end to end on the Hexagon NPU, then fill in
     inference/backends/qnn.py accordingly. -->
