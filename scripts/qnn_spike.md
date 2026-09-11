# NPU spike plan (milestone 1)

Goal: get *any* model running through the Q6A's Hexagon NPU from Python,
before implementing `inference/backends/qnn.py` for real. This is the
highest-risk unknown in the project (per `docs/hardware.md`) and needs to
happen on the physical board.

Suggested steps (to run on the Q6A itself, e.g. over SSH):

1. Confirm the board's OS image and Python version (`cat /etc/os-release`,
   `python3 --version`).
2. Install `qai-hub-models` and check whether it can target QCS6490 directly,
   or whether models need to be compiled via the Qualcomm AI Hub cloud
   service first (`pip install qai-hub qai-hub-models`, then
   `qai-hub configure` and check `qai_hub.get_devices()` for a QCS6490
   entry).
3. Export/compile a small model (start with a person detector, e.g.
   `qai_hub_models.models.yolov8_det`) targeting QCS6490 and download the
   resulting QNN context binary / `.so`.
4. Determine the runtime: try `onnxruntime` with the QNN execution provider
   first (`pip install onnxruntime-qnn` if it exists for this platform);
   fall back to AI Hub's own on-device inference harness
   (`qai_hub_models.models.<model>.demo` or `App` classes) if not.
5. Run inference on a static test image and confirm output shape/values are
   sane and that the NPU (not the CPU fallback) is actually being used --
   check `qcom` diagnostics or process/power monitoring during inference to
   distinguish NPU from CPU execution.
6. Repeat for a pose model (`qai_hub_models.models.mediapipe_pose` or
   similar) once the detector round-trip works.

Once this works end to end, port the working incantation into
`inference/backends/qnn.py`, matching the `PoseEstimator` interface in
`inference/pose.py`.

This step requires the physical Q6A and is not something that can be done
from a plain dev machine -- run it and report back what worked so the
`qnn.py` stub can be filled in.
