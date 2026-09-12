from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as oq

# The QNN execution provider is registered once per process (onnxruntime's
# registration is process-global, not per-session).
_qnn_registered = False


def _ensure_qnn_registered() -> None:
    global _qnn_registered
    if not _qnn_registered:
        ort.register_execution_provider_library("QNNExecutionProvider", oq.get_library_path())
        _qnn_registered = True


@dataclass(frozen=True)
class QuantSpec:
    """Scale/offset for one QNN-quantized uint8 tensor: real = (raw +
    offset) * scale. AI-Hub-compiled ONNX models expose their I/O as plain
    uint8 tensors with no quantization metadata in the ONNX graph itself
    (the whole compiled model is one opaque EPContext node) -- these values
    must be read once from the compiled model via `qnn-context-binary-utility
    --context_binary=model.bin --json_file=...` and hardcoded here (see
    scripts/qnn_spike.md)."""

    scale: float
    offset: float

    def quantize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x / self.scale - self.offset), 0, 255).astype(np.uint8)

    def dequantize(self, x: np.ndarray) -> np.ndarray:
        return (x.astype(np.float32) + self.offset) * self.scale


class OnnxQnnRunner:
    """Runs an AI-Hub-compiled, QNN-context-wrapped ONNX model on the Q6A's
    Hexagon NPU via onnxruntime's QNN execution provider.

    Replaces an earlier `qnn-net-run`-subprocess-per-call implementation,
    which reloaded the full context binary (~170-190ms) on every single
    call -- ~476ms per `estimate()` end to end, far below the 15-30fps
    target despite the NPU itself executing in ~1-2ms. Loading the model
    once into a persistent `onnxruntime.InferenceSession` and calling
    `.run()` repeatedly avoids that reload entirely: measured ~2.2ms
    (detector) / ~1.1ms (landmark) per warm call on this project's
    physical Q6A, matching Qualcomm AI Hub's own NPU-only profiling
    numbers. See scripts/qnn_spike.md for the full investigation,
    including the `add_provider_for_devices` API needed to attach the QNN
    execution provider to a session (the plain `providers=[...]` argument
    on `InferenceSession` doesn't recognize dynamically-registered plugin
    execution providers).
    """

    def __init__(
        self,
        onnx_path: str,
        input_quant: QuantSpec,
        output_quant: dict[str, QuantSpec],
    ):
        _ensure_qnn_registered()
        qnn_devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
        if not qnn_devices:
            raise RuntimeError("No QNNExecutionProvider device found by onnxruntime")

        session_options = ort.SessionOptions()
        session_options.add_provider_for_devices(
            qnn_devices, {"backend_path": oq.get_qnn_htp_path()}
        )
        self._session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=[])
        self._input_name = self._session.get_inputs()[0].name
        self._input_quant = input_quant
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._output_quant = output_quant

    def run(self, input_float_nhwc: np.ndarray) -> dict[str, np.ndarray]:
        quantized_input = self._input_quant.quantize(input_float_nhwc)
        raw_outputs = self._session.run(None, {self._input_name: quantized_input})
        return {
            name: self._output_quant[name].dequantize(raw)
            for name, raw in zip(self._output_names, raw_outputs)
        }
