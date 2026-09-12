from __future__ import annotations

import cv2
import numpy as np

EMBED_INPUT_SIZE = (256, 128)  # H, W -- OSNet's trained input aspect ratio


class ReidEmbedder:
    """Computes person re-identification appearance embeddings via
    OSNet-x0.25, run in plain FP32 on CPU via onnxruntime's default
    CPUExecutionProvider -- no NPU compile, no quantization, no AI Hub
    account needed.

    This is a deliberately different path from the pose models
    (`inference/backends/qnn.py`), which do need NPU/QNN: appearance
    matching only runs occasionally (when a detection can't be IoU-matched
    to an active track -- see `Tracker`), not for every person on every
    frame like pose estimation, so a ~0.7M-parameter model in plain FP32
    on the ARM CPU is fast enough without that complexity. It also
    sidesteps a real blocker: OSNet's official quantization calibration
    dataset (ENTIReID) requires a manual gated Google Drive download (see
    scripts/qnn_spike.md) -- not worth the accuracy risk of substituting
    ad-hoc calibration images for a model whose own code comments note
    w8a8 quantization is sensitive enough to tank ReID accuracy if
    calibrated poorly.

    Exported via a plain `torch.onnx.export` of
    `qai_hub_models.models.osnet.model.OSNet.from_pretrained("osnet_x0_25")`
    (NCHW, ImageNet normalization baked into the traced graph -- callers
    just pass a raw [0,1] RGB float32 crop). See
    models/osnet_x0_25/README.md.
    """

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def embed(
        self, frame_bgr: np.ndarray, bbox_xyxy_normalized: tuple[float, float, float, float]
    ) -> np.ndarray | None:
        """Returns a 512-dim L2-normalized appearance embedding for the
        person in `bbox_xyxy_normalized` (normalized [0,1] coordinates,
        matching `PoseResult.bbox`), or None if the box is degenerate."""
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = bbox_xyxy_normalized
        px0, py0 = max(0, int(x0 * w)), max(0, int(y0 * h))
        px1, py1 = min(w, int(x1 * w)), min(h, int(y1 * h))
        if px1 - px0 < 2 or py1 - py0 < 2:
            return None

        crop = frame_bgr[py0:py1, px0:px1]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (EMBED_INPUT_SIZE[1], EMBED_INPUT_SIZE[0]))
        input_nchw = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        return self._session.run(None, {self._input_name: input_nchw})[0][0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Both embeddings are L2-normalized, so cosine similarity is just the
    dot product."""
    return float(np.dot(a, b))
