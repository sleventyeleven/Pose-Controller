from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CaptureConfig:
    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class InferenceConfig:
    backend: str = "cpu"
    model_dir: str = "models"


@dataclass
class OverlayConfig:
    show_window: bool = True
    window_name: str = "Pose-Controller"


@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 1
    # Appearance-based re-ID (tracking.ReidEmbedder) so a track's ID
    # survives fully leaving and re-entering frame, not just brief
    # occlusion -- see docs/backlog.md / scripts/qnn_spike.md. Disable to
    # fall back to pure IoU/Kalman tracking (e.g. if onnxruntime isn't
    # installed).
    reid_enabled: bool = True
    reid_model_path: str = "models/osnet_x0_25/model.onnx"
    reid_similarity_threshold: float = 0.6
    reid_gallery_ttl: int = 300


@dataclass
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            capture=CaptureConfig(**data.get("capture", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            overlay=OverlayConfig(**data.get("overlay", {})),
            tracking=TrackingConfig(**data.get("tracking", {})),
        )
