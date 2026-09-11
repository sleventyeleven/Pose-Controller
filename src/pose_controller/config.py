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
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            capture=CaptureConfig(**data.get("capture", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            overlay=OverlayConfig(**data.get("overlay", {})),
        )
