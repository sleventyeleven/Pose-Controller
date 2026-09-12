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
    # "cpu" (dev-loop MediaPipe), "qnn" (on-device YOLOv8n-det +
    # BlazePose-landmark, the current default pipeline), "yolo26"
    # (on-device YOLO26-Pose, a single end-to-end detection+keypoints
    # model), or "hrnet" (on-device YOLOv8n-det + HRNetPose-landmark, a
    # second landmark-stage alternative to BlazePose) -- "yolo26" and
    # "hrnet" are both being evaluated *alongside* "qnn", not replacing it
    # -- see docs/journey.md and docs/backlog.md for why.
    backend: str = "cpu"
    model_dir: str = "models"
    # First-stage person detector for the qnn backend (see
    # inference/backends/_yolo_detect.py). Replaces BlazePose's own bundled
    # 128x128 detector, which was found to lose small/distant subjects at
    # ordinary room camera-to-subject distance -- see scripts/qnn_spike.md.
    detector_model_dir: str = "models/yolov8n_det_qcs6490"
    # Model dir for the yolo26 backend (inference/backends/yolo26_qnn.py).
    # Sourced from Ultralytics' own local QNN export, not Qualcomm AI Hub
    # (which failed to compile this model) -- see that dir's README.md.
    yolo26_model_dir: str = "models/yolo26n_pose_qcs6490"
    # Model dir for the hrnet backend's landmark stage
    # (inference/backends/hrnet_qnn.py) -- uses the same detector_model_dir
    # above as its first stage, same as the "qnn" backend.
    hrnet_model_dir: str = "models/hrnet_pose_qcs6490"
    # General 80-class object detector (yolo26_detect_qnn.Yolo26Detector),
    # staged for dashboard visualization -- NOT part of the pose/gesture
    # pipeline itself, only used when overlay.detect_overlay_enabled is
    # on. See inference/backends/_yolo26_detect.py's module docstring.
    yolo26_det_model_dir: str = "models/yolo26_det_qcs6490"


@dataclass
class OverlayConfig:
    show_window: bool = True
    window_name: str = "Pose-Controller"
    # Local browser dashboard (web/dashboard.py): live overlay feed +
    # gesture queue, for demos/debugging/pipeline iteration without a
    # display attached or a manual capture/scp/inspect cycle -- see
    # docs/backlog.md. Off by default: this binds an HTTP server, which
    # shouldn't happen silently just because the app started.
    web_enabled: bool = False
    web_port: int = 8080
    # Optional general-object-detection overlay (YOLO26-Detection, all 80
    # COCO classes -- see inference/backends/_yolo26_detect.py) drawn
    # alongside the pose skeleton, for visualizing what a general
    # detector sees versus what's being tracked as a person. Off by
    # default: it's an extra NPU model call per frame, purely for
    # visualization/future scene-context work, not needed for the core
    # pipeline. See docs/backlog.md.
    detect_overlay_enabled: bool = False


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
    # 0.6's original value was untested against real footage and turned
    # out to sit right at the edge of real same-person similarity dips --
    # measured on a real recording (scripts/qnn_spike.md): same-person
    # embedding similarity across real gaps got as low as 0.5946 for
    # short (<2s) gaps and 0.5290 across the whole video, both under 0.6,
    # which fragmented one continuous person into multiple track_ids.
    # 0.5 sits just below that measured floor with a small safety margin,
    # confirmed (via the same real-footage test) to collapse most of that
    # fragmentation without reducing gesture-trigger correctness. Not
    # validated against real multi-person footage -- a lower threshold
    # trades some false-merge risk between genuinely different people for
    # this same-person robustness, and that tradeoff is untested here.
    reid_similarity_threshold: float = 0.5
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
