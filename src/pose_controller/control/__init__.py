from __future__ import annotations

from pose_controller.config import ControlConfig
from pose_controller.control.media_controller import MediaController, MediaStatus, NullMediaController

__all__ = ["MediaController", "MediaStatus", "NullMediaController", "build_media_controller"]


def build_media_controller(config: ControlConfig) -> MediaController:
    """Mirrors `app.build_tracker`'s degrade-don't-crash pattern: an
    optional feature (media control, off by default) that reports and
    falls back to a no-op rather than taking the whole app down if the
    backend can't initialize (`playerctl` not installed, no D-Bus
    session, etc.)."""
    if not config.enabled:
        return NullMediaController()

    if config.backend == "playerctl":
        from pose_controller.control.backends.playerctl import PlayerctlController

        try:
            return PlayerctlController(volume_step=config.volume_step)
        except Exception as exc:  # noqa: BLE001 -- report and fall back, don't crash the app
            print(f"Media control disabled: could not start {config.backend!r} backend ({exc})")
            return NullMediaController()

    raise ValueError(f"Unknown media control backend: {config.backend!r}")
