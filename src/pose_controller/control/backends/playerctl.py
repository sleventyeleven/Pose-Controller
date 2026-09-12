"""`playerctl`-backed `MediaController` -- the on-device implementation of
the "emulate OS media keys via D-Bus MPRIS" approach from the original
project plan.

`playerctl` (https://github.com/altdesktop/playerctl, packaged as `apt
install playerctl` on Debian/Ubuntu-derived images including this
project's Radxa boards) was chosen over talking to D-Bus MPRIS
(`org.mpris.MediaPlayer2.Player`) directly for two reasons: it already
handles picking "the" active player when several are running (this
project has no opinion on that policy, and reimplementing it well is
real work), and it avoids adding a D-Bus client library as a pip
dependency (`dbus-python` needs system dev headers to build; pure-Python
alternatives exist but are one more thing to vet) -- shelling out to one
well-tested, single-purpose binary is the smaller footprint for a first
implementation. Works against Spotify's official Linux client (which
exposes MPRIS) with no Spotify-specific code here at all; any other
MPRIS-compliant player works identically.

**Not yet tested against a real player or real hardware** -- this dev
machine has no Linux D-Bus session to test against. Unit tests
(`tests/test_control.py`) mock `subprocess.run` directly; they verify
this module calls the right `playerctl` subcommands, not that
`playerctl`/D-Bus/Spotify actually respond correctly on a real device.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time

from pose_controller.control.media_controller import MediaController, MediaStatus

_SUBPROCESS_TIMEOUT_S = 2.0  # a hung D-Bus call must never block a gesture trigger
_STATUS_POLL_INTERVAL_S = 2.0  # dashboard display only -- no need to poll faster than a human reads it


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["playerctl", *args], capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S
    )


class PlayerctlController(MediaController):
    """`volume_step` is the fraction of full volume applied per
    VOLUME_UP/DOWN trigger (0.05 = 5%, matching typical OS media-key
    granularity). `poll_status` controls whether a background thread
    polls `playerctl status`/`metadata` every `_STATUS_POLL_INTERVAL_S`
    for `get_status()` (the web dashboard's media panel) -- disable it
    (e.g. in tests) when nothing reads status.

    Raises `FileNotFoundError` if the `playerctl` binary itself isn't on
    `PATH`, so `control.build_media_controller`'s caller can catch it and
    fall back to `NullMediaController` the same way `app.build_tracker`
    falls back when `ReidEmbedder` fails to load -- report and don't
    crash the app over an optional feature.
    """

    def __init__(self, volume_step: float = 0.05, poll_status: bool = True):
        if shutil.which("playerctl") is None:
            raise FileNotFoundError(
                "playerctl not found on PATH -- install it with 'sudo apt install playerctl'"
            )
        self._volume_step = volume_step
        self._status_lock = threading.Lock()
        self._status = MediaStatus(available=False, backend_name="playerctl")
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        if poll_status:
            self._refresh_status()  # one synchronous poll so get_status() isn't stale on the first call
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def _poll_loop(self) -> None:
        while not self._stop.wait(_STATUS_POLL_INTERVAL_S):
            self._refresh_status()

    def _refresh_status(self) -> None:
        try:
            status_result = _run("status")
            if status_result.returncode != 0:
                # Typically "No players found" -- playerctl itself works, nothing to control.
                with self._status_lock:
                    self._status = MediaStatus(available=False, backend_name="playerctl")
                return
            playing = status_result.stdout.strip() == "Playing"

            meta_result = _run("metadata", "--format", "{{title}}\t{{artist}}")
            title, artist = "", ""
            if meta_result.returncode == 0 and "\t" in meta_result.stdout:
                title, artist = meta_result.stdout.rstrip("\n").split("\t", 1)

            with self._status_lock:
                self._status = MediaStatus(
                    available=True, backend_name="playerctl", playing=playing, title=title, artist=artist
                )
        except (subprocess.SubprocessError, OSError) as exc:  # noqa: BLE001 -- status display only, never fatal
            print(f"[control] playerctl status poll failed: {exc}")
            with self._status_lock:
                self._status = MediaStatus(available=False, backend_name="playerctl")

    def _fire(self, *args: str) -> None:
        try:
            _run(*args)
        except (subprocess.SubprocessError, OSError) as exc:  # noqa: BLE001 -- a missed gesture action shouldn't crash the app
            print(f"[control] playerctl {' '.join(args)} failed: {exc}")

    def play_pause(self) -> None:
        self._fire("play-pause")

    def next(self) -> None:
        self._fire("next")

    def previous(self) -> None:
        self._fire("previous")

    def volume_up(self) -> None:
        self._fire("volume", f"{self._volume_step}+")

    def volume_down(self) -> None:
        self._fire("volume", f"{self._volume_step}-")

    def get_status(self) -> MediaStatus:
        with self._status_lock:
            return self._status

    def close(self) -> None:
        self._stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=_STATUS_POLL_INTERVAL_S)
