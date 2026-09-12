"""Abstract media-control interface -- the piece explicitly deferred when
`ControlAction` was first introduced (see `gestures/types.py`'s
docstring: "that's control/'s job (milestone 6, not yet built)").
Gesture recognition only ever decides *that* an action was triggered;
this module decides how it's actually carried out against a real media
player.

Per the original project plan, the first (and so far only) backend
emulates OS media keys via Linux D-Bus MPRIS
(`org.mpris.MediaPlayer2.Player`) rather than talking to Spotify's own
Web API directly -- this keeps the *trigger* path fully offline and
player-agnostic (works against Spotify's official Linux client, or any
other MPRIS-compliant player already running and authenticated) even
though Spotify itself still needs network access to actually stream.
See `backends/playerctl.py` for why `playerctl` specifically.

**Not yet validated against real hardware or a real player** -- built
and unit-tested (with subprocess calls mocked) on the dev machine only,
which has no Linux D-Bus session to test against. See `docs/backlog.md`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pose_controller.gestures.types import ControlAction

# SKIP (the overhead-sweep gesture) and NEXT (arm-out-to-side) are two
# different physical gestures for the same underlying media action --
# see gestures/state_machine.py's docstring for why both exist -- so both
# map to next() here. There is no dedicated "skip forward N seconds"
# concept anywhere in this project's gesture vocabulary.
_ACTION_TO_METHOD_NAME: dict[ControlAction, str] = {
    ControlAction.NEXT: "next",
    ControlAction.SKIP: "next",
    ControlAction.PREVIOUS: "previous",
    ControlAction.PLAY_PAUSE: "play_pause",
    ControlAction.VOLUME_UP: "volume_up",
    ControlAction.VOLUME_DOWN: "volume_down",
}


@dataclass
class MediaStatus:
    """A snapshot of the controlled player's state, for display (the web
    dashboard's media panel) rather than control -- nothing in this
    project's gesture/tracking logic reads this. `available=False` means
    no backend is reachable at all (nothing to control), distinct from
    `playing=False` (a real player exists but is paused/stopped)."""

    available: bool
    backend_name: str
    playing: bool = False
    title: str = ""
    artist: str = ""


class MediaController(ABC):
    """Backend-agnostic media control, driven by `ControlAction`s from
    `gestures.GestureStateMachine`.

    `dispatch` is the only method `app.py` needs to call -- it owns the
    `ControlAction` -> concrete-method mapping so that mapping lives in
    one place rather than being duplicated at every call site. Backends
    implement the five concrete methods and `get_status`; a backend that
    fails partway through (player closed mid-session, D-Bus hiccup)
    should log and return rather than raise, matching this project's
    existing degrade-don't-crash convention (`app.build_tracker`'s
    `ReidEmbedder` fallback).
    """

    def dispatch(self, action: ControlAction) -> None:
        method_name = _ACTION_TO_METHOD_NAME[action]
        getattr(self, method_name)()

    @abstractmethod
    def play_pause(self) -> None: ...

    @abstractmethod
    def next(self) -> None: ...

    @abstractmethod
    def previous(self) -> None: ...

    @abstractmethod
    def volume_up(self) -> None: ...

    @abstractmethod
    def volume_down(self) -> None: ...

    @abstractmethod
    def get_status(self) -> MediaStatus: ...

    def close(self) -> None:
        pass


class NullMediaController(MediaController):
    """No-op backend for `control.enabled: false` (the default) or when
    the real backend failed to initialize -- lets `app.py` call
    `dispatch()` unconditionally instead of null-checking at every
    trigger site."""

    def play_pause(self) -> None:
        pass

    def next(self) -> None:
        pass

    def previous(self) -> None:
        pass

    def volume_up(self) -> None:
        pass

    def volume_down(self) -> None:
        pass

    def get_status(self) -> MediaStatus:
        return MediaStatus(available=False, backend_name="none")
