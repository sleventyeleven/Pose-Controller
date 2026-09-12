import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pose_controller.config import ControlConfig
from pose_controller.control import NullMediaController, build_media_controller
from pose_controller.control.media_controller import MediaController, MediaStatus
from pose_controller.gestures.types import ControlAction


class _RecordingController(MediaController):
    """Minimal concrete MediaController that just records which method
    was called, for testing `dispatch`'s action->method mapping in
    isolation from any real backend."""

    def __init__(self):
        self.calls: list[str] = []

    def play_pause(self) -> None:
        self.calls.append("play_pause")

    def next(self) -> None:
        self.calls.append("next")

    def previous(self) -> None:
        self.calls.append("previous")

    def volume_up(self) -> None:
        self.calls.append("volume_up")

    def volume_down(self) -> None:
        self.calls.append("volume_down")

    def get_status(self) -> MediaStatus:
        return MediaStatus(available=True, backend_name="recording")


@pytest.mark.parametrize(
    "action,expected_method",
    [
        (ControlAction.NEXT, "next"),
        (ControlAction.SKIP, "next"),  # two different gestures, same underlying media action
        (ControlAction.PREVIOUS, "previous"),
        (ControlAction.PLAY_PAUSE, "play_pause"),
        (ControlAction.VOLUME_UP, "volume_up"),
        (ControlAction.VOLUME_DOWN, "volume_down"),
    ],
)
def test_dispatch_maps_every_control_action_to_the_right_method(action, expected_method):
    controller = _RecordingController()

    controller.dispatch(action)

    assert controller.calls == [expected_method]


def test_null_media_controller_dispatch_is_a_noop_for_every_action():
    controller = NullMediaController()

    for action in ControlAction:
        controller.dispatch(action)  # must not raise

    assert controller.get_status() == MediaStatus(available=False, backend_name="none")


def test_build_media_controller_returns_null_when_disabled():
    controller = build_media_controller(ControlConfig(enabled=False))

    assert isinstance(controller, NullMediaController)


def test_build_media_controller_raises_for_unknown_backend():
    with pytest.raises(ValueError):
        build_media_controller(ControlConfig(enabled=True, backend="not-a-real-backend"))


def test_build_media_controller_falls_back_to_null_when_playerctl_missing(capsys):
    with patch("shutil.which", return_value=None):
        controller = build_media_controller(ControlConfig(enabled=True, backend="playerctl"))

    assert isinstance(controller, NullMediaController)
    assert "Media control disabled" in capsys.readouterr().out


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestPlayerctlController:
    """`playerctl` itself is never actually invoked here -- `subprocess.run`
    is mocked throughout, since this dev machine has no Linux D-Bus
    session to test against. These tests verify this module calls the
    right `playerctl` subcommands with the right arguments, not that
    `playerctl`/D-Bus/a real player actually respond correctly on real
    hardware -- see docs/backlog.md."""

    def _make_controller(self, **kwargs):
        from pose_controller.control.backends.playerctl import PlayerctlController

        with patch("shutil.which", return_value="/usr/bin/playerctl"):
            with patch("subprocess.run", return_value=_completed()):
                return PlayerctlController(poll_status=False, **kwargs)

    def test_raises_file_not_found_when_playerctl_binary_missing(self):
        from pose_controller.control.backends.playerctl import PlayerctlController

        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError):
                PlayerctlController(poll_status=False)

    def test_play_pause_calls_playerctl_play_pause(self):
        controller = self._make_controller()
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            controller.play_pause()

        args = mock_run.call_args[0][0]
        assert args == ["playerctl", "play-pause"]

    def test_next_and_previous_call_correct_subcommands(self):
        controller = self._make_controller()
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            controller.next()
            controller.previous()

        called_args = [call[0][0] for call in mock_run.call_args_list]
        assert ["playerctl", "next"] in called_args
        assert ["playerctl", "previous"] in called_args

    def test_volume_up_and_down_apply_the_configured_step(self):
        controller = self._make_controller(volume_step=0.1)
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            controller.volume_up()
            controller.volume_down()

        called_args = [call[0][0] for call in mock_run.call_args_list]
        assert ["playerctl", "volume", "0.1+"] in called_args
        assert ["playerctl", "volume", "0.1-"] in called_args

    def test_action_failure_is_caught_and_does_not_raise(self, capsys):
        controller = self._make_controller()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="playerctl", timeout=2.0)):
            controller.next()  # must not raise

        assert "failed" in capsys.readouterr().out

    def test_refresh_status_reports_unavailable_when_no_players_found(self):
        controller = self._make_controller()
        with patch("subprocess.run", return_value=_completed(returncode=1, stderr="No players found")):
            controller._refresh_status()

        status = controller.get_status()
        assert status.available is False
        assert status.backend_name == "playerctl"

    def test_refresh_status_reports_now_playing_metadata(self):
        controller = self._make_controller()

        def fake_run(args, **kwargs):
            if args[1] == "status":
                return _completed(stdout="Playing\n")
            return _completed(stdout="Song Title\tSome Artist\n")

        with patch("subprocess.run", side_effect=fake_run):
            controller._refresh_status()

        status = controller.get_status()
        assert status.available is True
        assert status.playing is True
        assert status.title == "Song Title"
        assert status.artist == "Some Artist"

    def test_close_stops_the_poll_thread_without_error(self):
        from pose_controller.control.backends.playerctl import PlayerctlController

        with patch("shutil.which", return_value="/usr/bin/playerctl"):
            with patch("subprocess.run", return_value=_completed()):
                controller = PlayerctlController(poll_status=True)

        controller.close()  # must not raise or hang
