import http.client
import json
import urllib.error
import urllib.request

import numpy as np

from pose_controller.web import DashboardServer, DashboardState
from pose_controller.web.dashboard import MAX_EVENTS


def _start_server_with_frame():
    state = DashboardState()
    state.update_frame(np.zeros((10, 10, 3), dtype=np.uint8))
    server = DashboardServer(state, host="127.0.0.1", port=0)
    server.start()
    return state, server


def test_dashboard_state_frame_version_increments_on_update():
    state = DashboardState()
    assert state.get_frame_version() == 0
    assert state.get_frame() is None

    state.update_frame(np.zeros((4, 4, 3), dtype=np.uint8))

    assert state.get_frame_version() == 1
    assert state.get_frame() is not None


def test_dashboard_state_events_are_bounded_to_max_events():
    state = DashboardState()
    for i in range(MAX_EVENTS + 10):
        state.add_event(track_id=i, action_name="NEXT")

    events = state.get_events()

    assert len(events) == MAX_EVENTS
    assert events[0]["track_id"] == 10  # oldest 10 dropped


def test_index_serves_html_referencing_the_stream_endpoint():
    state, server = _start_server_with_frame()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/") as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
        assert "<html" in body.lower()
        assert "/stream" in body
        assert "/events" in body
    finally:
        server.stop()


def test_events_endpoint_returns_added_events_as_json():
    state, server = _start_server_with_frame()
    try:
        state.add_event(track_id=3, action_name="SKIP")

        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/events") as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "application/json"
            events = json.loads(resp.read().decode("utf-8"))

        assert len(events) == 1
        assert events[0]["track_id"] == 3
        assert events[0]["action"] == "SKIP"
        assert "t" in events[0]
    finally:
        server.stop()


def test_stream_endpoint_serves_multipart_jpeg_with_pending_frame():
    state, server = _start_server_with_frame()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        conn.request("GET", "/stream")
        resp = conn.getresponse()
        assert resp.status == 200
        assert "multipart/x-mixed-replace" in resp.getheader("Content-Type")

        chunk = resp.read(64)
        assert chunk.startswith(b"--posecontrollerframe")
        conn.close()
    finally:
        server.stop()


def test_unknown_path_returns_404():
    state, server = _start_server_with_frame()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}/nope")
            assert False, "expected an HTTPError"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.stop()
