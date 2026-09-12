"""Lightweight local web dashboard: streams the live overlay frame, a
gesture-trigger queue, and (if `control.enabled`) the connected media
player's status, to a browser, for demos, debugging, and pipeline
iteration -- see docs/backlog.md for why this exists (every finding this
project made about exposure, orientation, ROI tuning, and re-ID
thresholds went through a manual capture-on-device -> scp -> inspect
cycle; this replaces that with a live view for anything that doesn't
need per-pixel inspection).

Uses only the standard library's `http.server` (no Flask or similar) --
consistent with this project's existing minimal-dependency preference
(see models/osnet_x0_25/README.md for why re-ID stays on plain CPU
onnxruntime rather than pulling in more complexity than the occasional
use pattern needs; the same reasoning applies here: this is a debug/demo
tool, not a piece of the real-time inference path, so it doesn't need a
web framework's routing/templating machinery).

Not authenticated. Meant for a deliberately-started, LAN-local demo/debug
session -- see `AppConfig.overlay.web_enabled` (off by default) -- not
for exposing a camera feed beyond that without further thought.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from pose_controller.control import MediaStatus

MAX_EVENTS = 100  # gesture queue history kept for late-joining/refreshing browser tabs
MJPEG_BOUNDARY = "posecontrollerframe"
_STREAM_POLL_INTERVAL_S = 0.02  # how often the stream handler checks for a new frame

_INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Pose-Controller Dashboard</title>
<style>
  html, body { margin: 0; height: 100%; }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: #111417; color: #e8e8e8;
    display: flex; flex-direction: row;
  }
  #video-pane {
    flex: 1; display: flex; align-items: center; justify-content: center;
    background: #000; min-width: 0;
  }
  #video-pane img { max-width: 100%; max-height: 100vh; display: block; }
  #queue-pane {
    width: 320px; flex-shrink: 0; border-left: 1px solid #2a2f36;
    padding: 14px; overflow-y: auto; box-sizing: border-box;
  }
  #queue-pane h1 {
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em;
    color: #8a8f98; margin: 0 0 12px;
  }
  .event {
    padding: 8px 10px; margin-bottom: 6px; background: #1b1f24;
    border-radius: 6px; font-size: 13px; border-left: 3px solid #3b82f6;
  }
  .event .time { color: #8a8f98; margin-right: 8px; font-variant-numeric: tabular-nums; }
  .event .track { color: #8a8f98; margin-right: 6px; }
  .event .action { font-weight: 600; color: #4ade80; }
  #empty { color: #5a5f68; font-size: 13px; }
  #media-pane {
    padding: 12px 14px; margin-bottom: 14px; background: #1b1f24;
    border-radius: 6px; box-sizing: border-box;
  }
  #media-pane h1 {
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em;
    color: #8a8f98; margin: 0 0 8px;
  }
  #media-status { display: flex; align-items: center; font-size: 13px; }
  .dot {
    width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; flex-shrink: 0;
  }
  .dot-offline { background: #5a5f68; }
  .dot-paused { background: #eab308; }
  .dot-playing { background: #4ade80; }
  #media-track { font-size: 12px; color: #8a8f98; margin-top: 6px; }
  #media-track .title { color: #e8e8e8; }
</style>
</head>
<body>
  <div id="video-pane"><img src="/stream" alt="Live overlay feed"></div>
  <div id="queue-pane">
    <div id="media-pane">
      <h1>Media Control</h1>
      <div id="media-status"><span class="dot dot-offline"></span><span id="media-text">Not connected</span></div>
      <div id="media-track"></div>
    </div>
    <h1>Gesture Queue</h1>
    <div id="events"><div id="empty">Waiting for gestures...</div></div>
  </div>
<script>
let lastLen = -1;
async function pollEvents() {
  try {
    const res = await fetch("/events");
    const events = await res.json();
    if (events.length !== lastLen) {
      const container = document.getElementById("events");
      if (events.length === 0) {
        container.innerHTML = '<div id="empty">Waiting for gestures...</div>';
      } else {
        container.innerHTML = events.slice().reverse().map(function (e) {
          const t = new Date(e.t * 1000).toLocaleTimeString();
          return '<div class="event"><span class="time">' + t + '</span>' +
                 '<span class="track">#' + e.track_id + '</span>' +
                 '<span class="action">' + e.action + '</span></div>';
        }).join("");
      }
      lastLen = events.length;
    }
  } catch (err) {
    // Server not ready yet, or a transient connection hiccup -- just retry.
  }
  setTimeout(pollEvents, 500);
}
function escapeHtml(s) {
  return s.replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; });
}
async function pollMedia() {
  try {
    const res = await fetch("/media");
    const m = await res.json();
    const dot = document.querySelector("#media-status .dot");
    const text = document.getElementById("media-text");
    const track = document.getElementById("media-track");
    if (!m.available) {
      dot.className = "dot dot-offline";
      text.textContent = m.backend === "none" ? "Media control disabled" : "No player found";
      track.textContent = "";
    } else {
      dot.className = m.playing ? "dot dot-playing" : "dot dot-paused";
      text.textContent = m.playing ? "Playing" : "Paused";
      track.innerHTML = m.title
        ? '<span class="title">' + escapeHtml(m.title) + "</span>" + (m.artist ? " -- " + escapeHtml(m.artist) : "")
        : "";
    }
  } catch (err) {
    // Server not ready yet, or a transient connection hiccup -- just retry.
  }
  setTimeout(pollMedia, 2000);
}
pollEvents();
pollMedia();
</script>
</body>
</html>
"""


class DashboardState:
    """Thread-safe state shared between the capture loop (one producer)
    and the dashboard's HTTP request handlers (one thread per open
    connection, via `ThreadingHTTPServer`)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_jpeg: bytes | None = None
        self._frame_version = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)
        self._media_status = MediaStatus(available=False, backend_name="none")

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Encode and store the latest overlay frame. Call this once per
        loop iteration, after all overlay drawing is done -- cheap enough
        (~1-2ms for a typical frame) to not meaningfully affect the
        capture loop's own framerate."""
        import cv2  # local import: keep cv2 out of module import for callers that don't need it (e.g. tests)

        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return
        with self._lock:
            self._frame_jpeg = encoded.tobytes()
            self._frame_version += 1

    def add_event(self, track_id: int, action_name: str) -> None:
        with self._lock:
            self._events.append({"t": time.time(), "track_id": track_id, "action": action_name})

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg

    def get_frame_version(self) -> int:
        with self._lock:
            return self._frame_version

    def get_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def update_media_status(self, status: MediaStatus) -> None:
        """Call once per loop iteration with `MediaController.get_status()`
        -- cheap (a lock-protected attribute read on the controller side,
        no subprocess call), so per-frame is fine even though the
        underlying player status only actually changes every couple of
        seconds (`control.backends.playerctl`'s own poll interval)."""
        with self._lock:
            self._media_status = status

    def get_media_status(self) -> MediaStatus:
        with self._lock:
            return self._media_status


def _make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass  # keep stdout clean -- app.py already prints its own status lines

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._serve_index()
            elif self.path == "/stream":
                self._serve_stream()
            elif self.path == "/events":
                self._serve_events()
            elif self.path == "/media":
                self._serve_media()
            else:
                self.send_error(404)

        def _serve_index(self) -> None:
            body = _INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_events(self) -> None:
            body = json.dumps(state.get_events()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_media(self) -> None:
            status = state.get_media_status()
            body = json.dumps(
                {
                    "available": status.available,
                    "backend": status.backend_name,
                    "playing": status.playing,
                    "title": status.title,
                    "artist": status.artist,
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}")
            self.end_headers()
            last_version = -1
            try:
                while True:
                    version = state.get_frame_version()
                    if version == last_version:
                        time.sleep(_STREAM_POLL_INTERVAL_S)
                        continue
                    frame = state.get_frame()
                    last_version = version
                    if frame is None:
                        time.sleep(_STREAM_POLL_INTERVAL_S)
                        continue
                    self.wfile.write(f"--{MJPEG_BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # the viewer closed the tab/connection -- not an error

    return Handler


class DashboardServer:
    """Runs the dashboard's HTTP server on a background daemon thread
    alongside the main capture loop. `port=0` lets the OS assign a free
    port -- read it back via `self.port` after construction."""

    def __init__(self, state: DashboardState, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(state))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
