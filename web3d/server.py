#!/usr/bin/env python3
"""Serve the VehicleArena browser-3D prototype and map API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import queue
import re
import sys
import threading
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


WEB_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = WEB_ROOT.parent
PACKAGE_ROOT = REPOSITORY_ROOT / "vehiclearena"
for import_root in (REPOSITORY_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from visualization.web3d_scene import build_web3d_scene


MAX_LIVE_FRAME_BYTES = 2 * 1024 * 1024
_SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


class LiveFrameHub:
    """Thread-safe latest-frame store and bounded WebSocket fan-out."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, dict] = {}
        self._subscribers: dict[str, set[queue.Queue]] = {}

    def publish(self, session_id: str, frame: dict) -> None:
        with self._lock:
            self._latest[session_id] = frame
            subscribers = list(self._subscribers.get(session_id, ()))
        for subscriber in subscribers:
            while True:
                try:
                    subscriber.put_nowait(frame)
                    break
                except queue.Full:
                    try:
                        subscriber.get_nowait()
                    except queue.Empty:
                        break

    def latest(self, session_id: str) -> dict | None:
        with self._lock:
            return self._latest.get(session_id)

    def subscribe(self, session_id: str) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=2)
        with self._lock:
            self._subscribers.setdefault(session_id, set()).add(subscriber)
            latest = self._latest.get(session_id)
        if latest is not None:
            subscriber.put_nowait(latest)
        return subscriber

    def unsubscribe(self, session_id: str, subscriber: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(session_id)
            if subscribers is None:
                return
            subscribers.discard(subscriber)
            if not subscribers:
                self._subscribers.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._latest)


LIVE_HUB = LiveFrameHub()


@lru_cache(maxsize=24)
def _scene_bytes(
    map_id: str,
    junction_id: str,
    radius_m: float,
    center_x: float | None,
    center_y: float | None,
    include_demo_actors: bool,
) -> bytes:
    return json.dumps(
        build_web3d_scene(
            map_id,
            junction_id=junction_id or None,
            radius_m=radius_m,
            center_world_xy=(
                [center_x, center_y]
                if center_x is not None and center_y is not None else None),
            include_demo_actors=include_demo_actors,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class VehicleArena3DHandler(BaseHTTPRequestHandler):
    """Minimal local-only HTTP server with one generated scene endpoint."""

    server_version = "VehicleArena3D/0.1"
    protocol_version = "HTTP/1.1"

    @staticmethod
    def _session_id(query: dict) -> str:
        session_id = query.get("session_id", ["live"])[0]
        if not _SESSION_PATTERN.fullmatch(session_id):
            raise ValueError(
                "session_id must contain only letters, numbers, _, ., : or -")
        return session_id

    def log_message(self, format_string: str, *args) -> None:
        if (self.path.startswith("/api/live/frame")
                and len(args) > 1 and str(args[1]) == "202"):
            return
        sys.stderr.write(
            f"[web3d] {self.address_string()} "
            f"{format_string % args}\n")

    def _write(
        self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path) -> None:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            self._write(
                b"not found\n", "text/plain; charset=utf-8",
                HTTPStatus.NOT_FOUND,
            )
            return
        allowed_roots = [WEB_ROOT.resolve(), (WEB_ROOT / "node_modules").resolve()]
        if not any(
                resolved == root or root in resolved.parents
                for root in allowed_roots):
            self._write(
                b"forbidden\n", "text/plain; charset=utf-8",
                HTTPStatus.FORBIDDEN,
            )
            return
        content_type = mimetypes.guess_type(resolved.name)[0] or \
            "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
                "application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self._write(resolved.read_bytes(), content_type)

    @staticmethod
    def _websocket_packet(payload: bytes, opcode: int = 0x1) -> bytes:
        length = len(payload)
        header = bytes([0x80 | opcode])
        if length < 126:
            return header + bytes([length]) + payload
        if length <= 0xFFFF:
            return header + bytes([126]) + length.to_bytes(2, "big") + payload
        return header + bytes([127]) + length.to_bytes(8, "big") + payload

    def _serve_websocket(self, session_id: str) -> None:
        key = self.headers.get("Sec-WebSocket-Key", "")
        if (self.headers.get("Upgrade", "").lower() != "websocket"
                or not key):
            self._write(
                b"websocket upgrade required\n",
                "text/plain; charset=utf-8",
                HTTPStatus.UPGRADE_REQUIRED,
            )
            return
        accept = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode(
                "ascii")
        ).digest()).decode("ascii")
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        # WebSocket frames are not HTTP requests. Prevent BaseHTTPRequestHandler
        # from parsing the browser's masked close frame as a second request.
        self.close_connection = True

        subscriber = LIVE_HUB.subscribe(session_id)
        try:
            while True:
                try:
                    frame = subscriber.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(self._websocket_packet(b"", opcode=0x9))
                    self.wfile.flush()
                    continue
                payload = json.dumps(
                    frame, ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")
                self.wfile.write(self._websocket_packet(payload))
                self.wfile.flush()
                if frame.get("stream_status") == "ended":
                    self.wfile.write(self._websocket_packet(b"", opcode=0x8))
                    self.wfile.flush()
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            LIVE_HUB.unsubscribe(session_id, subscriber)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path != "/api/live/frame":
            self._write(
                b"not found\n", "text/plain; charset=utf-8",
                HTTPStatus.NOT_FOUND,
            )
            return
        try:
            session_id = self._session_id(parse_qs(parsed.query))
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_LIVE_FRAME_BYTES:
                raise ValueError("live frame body size is invalid")
            frame = json.loads(self.rfile.read(length))
            if not isinstance(frame, dict):
                raise ValueError("live frame must be a JSON object")
            if frame.get("schema") != "vehiclearena-web3d-frame-v0.1":
                raise ValueError("unsupported live frame schema")
            LIVE_HUB.publish(session_id, frame)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._write(
                json.dumps({"error": str(error)}).encode("utf-8"),
                "application/json; charset=utf-8",
                HTTPStatus.BAD_REQUEST,
            )
            return
        self._write(
            b'{"accepted":true}', "application/json; charset=utf-8",
            HTTPStatus.ACCEPTED,
        )

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write(
                json.dumps({
                    "ok": True,
                    "live_sessions": LIVE_HUB.session_count(),
                }).encode("utf-8"),
                "application/json; charset=utf-8")
            return
        if parsed.path in {"/api/live/frame", "/api/live/ws"}:
            try:
                session_id = self._session_id(parse_qs(parsed.query))
            except ValueError as error:
                self._write(
                    json.dumps({"error": str(error)}).encode("utf-8"),
                    "application/json; charset=utf-8",
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if parsed.path == "/api/live/ws":
                self._serve_websocket(session_id)
                return
            frame = LIVE_HUB.latest(session_id)
            if frame is None:
                self._write(
                    b'{"error":"unknown live session"}',
                    "application/json; charset=utf-8",
                    HTTPStatus.NOT_FOUND,
                )
                return
            self._write(
                json.dumps(frame, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        if parsed.path == "/api/scene":
            query = parse_qs(parsed.query)
            map_id = query.get("map_id", ["beijing_tiananmen"])[0]
            junction_id = query.get("junction_id", ["n31194143"])[0]
            try:
                radius_m = float(query.get("radius_m", ["145"])[0])
                center_values = (
                    query.get("center_x", [None])[0],
                    query.get("center_y", [None])[0],
                )
                if (center_values[0] is None) != (center_values[1] is None):
                    raise ValueError(
                        "center_x and center_y must be provided together")
                center_x = (
                    float(center_values[0])
                    if center_values[0] is not None else None)
                center_y = (
                    float(center_values[1])
                    if center_values[1] is not None else None)
                include_demo = query.get("demo", ["1"])[0] not in {
                    "0", "false", "no"}
                body = _scene_bytes(
                    map_id, junction_id, radius_m,
                    center_x, center_y, include_demo)
            except (OSError, TypeError, ValueError) as error:
                self._write(
                    json.dumps({"error": str(error)}).encode("utf-8"),
                    "application/json; charset=utf-8",
                    HTTPStatus.BAD_REQUEST,
                )
                return
            self._write(body, "application/json; charset=utf-8")
            return
        if parsed.path.startswith("/vendor/addons/"):
            relative = parsed.path.removeprefix("/vendor/addons/")
            self._serve_file(
                WEB_ROOT / "node_modules/three/examples/jsm" / relative)
            return
        if parsed.path.startswith("/vendor/"):
            relative = parsed.path.removeprefix("/vendor/")
            self._serve_file(WEB_ROOT / "node_modules/three/build" / relative)
            return
        static_files = {
            "/": "index.html",
            "/index.html": "index.html",
            "/app.js": "app.js",
            "/lidar_bev.js": "lidar_bev.js",
            "/styles.css": "styles.css",
        }
        relative = static_files.get(parsed.path)
        if relative is None:
            self._write(
                b"not found\n", "text/plain; charset=utf-8",
                HTTPStatus.NOT_FOUND,
            )
            return
        self._serve_file(WEB_ROOT / relative)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port), VehicleArena3DHandler)
    print(f"VehicleArena Web3D: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
