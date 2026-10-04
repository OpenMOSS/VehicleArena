"""Headless capture of the existing Web3D cockpit for LLM camera input."""

from __future__ import annotations

import base64
import math
import threading
from http.server import ThreadingHTTPServer
from typing import Any, Dict

from visualization.agent_visual_renderer import RenderedAgentImage
from visualization.web3d_live import Web3DFrameEncoder
from module.lidar import resolve_lidar_spec


class Web3DCameraRenderer:
    """Render exact synchronized SUMO state through the browser cockpit.

    One browser is reused for the episode and each evaluated vehicle owns a
    page plus a focus-specific frame encoder. The browser never advances the
    physical world; it only renders a supplied frozen boundary.
    """

    def __init__(self, *, width: int = 1024, height: int = 960):
        self.width = int(width)
        self.height = int(height)
        self._lock = threading.RLock()
        self._server = None
        self._server_thread = None
        self._playwright = None
        self._browser = None
        self._pages: Dict[str, Any] = {}
        self._encoders: Dict[str, Web3DFrameEncoder] = {}
        self._last_capture: Dict[str, dict] = {}

    def _start(self) -> None:
        if self._browser is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Web3D CameraVisual requires Playwright. Install project "
                "requirements and run `playwright install chromium`."
            ) from exc
        from web3d.server import VehicleArena3DHandler

        self._server = ThreadingHTTPServer(
            ("127.0.0.1", 0), VehicleArena3DHandler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="vehiclearena-camera-web3d",
            daemon=True,
        )
        self._server_thread.start()
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=["--use-gl=swiftshader", "--disable-dev-shm-usage"],
        )

    def _page(self, vehicle_id: str):
        page = self._pages.get(vehicle_id)
        if page is not None:
            return page
        context = self._browser.new_context(
            viewport={"width": self.width, "height": self.height},
            device_scale_factor=1,
        )
        page = context.new_page()
        port = self._server.server_address[1]
        page.goto(
            f"http://127.0.0.1:{port}/?view=cockpit&capture=1",
            wait_until="networkidle",
        )
        page.wait_for_function(
            "typeof window.vehicleArenaCaptureFrame === 'function'")
        self._pages[vehicle_id] = page
        return page

    def render(
        self, engine: Any, vehicle_id: str,
        sim_time_s: float, tick_index: int,
    ) -> RenderedAgentImage:
        with self._lock:
            self._start()
            encoder = self._encoders.setdefault(
                vehicle_id,
                Web3DFrameEncoder(
                    focus_entity_id=vehicle_id, radius_m=220.0),
            )
            frame = encoder.encode(
                engine, sim_time_s, tick_index, trigger_events=())
            page = self._page(vehicle_id)
            result = page.evaluate(
                "frame => window.vehicleArenaCaptureFrame(frame)", frame)
            self._validate_capture(result, vehicle_id, sim_time_s, frame["sequence"])
            if result.get("observation_layout") != (
                    "front_with_upper_side_glance_strip"):
                raise RuntimeError(
                    "Web3D camera did not render the driver observation layout")
            if result.get("render_mode") != "on_demand":
                raise RuntimeError(
                    "Web3D capture page left continuous rendering enabled")
            png = page.screenshot(type="png", animations="disabled")
            self._last_capture[vehicle_id] = result
            return RenderedAgentImage(
                kind="camera_visual",
                sim_time_s=float(sim_time_s),
                width=self.width,
                height=self.height,
                png_bytes=png,
                view="web3d_driver_observation",
            )

    @staticmethod
    def _validate_capture(result, vehicle_id, sim_time_s, sequence):
        if (not isinstance(result, dict)
                or not math.isclose(float(result.get("sim_time_s", -1)),
                                    float(sim_time_s), rel_tol=0, abs_tol=1e-6)
                or result.get("focus_entity_id") != vehicle_id
                or result.get("frame_sequence") != sequence):
            raise RuntimeError("Web3D capture did not acknowledge the frozen frame/focus")

    def render_observations(
        self, engine: Any, vehicle_id: str, sim_time_s: float, tick_index: int,
        sensor_overrides=None,
    ) -> tuple[RenderedAgentImage, RenderedAgentImage]:
        """Capture cockpit and optional geometry view from one frozen frame.

        The engine calls this only for lidar-equipped vehicles. Reuse the same
        page, encoder, scene and browser; do not encode or advance a second frame.
        """
        sensor = resolve_lidar_spec("lidar", sensor_overrides).as_dict()
        with self._lock:
            camera = self.render(engine, vehicle_id, sim_time_s, tick_index)
            page = self._pages[vehicle_id]
            result = page.evaluate(
                "request => window.vehicleArenaCaptureLidarBEV(request)",
                {"sim_time_s": float(sim_time_s), "focus_entity_id": vehicle_id,
                 "sensor": sensor})
            self._validate_capture(result, vehicle_id, sim_time_s,
                                   self._last_capture[vehicle_id]["frame_sequence"])
            if (result.get("view") != "web3d_ego_oblique"
                    or result.get("render_mode") != "on_demand"
                    or (result.get("width"), result.get("height")) != (1024, 1024)):
                raise RuntimeError("Web3D LiDAR BEV returned an invalid view")
            prefix = "data:image/png;base64,"
            encoded = result.get("png_data_url", "")
            if not isinstance(encoded, str) or not encoded.startswith(prefix):
                raise RuntimeError("Web3D LiDAR BEV did not return a PNG")
            png = base64.b64decode(encoded[len(prefix):], validate=True)
            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("Web3D LiDAR BEV returned invalid PNG bytes")
            return camera, RenderedAgentImage(
                kind="lidar_bev", sim_time_s=float(sim_time_s),
                width=1024, height=1024, png_bytes=png,
                view="web3d_ego_oblique")

    def close(self) -> None:
        with self._lock:
            for page in list(self._pages.values()):
                try:
                    page.context.close()
                except Exception:
                    pass
            self._pages.clear()
            self._last_capture.clear()
            self._encoders.clear()
            if self._browser is not None:
                try:
                    self._browser.close()
                except Exception:
                    pass
                self._browser = None
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
                self._server = None
            if self._server_thread is not None:
                self._server_thread.join(timeout=2.0)
                self._server_thread = None


__all__ = ["Web3DCameraRenderer"]
