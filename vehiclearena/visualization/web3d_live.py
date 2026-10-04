"""Read-only live bridge from the authoritative SUMO world to Web3D.

The bridge never advances actors and never writes to ``TrafficCoordinator``.
It samples state only after SUMO has completed a public physics step, then
publishes the newest frame on a bounded background queue.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from visualization.signal_layout import signal_stop_lines

LIVE_FRAME_SCHEMA = "vehiclearena-web3d-frame-v0.1"


def _round(value: Any, digits: int = 4) -> float:
    return round(float(value), digits)


class Web3DFrameEncoder:
    """Convert synchronized public world state into compact dynamic frames."""

    def __init__(
        self,
        *,
        focus_entity_id: str = "",
        junction_id: Optional[str] = None,
        radius_m: float = 220.0,
    ) -> None:
        if not 50.0 <= float(radius_m) <= 500.0:
            raise ValueError("radius_m must be between 50 and 500")
        self.focus_entity_id = str(focus_entity_id)
        self.junction_id = junction_id
        self.radius_m = float(radius_m)
        self._sequence = 0
        self._scene_spec: Optional[dict] = None
        self._signal_connector_ids: tuple[str, ...] = ()
        self._scene_revision = 0

    @property
    def frame_count(self) -> int:
        return self._sequence

    def _choose_focus(self, engine: Any) -> str:
        manager = engine.traffic_mgr
        if self.focus_entity_id in manager.vehicles:
            return self.focus_entity_id
        llm_ids = sorted(
            vehicle_id for vehicle_id, vehicle in manager.vehicles.items()
            if vehicle.is_llm)
        if llm_ids:
            return llm_ids[0]
        vehicle_ids = sorted(manager.vehicles)
        if vehicle_ids:
            return vehicle_ids[0]
        return ""

    def _prepare_scene(self, engine: Any, focus_id: str) -> None:
        manager = engine.traffic_mgr
        data = manager._lane_geometry.data
        focus = manager.vehicles.get(focus_id)
        focus_xy = (
            (float(focus.pose_x_m), float(focus.pose_y_m))
            if focus is not None else (0.0, 0.0))
        requested_junction = None
        if self.junction_id:
            requested_junction = next((
                item for item in data["intersections"]
                if str(item["id"]) == self.junction_id
            ), None)
            if requested_junction is None:
                raise ValueError(
                    f"unknown Web3D junction {self.junction_id!r}")
        if self._scene_spec is not None:
            if requested_junction is not None:
                return
            old_center = self._scene_spec["center_world_xy"]
            if math.hypot(
                    focus_xy[0] - old_center[0],
                    focus_xy[1] - old_center[1]) <= self.radius_m * 0.60:
                return

        # Quantized local tiles let different runs reuse the server's static
        # scene cache while keeping a long route inside rendered geometry.
        if requested_junction is not None:
            center = [float(value) for value in requested_junction["center_xy"]]
            nearest_junction = requested_junction
        else:
            tile_span = self.radius_m * 0.75
            center = [
                round(focus_xy[0] / tile_span) * tile_span,
                round(focus_xy[1] / tile_span) * tile_span,
            ]
            nearest_junction = min(
                data["intersections"],
                key=lambda item: math.hypot(
                    float(item["center_xy"][0]) - center[0],
                    float(item["center_xy"][1]) - center[1]),
            )
        junction_id = str(nearest_junction["id"])
        nearby_junction_ids = {
            str(item["id"])
            for item in data["intersections"]
            if math.hypot(
                float(item["center_xy"][0]) - center[0],
                float(item["center_xy"][1]) - center[1],
            ) <= self.radius_m
        }
        lanes_by_id = {
            str(lane["id"]): lane for lane in data["lanes"]}
        connectors_by_from: Dict[str, list] = {}
        for connector in data["connectors"]:
            connectors_by_from.setdefault(
                str(connector.get("from_lane", "")), []).append(connector)
        signal_connectors = set()
        for stop_line in signal_stop_lines(data):
            if str(stop_line.get("node_id", "")) not in nearby_junction_ids:
                continue
            lane = lanes_by_id.get(str(stop_line.get("lane_id", "")))
            if lane is None:
                continue
            candidates = connectors_by_from.get(str(lane["id"]), [])
            # Publish every movement, not just the preferred through lamp.
            # encode() skips connectors without a lane-level signal plan.
            signal_connectors.update(str(item["id"]) for item in candidates)
        self._signal_connector_ids = tuple(sorted(signal_connectors))
        self._scene_revision += 1
        self._scene_spec = {
            "map_id": engine.scenario.road_network_id,
            "junction_id": junction_id,
            "center_world_xy": center,
            "radius_m": self.radius_m,
            "revision": self._scene_revision,
        }

    @staticmethod
    def _vehicle_actor(vehicle_id: str, vehicle: Any, focus_id: str) -> dict:
        # SUMO getPosition() is the front-bumper midpoint. Web3D meshes
        # (and their cockpit/BEV cameras) use a body-centred actor origin.
        # Convert once at this read-only boundary, for ego and peers alike;
        # never overwrite the authoritative physics pose.
        yaw = float(vehicle.yaw_rad)
        half_length = float(vehicle.length_m) / 2.0
        center_x = float(vehicle.pose_x_m) - half_length * math.cos(yaw)
        center_y = float(vehicle.pose_y_m) - half_length * math.sin(yaw)
        signals = (
            vehicle.signal_state.as_dict()
            if hasattr(vehicle.signal_state, "as_dict")
            else dict(vehicle.signal_state or {}))
        return {
            "id": vehicle_id,
            "kind": "vehicle",
            "control": str(vehicle.control_authority),
            "is_focus": vehicle_id == focus_id,
            "position_world_xy": [
                _round(center_x), _round(center_y)],
            "position_reference": "body_center",
            "yaw_rad": _round(vehicle.yaw_rad, 6),
            "dimensions_m": [
                _round(vehicle.width_m, 3), 1.55,
                _round(vehicle.length_m, 3)],
            "speed_mps": _round(vehicle.current_speed_kmh / 3.6),
            "acceleration_mps2": _round(vehicle.acceleration_mps2),
            "z_level": int(vehicle.z_level),
            # The source geometry is XY-only. A topology layer is neither
            # metres nor a ramp profile; never turn it into a visual offset.
            "elevation_m": 0.0,
            "signals": signals,
            "crashed": bool(vehicle.is_crashed),
        }

    @staticmethod
    def _pedestrian_actor(manager: Any, pedestrian_id: str, pedestrian: Any) -> Optional[dict]:
        pose = manager._lane_geometry.pedestrian_pose(pedestrian)
        if pose is None:
            return None
        return {
            "id": pedestrian_id,
            "kind": "pedestrian",
            "control": str(pedestrian.control_authority),
            "is_focus": False,
            "position_world_xy": [_round(pose[0]), _round(pose[1])],
            "yaw_rad": (
                _round(pedestrian.physical_yaw_rad, 6)
                if getattr(pedestrian, "physical_yaw_rad", None) is not None
                else None),
            "dimensions_m": [
                _round(pedestrian.collision_radius_m * 2.0, 3),
                1.72,
                _round(pedestrian.collision_radius_m * 2.0, 3),
            ],
            "speed_mps": _round(pedestrian.speed),
            "crashed": bool(pedestrian.is_crashed),
        }

    def encode(
        self,
        engine: Any,
        physics_time: float,
        tick_index: int,
        trigger_events: Any = (),
    ) -> dict:
        manager = engine.traffic_mgr
        focus_id = self._choose_focus(engine)
        self._prepare_scene(engine, focus_id)
        actors = [
            self._vehicle_actor(vehicle_id, vehicle, focus_id)
            for vehicle_id, vehicle in sorted(manager.vehicles.items())
            if vehicle.present_in_physics_world and not vehicle.arrived
        ]
        for pedestrian_id, pedestrian in sorted(manager.pedestrians.items()):
            if (not pedestrian.is_spawned or pedestrian.has_arrived):
                continue
            actor = self._pedestrian_actor(
                manager, pedestrian_id, pedestrian)
            if actor is not None:
                actors.append(actor)

        signals: Dict[str, str] = {}
        for connector_id in self._signal_connector_ids:
            signal = manager._lane_geometry.signal_state(
                connector_id, float(physics_time))
            if signal is not None:
                signals[connector_id] = str(signal.signal)

        frame = {
            "schema": LIVE_FRAME_SCHEMA,
            "sequence": self._sequence,
            "sim_time_s": _round(physics_time, 6),
            "tick_index": int(tick_index),
            "physics_step_s": _round(engine.scenario.physics_step_s, 6),
            "physics_source": "sumo",
            "stream_status": "running",
            "focus_entity_id": focus_id,
            "scene": dict(self._scene_spec or {}),
            "environment": {
                "weather": str(manager._current_weather),
                "wind_speed_mps": _round(manager._current_wind_speed_mps),
                "daylight_level": int(manager._daylight_level),
                "is_night": bool(manager._is_night),
            },
            "signals": signals,
            "actors": actors,
            "events": [
                {
                    "type": str(event.type),
                    "entity_id": str(event.vehicle_id),
                }
                for event in trigger_events
            ],
        }
        self._sequence += 1
        return frame


class Web3DFramePublisher:
    """Publish latest-only Web3D frames without blocking SUMO on I/O."""

    def __init__(
        self,
        stream_url: str = "http://127.0.0.1:8765/api/live/frame",
        *,
        session_id: str = "live",
        focus_entity_id: str = "",
        junction_id: Optional[str] = None,
        radius_m: float = 220.0,
        realtime: bool = False,
        request_timeout_s: float = 1.0,
    ) -> None:
        if not str(session_id).strip():
            raise ValueError("session_id must not be empty")
        self.session_id = str(session_id)
        self.stream_url = self._with_session(stream_url, self.session_id)
        self.encoder = Web3DFrameEncoder(
            focus_entity_id=focus_entity_id,
            junction_id=junction_id,
            radius_m=radius_m,
        )
        self.realtime = bool(realtime)
        self.request_timeout_s = max(0.1, float(request_timeout_s))
        self._queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=3)
        self._worker = threading.Thread(
            target=self._send_loop,
            name=f"web3d-publisher-{self.session_id}",
            daemon=True,
        )
        self._worker.start()
        self._closed = False
        self._last_frame: Optional[dict] = None
        self._first_sim_time: Optional[float] = None
        self._first_wall_time: Optional[float] = None
        self.published_count = 0
        self.dropped_count = 0
        self.failed_count = 0
        self.last_error = ""

    @staticmethod
    def _with_session(url: str, session_id: str) -> str:
        parsed = urllib.parse.urlsplit(str(url))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("stream_url must be an absolute HTTP(S) URL")
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query = [(key, value) for key, value in query if key != "session_id"]
        query.append(("session_id", session_id))
        return urllib.parse.urlunsplit((
            parsed.scheme, parsed.netloc, parsed.path,
            urllib.parse.urlencode(query), parsed.fragment,
        ))

    def _pace(self, sim_time_s: float) -> None:
        if not self.realtime:
            return
        now = time.monotonic()
        if self._first_sim_time is None:
            self._first_sim_time = float(sim_time_s)
            self._first_wall_time = now
            return
        target = (
            float(self._first_wall_time)
            + float(sim_time_s) - float(self._first_sim_time))
        delay = target - now
        if delay > 0.0:
            time.sleep(delay)

    def _enqueue(self, payload: bytes) -> None:
        while True:
            try:
                self._queue.put_nowait(payload)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.dropped_count += 1
                except queue.Empty:
                    return

    def on_world_step(
        self, engine: Any, physics_time: float, tick_index: int,
        trigger_events: Any,
    ) -> None:
        if self._closed:
            return
        frame = self.encoder.encode(
            engine, physics_time, tick_index, trigger_events)
        self._pace(physics_time)
        self._last_frame = frame
        self._enqueue(json.dumps(
            frame, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))

    def _post(self, body: bytes) -> None:
        request = urllib.request.Request(
            self.stream_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(
                request, timeout=self.request_timeout_s) as response:
            if response.status != 202:
                raise RuntimeError(
                    f"Web3D stream returned HTTP {response.status}")

    def _send_loop(self) -> None:
        while True:
            payload = self._queue.get()
            if payload is None:
                return
            try:
                self._post(payload)
                self.published_count += 1
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                self.failed_count += 1
                self.last_error = f"{type(error).__name__}: {error}"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._last_frame is not None:
            ended = dict(self._last_frame)
            ended["stream_status"] = "ended"
            self._enqueue(json.dumps(
                ended, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8"))
        self._queue.put(None)
        self._worker.join(timeout=max(2.0, self.request_timeout_s + 0.5))
