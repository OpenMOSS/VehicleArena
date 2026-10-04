"""SUMO-backed VehicleArena traffic world.

SUMO owns all background vehicle and pedestrian behaviour. LLM actors own
their explicit commands; SUMO executes those commands and all shared physics
on the same 0.1-second clock.
"""

from __future__ import annotations

import importlib
import math
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from simulation.sumo_map import (
    SumoMapBundle, SumoMapConverter, pedestrian_node_pair_key,
    route_edges_for_vehicle,
)
from simulation.traffic_manager import (
    CollisionEvent, TrafficCoordinator, TriggerEvent,
)
from simulation.native_npc_behavior import sample_driver, validate_behavior


@dataclass(frozen=True)
class SumoPhysicsConfig:
    cache_root: str = ""
    step_length_s: float = 0.1
    lateral_resolution_m: float = 0.2
    collision_action: str = "warn"
    collision_stop_time_s: float = 1_000_000_000.0
    time_to_teleport_s: float = -1.0
    suppress_warnings: bool = True
    gui: bool = False
    auto_start_xvfb: bool = True
    gui_width: int = 1280
    gui_height: int = 1024
    gui_settings_file: str = ""
    gui_schema: str = ""
    npc_behavior: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SumoPhysicsConfig":
        unknown = set(raw) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(
                f"unknown SUMO physics settings: {sorted(unknown)}")
        value = cls(**raw)
        validate_behavior(value.npc_behavior)
        if abs(float(value.step_length_s) - 0.1) > 1e-9:
            raise ValueError(
                "SUMO physics engine currently requires step_length_s=0.1")
        if float(value.lateral_resolution_m) <= 0:
            raise ValueError("lateral_resolution_m must be positive")
        if value.collision_action not in {"warn", "none"}:
            raise ValueError("collision_action must be warn or none")
        if float(value.collision_stop_time_s) < 0:
            raise ValueError("collision_stop_time_s must be non-negative")
        if int(value.gui_width) <= 0 or int(value.gui_height) <= 0:
            raise ValueError("SUMO gui_width and gui_height must be positive")
        if value.gui_settings_file and not Path(
                value.gui_settings_file).expanduser().is_file():
            raise ValueError(
                "SUMO gui_settings_file does not exist: "
                f"{value.gui_settings_file}")
        return value


class SumoTrafficManager(TrafficCoordinator):
    """TrafficCoordinator API whose physical vehicle advancement comes from SUMO."""

    engine_name = "sumo"
    _active_instance: Optional["SumoTrafficManager"] = None
    _LLM_COLOR = (42, 205, 137, 255)
    _BACKGROUND_COLOR = (72, 175, 230, 255)

    def __init__(
        self,
        road_network: Any,
        *,
        lane_geometry_runtime: Any = None,
        sumo_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            road_network,
            lane_geometry_runtime=lane_geometry_runtime,
        )
        self.sumo_config = SumoPhysicsConfig.from_dict(
            dict(sumo_config or {}))
        self.npc_behavior_assignments: Dict[str, dict] = {}
        if SumoTrafficManager._active_instance is not None:
            raise RuntimeError(
                "libsumo supports one in-process episode at a time; close the "
                "previous VehicleArena SUMO physics engine first")
        self._using_traci = bool(self.sumo_config.gui)
        self._traci_label = (
            f"vehiclearena-{os.getpid()}-{id(self)}"
            if self._using_traci else "")
        self._sumo_process: Optional[subprocess.Popen] = None
        self._xvfb_process: Optional[subprocess.Popen] = None
        self._gui_display = ""
        module_name = "traci" if self._using_traci else "libsumo"
        try:
            sumo_module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                f"{module_name} is unavailable; install the SUMO runtime "
                "and its Python tools") from exc
        lane_level_path = road_network.lane_level_path
        self.sumo_map: SumoMapBundle = SumoMapConverter().convert(
            lane_level_path,
            cache_root=self.sumo_config.cache_root or None,
        )
        # Include every internal yield segment, not only the first via lane.
        chains = SumoMapConverter._connector_chains(
            ET.parse(self.sumo_map.net_file).getroot(),
            {"connector_via_lane": self.sumo_map.connector_via_lane})
        self._sumo_connector_by_internal_lane: Dict[str, str] = {}
        for connector_id, (_, lanes, _) in chains.items():
            for lane_id in lanes:
                previous = self._sumo_connector_by_internal_lane.setdefault(
                    lane_id, connector_id)
                if previous != connector_id:
                    raise RuntimeError(
                        f"ambiguous SUMO internal lane: {lane_id}")
        self._sumo_started_ids: set[str] = set()
        self._sumo_pedestrian_proxy: Dict[str, str] = {}
        self._sumo_pedestrian_owner: Dict[str, str] = {}
        self._sumo_pedestrian_epoch: Dict[str, int] = {}
        self._sumo_pedestrian_leg: Dict[str, Dict[str, Any]] = {}
        self._sumo_pedestrian_types: set[str] = set()
        self._sumo_last_edge: Dict[str, str] = {}
        self._sumo_installed_routes: Dict[str, List[str]] = {}
        self._sumo_last_distance_m: Dict[str, float] = {}
        self._sumo_last_position_m: Dict[str, Tuple[float, float]] = {}
        self._sumo_lane_changes: Dict[
            str, Tuple[int, str, float, float]
        ] = {}
        self._sumo_lane_change_observations: Dict[
            str, Tuple[int, str, float, float]
        ] = {}
        self._sumo_collision_pairs: set[frozenset[str]] = set()
        self._sumo_lane_constraint_state: Dict[
            str, Tuple[bool, float]
        ] = {}
        self._sumo_lane_defaults: Dict[str, Tuple[float, Tuple[str, ...]]] = {}
        self._sumo_lanes_by_segment: Dict[str, List[Tuple[int, str]]] = {}
        for lane_id, lane in self._lane_geometry._lane_by_id.items():
            edge = self.sumo_map.edge_by_lane.get(lane_id)
            lane_index = self.sumo_map.lane_index_by_lane.get(lane_id)
            if edge is None or lane_index is None:
                continue
            sumo_lane_id = f"{edge}_{lane_index}"
            segment_id = str(lane["segment_id"])
            self._sumo_lanes_by_segment.setdefault(segment_id, []).append(
                (int(lane_index), sumo_lane_id))
            self._sumo_lane_defaults[sumo_lane_id] = (
                float(lane.get("speed_limit_kmh", 50.0)) / 3.6,
                ("passenger", "bus", "truck", "delivery", "emergency"),
            )
        self._sumo_time_s = 0.0
        self._initial_state_bootstrapped = False
        self._closed = False
        command = [
            "sumo-gui" if self._using_traci else "sumo",
            "--net-file", self.sumo_map.net_file,
            "--step-length", f"{self.sumo_config.step_length_s:g}",
            "--begin", "0",
            "--collision.action", self.sumo_config.collision_action,
            # Report bumper contact, not a gap smaller than the vehicle minGap.
            "--collision.mingap-factor", "0",
            "--collision.stoptime",
            f"{self.sumo_config.collision_stop_time_s:g}",
            "--intermodal-collision.action",
            self.sumo_config.collision_action,
            "--intermodal-collision.stoptime",
            f"{self.sumo_config.collision_stop_time_s:g}",
            "--collision.check-junctions", "true",
            "--time-to-teleport", f"{self.sumo_config.time_to_teleport_s:g}",
            "--lateral-resolution",
            f"{self.sumo_config.lateral_resolution_m:g}",
            "--no-step-log", "true",
            "--duration-log.disable", "true",
        ]
        if self.sumo_config.npc_behavior:
            command.extend(["--seed", str(self.sumo_config.npc_behavior["seed"])])
        if self._using_traci:
            command.extend([
                "--start",
                "--delay", "0",
                "--window-size",
                f"{int(self.sumo_config.gui_width)},{int(self.sumo_config.gui_height)}",
            ])
            if self.sumo_config.gui_settings_file:
                command.extend([
                    "--gui-settings-file",
                    str(Path(self.sumo_config.gui_settings_file).expanduser(
                    ).resolve()),
                ])
        if self.sumo_config.suppress_warnings:
            command.extend(["--no-warnings", "true"])
        try:
            if self._using_traci:
                self._gui_display = self._prepare_gui_display()
                previous_display = os.environ.get("DISPLAY")
                os.environ["DISPLAY"] = self._gui_display
                try:
                    sumo_module.start(
                        command,
                        label=self._traci_label,
                        doSwitch=False,
                        stdout=subprocess.DEVNULL,
                    )
                finally:
                    if previous_display is None:
                        os.environ.pop("DISPLAY", None)
                    else:
                        os.environ["DISPLAY"] = previous_display
                self._sumo = sumo_module.getConnection(self._traci_label)
                self._sumo_process = getattr(
                    self._sumo, "_process", None)
                if self.sumo_config.gui_schema:
                    views = list(self._sumo.gui.getIDList())
                    if not views:
                        raise RuntimeError(
                            "sumo-gui did not expose a render view")
                    self._sumo.gui.setSchema(
                        views[0], self.sumo_config.gui_schema)
            else:
                self._sumo = sumo_module
                self._sumo.start(command)
            self.sumo_version = str(self._sumo.getVersion()[1])
        except Exception:
            self._closed = True
            self._stop_gui_processes()
            raise
        SumoTrafficManager._active_instance = self

    def _prepare_gui_display(self) -> str:
        """Return a display for sumo-gui, starting Xvfb when headless."""
        configured = os.environ.get("DISPLAY", "").strip()
        if configured:
            return configured
        if not self.sumo_config.auto_start_xvfb:
            raise RuntimeError(
                "SUMO native rendering requires DISPLAY or "
                "sumo_config.auto_start_xvfb=true")
        xvfb = shutil.which("Xvfb")
        if not xvfb:
            raise RuntimeError(
                "SUMO native rendering is headless but Xvfb is unavailable; "
                "install xvfb or provide DISPLAY")
        for display_number in range(90, 200):
            display = f":{display_number}"
            socket_path = Path(f"/tmp/.X11-unix/X{display_number}")
            lock_path = Path(f"/tmp/.X{display_number}-lock")
            if socket_path.exists() or lock_path.exists():
                continue
            process = subprocess.Popen(
                [
                    xvfb, display, "-screen", "0",
                    f"{max(640, int(self.sumo_config.gui_width))}x"
                    f"{max(480, int(self.sumo_config.gui_height))}x24",
                    "-nolisten", "tcp",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(50):
                if process.poll() is not None:
                    break
                if socket_path.exists():
                    self._xvfb_process = process
                    return display
                time.sleep(0.02)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            continue
        raise RuntimeError("could not allocate a virtual X display for sumo-gui")

    def _stop_gui_processes(self) -> None:
        process = self._sumo_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._sumo_process = None
        xvfb = self._xvfb_process
        if xvfb is not None and xvfb.poll() is None:
            xvfb.terminate()
            try:
                xvfb.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                xvfb.kill()
                xvfb.wait(timeout=2.0)
        self._xvfb_process = None

    def close(self) -> None:
        if self._closed:
            return
        try:
            if hasattr(self, "_sumo"):
                if self._using_traci:
                    self._sumo.close(wait=False)
                else:
                    self._sumo.close()
        finally:
            self._closed = True
            self._stop_gui_processes()
            if SumoTrafficManager._active_instance is self:
                SumoTrafficManager._active_instance = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def physics_metadata(self) -> Dict[str, Any]:
        closed_lanes = sum(
            closed for closed, _ in self._sumo_lane_constraint_state.values())
        speed_restricted_lanes = sum(
            speed < self._sumo_lane_defaults[lane_id][0] - 1e-9
            for lane_id, (_, speed) in
            self._sumo_lane_constraint_state.items()
        )
        return {
            "name": self.engine_name,
            "sumo_version": self.sumo_version,
            "map_manifest": self.sumo_map.manifest_file,
            "step_length_s": self.sumo_config.step_length_s,
            "vehicle_decision_authority": "vehiclearena_driver",
            "vehicle_dynamics_authority": "sumo",
            "vehicle_pose_authority": "sumo",
            "vehicle_route_execution_authority": "sumo",
            "vehicle_route_decision_authority": (
                "llm_per_junction_for_llm_sumo_for_background"),
            "vehicle_collision_authority": "sumo",
            "rendering_authority": (
                "sumo-gui" if self._using_traci else "vehiclearena"),
            "vehicle_pedestrian_collision_authority": "sumo",
            "pedestrian_physics_authority": "sumo",
            "sumo_pedestrians": sum(
                pedestrian.physical_pose_authority == "sumo"
                for pedestrian in self.pedestrians.values()),
            "native_sumo_crosswalks": len(
                self.sumo_map.crosswalk_edge_by_id),
            "closed_lanes": closed_lanes,
            "speed_restricted_lanes": speed_restricted_lanes,
        }

    @property
    def native_rendering_enabled(self) -> bool:
        """Whether this episode is connected to a live sumo-gui view."""
        return self._using_traci and not self._closed

    def request_native_screenshot(
        self,
        path: str | os.PathLike[str],
        *,
        viewport: Any = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """Queue a PNG for the next public SUMO simulation step."""
        if not self.native_rendering_enabled:
            raise RuntimeError(
                "native SUMO screenshots require sumo_config.gui=true")
        views = list(self._sumo.gui.getIDList())
        if not views:
            raise RuntimeError("sumo-gui did not expose a render view")
        view_id = views[0]
        if viewport is not None:
            self._sumo.gui.setBoundary(
                view_id,
                float(viewport.min_x), float(viewport.min_y),
                float(viewport.max_x), float(viewport.max_y),
            )
        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        self._sumo.gui.screenshot(
            view_id, str(output),
            int(width or self.sumo_config.gui_width),
            int(height or self.sumo_config.gui_height),
        )
        return str(output)

    @staticmethod
    def wait_for_native_screenshot(
        path: str | os.PathLike[str], timeout_s: float = 3.0,
    ) -> str:
        """Wait for a screenshot queued before the last SUMO step."""
        output = Path(path).resolve()
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() <= deadline:
            if output.is_file() and output.stat().st_size > 0:
                return str(output)
            time.sleep(0.02)
        raise RuntimeError(f"sumo-gui did not create screenshot: {output}")

    @staticmethod
    def _safe_id(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", value)

    @staticmethod
    def _project_position_onto_shape(
        point: Tuple[float, float], shape: List[Tuple[float, float]],
        lane_length_m: float,
    ) -> float:
        """Return SUMO lane position nearest an authored world-space pose."""
        if len(shape) < 2:
            return 0.0
        best_distance_sq = float("inf")
        best_shape_offset = 0.0
        traversed = 0.0
        total_shape_length = 0.0
        segments = []
        px, py = point
        for start, end in zip(shape, shape[1:]):
            dx = float(end[0]) - float(start[0])
            dy = float(end[1]) - float(start[1])
            length = math.hypot(dx, dy)
            segments.append((start, dx, dy, length))
            total_shape_length += length
        for start, dx, dy, length in segments:
            if length <= 1e-12:
                continue
            fraction = max(0.0, min(1.0, (
                (px - float(start[0])) * dx
                + (py - float(start[1])) * dy
            ) / (length * length)))
            qx = float(start[0]) + fraction * dx
            qy = float(start[1]) + fraction * dy
            distance_sq = (px - qx) ** 2 + (py - qy) ** 2
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_shape_offset = traversed + fraction * length
            traversed += length
        if total_shape_length <= 1e-12:
            return 0.0
        return max(0.0, min(
            float(lane_length_m),
            best_shape_offset * float(lane_length_m) / total_shape_length,
        ))

    def _route_id(self, vehicle_id: str) -> str:
        return f"va_route_{self._safe_id(vehicle_id)}"

    def _route_edges(self, vehicle: Any) -> List[str]:
        edges = route_edges_for_vehicle(vehicle, self.sumo_map)
        if not edges:
            raise ValueError(
                f"vehicle {vehicle.vehicle_id!r} has no SUMO-compatible "
                "lane route")
        return edges

    def _speed_mode(self, vehicle: Any) -> int:
        # Background actors use SUMO's complete safety/right-of-way model.
        # LLM actors retain direct behavioural authority while SUMO still
        # enforces installed acceleration and braking capabilities.
        return 102 if vehicle.is_llm else 31

    def _ensure_sumo_vehicle(self, vehicle: Any) -> None:
        vehicle_id = vehicle.vehicle_id
        if (vehicle_id in self._sumo_started_ids or vehicle.arrived
                or vehicle.route_failed):
            return
        edges = self._route_edges(vehicle)
        route_id = self._route_id(vehicle_id)
        self._sumo.route.add(route_id, edges)
        self._sumo_installed_routes[vehicle_id] = list(edges)
        lane_index = self.sumo_map.lane_index_by_lane.get(
            vehicle.current_lane_id, max(0, int(vehicle.current_lane)))
        self._sumo.vehicle.add(
            vehicle_id,
            route_id,
            depart="now",
            departLane=str(lane_index),
            # This is only a materialisation position. After SUMO's first
            # private step every vehicle is moved back to its authoritative
            # VehicleArena lane/connector position before public time moves.
            departPos="random_free",
            departSpeed="0",
            arrivalLane="current",
            arrivalPos="max",
        )
        self._sumo.vehicle.setLength(vehicle_id, float(vehicle.length_m))
        self._sumo.vehicle.setWidth(vehicle_id, float(vehicle.width_m))
        self._sumo.vehicle.setColor(
            vehicle_id,
            self._LLM_COLOR if vehicle.is_llm else self._BACKGROUND_COLOR,
        )
        self._sumo.vehicle.setAccel(
            vehicle_id, float(vehicle.max_acceleration_mps2))
        self._sumo.vehicle.setDecel(
            vehicle_id, float(vehicle.max_braking_mps2))
        self._sumo.vehicle.setEmergencyDecel(
            vehicle_id, float(vehicle.max_braking_mps2))
        self._sumo.vehicle.setSpeedMode(
            vehicle_id, self._speed_mode(vehicle))
        # 1621 is SUMO's native strategic/cooperative/speed-gain/keep-right
        # lane-changing mode. LLM actors change lanes only by explicit tool.
        self._sumo.vehicle.setLaneChangeMode(
            vehicle_id, 0 if vehicle.is_llm else 1621)
        self._initialize_native_driver(vehicle)
        self._sumo_started_ids.add(vehicle_id)
        self._sumo_last_edge[vehicle_id] = edges[0]

    def _initialize_native_driver(self, vehicle: Any) -> None:
        """Apply a driver personality once, never speed/lane commands per tick."""
        config = self.sumo_config.npc_behavior
        vid = vehicle.vehicle_id
        if (not config or vehicle.is_llm or vehicle.is_crashed
                or vid not in config["vehicle_ids"]):
            return
        assignment = sample_driver(config["seed"], vid)
        api = self._sumo.vehicle
        api.setSpeedFactor(vid, assignment["speedFactor"])
        api.setTau(vid, assignment["tau"])
        api.setMinGap(vid, assignment["minGap"])
        api.setImperfection(vid, assignment["sigma"])
        for name in ("lcSpeedGain", "lcCooperative"):
            api.setParameter(vid, f"laneChangeModel.{name}", str(assignment[name]))
        self.npc_behavior_assignments[vid] = {
            "schema": config["schema"], "seed": config["seed"], **assignment}

    def register_pedestrian(self, *args: Any, **kwargs: Any) -> Any:
        """Register a pedestrian whose mapped physical pose belongs to SUMO."""
        pedestrian = super().register_pedestrian(*args, **kwargs)
        pedestrian.physical_pose_authority = "sumo"
        return pedestrian

    def unregister_pedestrian(self, ped_id: str) -> None:
        self._retire_sumo_pedestrian(ped_id)
        super().unregister_pedestrian(ped_id)

    def _pedestrian_type_id(self, pedestrian: Any) -> str:
        diameter = max(0.2, 2.0 * float(pedestrian.collision_radius_m))
        type_id = f"va_pedtype_{diameter:.3f}".replace(".", "_")
        if type_id not in self._sumo_pedestrian_types:
            self._sumo.vehicletype.copy("DEFAULT_PEDTYPE", type_id)
            self._sumo.vehicletype.setWidth(type_id, diameter)
            self._sumo.vehicletype.setLength(type_id, diameter)
            self._sumo.vehicletype.setMaxSpeed(type_id, 5.0)
            self._sumo.vehicletype.setColor(
                type_id, (245, 193, 66, 255))
            self._sumo_pedestrian_types.add(type_id)
        return type_id

    def _crosswalk_pedestrian_spec(
        self, pedestrian: Any,
    ) -> Optional[Dict[str, Any]]:
        crosswalk_id = (
            pedestrian.active_crosswalk_id
            or pedestrian.authored_crosswalk_id)
        edge_id = self.sumo_map.crosswalk_edge_by_id.get(
            str(crosswalk_id or ""))
        if not edge_id:
            return None
        lane_id = f"{edge_id}_0"
        length = max(0.001, float(self._sumo.lane.getLength(lane_id)))
        lane_shape = list(self._sumo.lane.getShape(lane_id))
        reference_path = (
            pedestrian.crosswalk_path_xy
            or pedestrian.authored_crosswalk_path_xy)
        reference_start = (
            tuple(reference_path[0]) if reference_path else
            self._lane_geometry.nodes_xy.get(
                pedestrian.position.crossing_from
                or pedestrian.position.at_node)
        )
        forward = True
        if reference_start is not None and len(lane_shape) >= 2:
            forward = (
                math.dist(tuple(reference_start), tuple(lane_shape[0]))
                <= math.dist(tuple(reference_start), tuple(lane_shape[-1])))
        progress = (
            float(pedestrian.crossing_progress)
            if pedestrian.is_on_crosswalk else 0.0)
        progress = max(0.0, min(1.0, progress))
        start_pos = progress * length if forward else (1.0 - progress) * length
        return {
            "kind": "crosswalk",
            "edge_id": edge_id,
            "lane_id": lane_id,
            "start_pos": start_pos,
            "arrival_pos": length if forward else 0.0,
            "start_progress": progress,
            "full_length": length,
            "target_node": (
                pedestrian.position.crossing_to
                or pedestrian.authored_crosswalk_to_node
                or pedestrian.next_node),
            "crosswalk_id": str(crosswalk_id),
        }

    def _sidewalk_pedestrian_spec(
        self, pedestrian: Any,
    ) -> Optional[Dict[str, Any]]:
        current_node = pedestrian.position.at_node
        next_node = pedestrian.next_node
        if not current_node or not next_node:
            return None
        direct_key = pedestrian_node_pair_key(current_node, next_node)
        reverse_key = pedestrian_node_pair_key(next_node, current_node)
        edge_id = self.sumo_map.pedestrian_edge_by_node_pair.get(direct_key)
        forward = edge_id is not None
        if edge_id is None:
            edge_id = self.sumo_map.pedestrian_edge_by_node_pair.get(
                reverse_key)
        if edge_id is None:
            return None
        lane_id = self.sumo_map.sidewalk_lane_by_edge.get(edge_id)
        if not lane_id:
            return None
        length = max(0.001, float(self._sumo.lane.getLength(lane_id)))
        progress = (
            float(pedestrian.walking_progress)
            if pedestrian.is_walking else 0.0)
        progress = max(0.0, min(1.0, progress))
        start_pos = progress * length if forward else (1.0 - progress) * length
        return {
            "kind": "sidewalk",
            "edge_id": edge_id,
            "lane_id": lane_id,
            "start_pos": start_pos,
            "arrival_pos": length if forward else 0.0,
            "start_progress": progress,
            "full_length": length,
            "target_node": next_node,
            "crosswalk_id": "",
        }

    def _pedestrian_spec(self, pedestrian: Any) -> Optional[Dict[str, Any]]:
        if (pedestrian.is_on_crosswalk
                or pedestrian.authored_crosswalk_id):
            return self._crosswalk_pedestrian_spec(pedestrian)
        return self._sidewalk_pedestrian_spec(pedestrian)

    def _retire_sumo_pedestrian(self, ped_id: str) -> None:
        proxy_id = self._sumo_pedestrian_proxy.pop(ped_id, None)
        self._sumo_pedestrian_leg.pop(ped_id, None)
        if proxy_id is None:
            return
        self._sumo_pedestrian_owner.pop(proxy_id, None)
        try:
            # ``remove`` also cancels a person that has been added but has not
            # yet reached its departure step. This keeps repeated decisions
            # in one frozen wake batch from leaving an orphan SUMO proxy.
            self._sumo.person.remove(proxy_id)
        except Exception:
            # A naturally completed person is already absent from SUMO.
            pass

    def _spawn_sumo_pedestrian(
        self, pedestrian: Any, spec: Dict[str, Any], *, moving: bool,
    ) -> None:
        ped_id = pedestrian.ped_id
        self._retire_sumo_pedestrian(ped_id)
        epoch = self._sumo_pedestrian_epoch.get(ped_id, -1) + 1
        self._sumo_pedestrian_epoch[ped_id] = epoch
        proxy_id = (
            f"va_ped_{self._safe_id(ped_id)}_{epoch}")
        self._sumo.person.add(
            proxy_id,
            spec["edge_id"],
            float(spec["start_pos"]),
            -3.0,
            self._pedestrian_type_id(pedestrian),
        )
        if moving:
            self._sumo.person.appendWalkingStage(
                proxy_id, [spec["edge_id"]],
                float(spec["arrival_pos"]),
                speed=max(0.01, float(pedestrian.speed)),
            )
        self._sumo.person.appendWaitingStage(
            proxy_id, 1_000_000_000.0, "VehicleArena decision wait")
        self._sumo_pedestrian_proxy[ped_id] = proxy_id
        self._sumo_pedestrian_owner[proxy_id] = ped_id
        pedestrian.physical_pose_authority = "sumo"
        if moving:
            if spec["kind"] == "crosswalk":
                pedestrian.crosswalk_length_m = float(spec["full_length"])
            self._sumo_pedestrian_leg[ped_id] = dict(spec)

    def _ensure_sumo_pedestrian(
        self, pedestrian: Any, depart_before_s: float,
    ) -> None:
        if pedestrian.has_arrived:
            self._retire_sumo_pedestrian(pedestrian.ped_id)
            return
        if pedestrian.start_time > depart_before_s + 1e-9:
            return
        if pedestrian.ped_id in self._sumo_pedestrian_proxy:
            return
        # Background people start/resume their mapped itinerary without a
        # VehicleArena policy decision. SUMO owns walking, right-of-way and
        # interaction with traffic once the stage is installed.
        if (not pedestrian.is_llm and not pedestrian.is_crashed
                and not pedestrian.is_on_crosswalk
                and not pedestrian.is_walking):
            if pedestrian.authored_crosswalk_id:
                pedestrian.start_crossing(
                    pedestrian.position.at_node or "",
                    pedestrian.authored_crosswalk_to_node
                    or pedestrian.next_node or "",
                    crosswalk_id=pedestrian.authored_crosswalk_id,
                    path_xy=pedestrian.authored_crosswalk_path_xy,
                )
                pedestrian.crossing_progress = float(
                    pedestrian.authored_crosswalk_progress)
            elif pedestrian.next_node:
                pedestrian.set_walking(speed=pedestrian.base_speed)
        spec = self._pedestrian_spec(pedestrian)
        if spec is None:
            raise ValueError(
                f"pedestrian {pedestrian.ped_id!r} has no SUMO-mapped "
                "sidewalk/crosswalk leg")
        moving = bool(
            (pedestrian.is_on_crosswalk or pedestrian.is_walking)
            and pedestrian.speed > 0.0 and not pedestrian.is_waiting)
        self._spawn_sumo_pedestrian(pedestrian, spec, moving=moving)

    def _submit_pedestrian_controls(self, depart_before_s: float) -> None:
        for pedestrian in self.pedestrians.values():
            self._ensure_sumo_pedestrian(pedestrian, depart_before_s)
        active_ids = set(self._sumo.person.getIDList())
        for ped_id, proxy_id in list(self._sumo_pedestrian_proxy.items()):
            pedestrian = self.pedestrians.get(ped_id)
            if pedestrian is None or pedestrian.has_arrived:
                self._retire_sumo_pedestrian(ped_id)
                continue
            if proxy_id not in active_ids:
                continue
            if pedestrian.is_llm:
                speed = (
                    0.0 if pedestrian.is_crashed or pedestrian.is_waiting
                    else max(0.0, float(pedestrian.speed)))
                self._sumo.person.setSpeed(proxy_id, speed)

    def _bootstrap_initial_sumo_state(self) -> None:
        """Materialise departures, then restore the authored t=0 state.

        SUMO does not create a queryable vehicle object until one simulation
        step has completed. Calling ``moveTo`` before that boundary can crash
        libsumo. The private bootstrap step is therefore kept off the public
        VehicleArena clock; every vehicle is then placed on its exact external
        lane or internal connector and receives its authored speed.
        """
        if self._initial_state_bootstrapped:
            return
        self._sumo.simulationStep(
            self._sumo_time_s + self.sumo_config.step_length_s)
        self._sumo_time_s = float(self._sumo.simulation.getTime())
        active_ids = set(self._sumo.vehicle.getIDList())
        expected_ids = {
            vehicle_id for vehicle_id, vehicle in self.vehicles.items()
            if not vehicle.arrived
        }
        missing = sorted(expected_ids - active_ids)
        if missing:
            raise RuntimeError(
                "SUMO could not materialise initial vehicles: "
                f"{missing[:5]}")

        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in active_ids:
                continue
            if vehicle.active_connector_id:
                lane_id = self.sumo_map.connector_via_lane.get(
                    vehicle.active_connector_id, "")
                if not lane_id:
                    raise RuntimeError(
                        "SUMO map has no internal lane for initial connector "
                        f"{vehicle.active_connector_id!r}")
            else:
                edge = self.sumo_map.edge_by_lane.get(vehicle.current_lane_id)
                lane_index = self.sumo_map.lane_index_by_lane.get(
                    vehicle.current_lane_id)
                if edge is None or lane_index is None:
                    raise RuntimeError(
                        "SUMO map has no lane for initial vehicle state "
                        f"{vehicle.current_lane_id!r}")
                lane_id = f"{edge}_{lane_index}"
            lane_length = max(0.001, float(
                self._sumo.lane.getLength(lane_id)))
            # Scenario progress is defined on the authored lane geometry,
            # while netconvert may trim a few metres at a junction.  Project
            # the already-authored body-centre pose onto SUMO's compiled lane
            # instead of multiplying two non-identical lane lengths.
            lane_position = self._project_position_onto_shape(
                (float(vehicle.pose_x_m), float(vehicle.pose_y_m)),
                list(self._sumo.lane.getShape(lane_id)), lane_length)
            if (self._has_unresolved_route_endpoint(vehicle)
                    and lane_position >= lane_length - 0.001):
                raise ValueError(
                    "Initial LLM placement projects onto a closed route endpoint: "
                    f"vehicle={vehicle_id}, authored_lane={vehicle.current_lane_id}, "
                    f"compiled_lane={lane_id}, length_m={lane_length:.3f}, "
                    f"projected_position_m={lane_position:.3f}. "
                    "Check compiled junction geometry and scenario placement.")
            lane_position = max(
                0.0, min(lane_length - 0.001, lane_position))
            self._sumo.vehicle.moveTo(
                vehicle_id, lane_id, lane_position)
            # TraCI applies moveTo at the next simulation boundary. Freeze the
            # body for that private commit so authored speed cannot turn into
            # hidden episode travel.
            self._sumo.vehicle.setPreviousSpeed(vehicle_id, 0.0, 0.0)
            self._sumo.vehicle.setSpeed(vehicle_id, 0.0)

        active_pedestrian_ids = set(self._sumo.person.getIDList())
        expected_pedestrian_ids = {
            proxy_id for ped_id, proxy_id in
            self._sumo_pedestrian_proxy.items()
            if self.pedestrians[ped_id].start_time <= 1e-9
        }
        missing_pedestrians = sorted(
            expected_pedestrian_ids - active_pedestrian_ids)
        if missing_pedestrians:
            raise RuntimeError(
                "SUMO could not materialise initial pedestrians: "
                f"{missing_pedestrians[:5]}")
        for proxy_id in active_pedestrian_ids:
            self._sumo.person.setSpeed(proxy_id, 0.0)

        # Commit every deferred moveTo while the actors are frozen.  This is
        # still private bootstrap time: VehicleArena remains at t=0 and no
        # evaluator or trajectory logger can observe either materialisation
        # step.
        self._sumo.simulationStep(
            self._sumo_time_s + self.sumo_config.step_length_s)
        self._sumo_time_s = float(self._sumo.simulation.getTime())
        active_ids = set(self._sumo.vehicle.getIDList())
        missing = sorted(expected_ids - active_ids)
        if missing:
            raise RuntimeError(
                "SUMO lost vehicles while committing initial placement: "
                f"{missing[:5]}")

        # Only after the authoritative position is queryable do we restore
        # the authored initial speed and normal SUMO/LLM speed authority.
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in active_ids:
                continue
            current_speed = (
                0.0 if vehicle.is_crashed
                else max(0.0, float(vehicle.current_speed_kmh) / 3.6))
            acceleration = (
                0.0 if vehicle.is_crashed
                else float(vehicle.acceleration_mps2))
            self._sumo.vehicle.setPreviousSpeed(
                vehicle_id, current_speed, acceleration)
            if vehicle.is_crashed:
                self._sumo.vehicle.setSpeed(vehicle_id, 0.0)
            elif vehicle.is_llm:
                self._sumo.vehicle.setSpeed(
                    vehicle_id,
                    max(0.0, float(vehicle.target_speed_kmh) / 3.6),
                )
            else:
                self._sumo.vehicle.setSpeed(vehicle_id, -1.0)
            self._sumo_last_edge[vehicle_id] = str(
                self._sumo.vehicle.getRoadID(vehicle_id))
            self._sumo_last_distance_m[vehicle_id] = max(
                0.0, float(self._sumo.vehicle.getDistance(vehicle_id)))
        # Unlike vehicles, a newly departed SUMO person does not advance on
        # its materialisation step. Calling person.moveTo on an internal
        # crossing would instead snap it to the stage endpoint, so the exact
        # authored departPos is already the correct hidden-step restoration.
        # Release the private bootstrap freeze once. Background pedestrians
        # receive no per-step speed commands after initialization.
        for proxy_id in self._sumo.person.getIDList():
            self._sumo.person.setSpeed(proxy_id, -1.0)
        self._initial_state_bootstrapped = True

    def _publish_bootstrap_poses(self) -> None:
        """Publish SUMO's t=0 placement before the first public snapshot."""
        active_ids = set(self._sumo.vehicle.getIDList())
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in active_ids:
                continue
            x, y = self._sumo.vehicle.getPosition(vehicle_id)
            angle = float(self._sumo.vehicle.getAngle(vehicle_id))
            vehicle.pose_x_m = float(x)
            vehicle.pose_y_m = float(y)
            # Episode mileage starts at the first public SUMO pose.  SUMO's
            # hidden materialisation travel and route-coordinate position are
            # never part of the evaluated trip distance.
            vehicle.distance_traveled_m = 0.0
            self._sumo_last_position_m[vehicle_id] = (float(x), float(y))
            vehicle.heading = angle
            vehicle.yaw_rad = math.radians(90.0 - angle)
            vehicle.physical_pose_authority = "sumo"
        active_pedestrian_ids = set(self._sumo.person.getIDList())
        for ped_id, proxy_id in self._sumo_pedestrian_proxy.items():
            if proxy_id not in active_pedestrian_ids:
                continue
            pedestrian = self.pedestrians[ped_id]
            x, y = self._sumo.person.getPosition(proxy_id)
            pedestrian.physical_pose_xy = [float(x), float(y)]
            angle = float(self._sumo.person.getAngle(proxy_id))
            pedestrian.physical_yaw_rad = math.radians(90.0 - angle)
            pedestrian.physical_pose_authority = "sumo"

    def _sync_signal_states(self, public_time_s: float) -> None:
        """Keep SUMO signals on VehicleArena's public simulation clock."""
        states: Dict[str, List[str]] = {
            tls_id: ["r"] * link_count
            for tls_id, link_count in self.sumo_map.tls_link_count.items()
        }
        for connector_id, link_index in (
                self.sumo_map.connector_link_index.items()):
            tls_id = self.sumo_map.connector_tls[connector_id]
            state = states[tls_id]
            signal = self._lane_geometry.signal_state(
                connector_id, public_time_s)
            state[link_index] = {
                "green": "g", "yellow": "y", "red": "r",
            }.get(signal.signal if signal is not None else "red", "r")
        # Pedestrian links follow the authored pedestrian signal phase. SUMO
        # then decides whether native pedestrians wait or enter the crossing.
        for crosswalk_id, link_index in (
                self.sumo_map.crosswalk_link_index.items()):
            tls_id = self.sumo_map.crosswalk_tls[crosswalk_id]
            signal = self._lane_geometry.pedestrian_signal_state(
                tls_id, public_time_s)
            states[tls_id][link_index] = (
                "G" if signal is not None and signal.pedestrian_green
                else "r")
        for tls_id, state in states.items():
            self._sumo.trafficlight.setRedYellowGreenState(
                tls_id, "".join(state))

    def _sync_lane_constraints(self, public_time_s: float) -> None:
        """Apply authored closures and speed zones to SUMO lanes.

        Traffic observations never become a second speed cap; SUMO's
        car-following state is the physical source of congestion.
        """
        active_by_segment: Dict[str, List[Any]] = {}
        for event in self.road_network.road_events:
            if event.start_sec <= public_time_s < event.end_sec:
                active_by_segment.setdefault(event.edge_id, []).append(event)
        affected_segments = (
            set(active_by_segment)
            | {
                segment_id
                for segment_id, lanes in self._sumo_lanes_by_segment.items()
                if any(
                    lane_id in self._sumo_lane_constraint_state
                    for _, lane_id in lanes)
            }
        )
        for segment_id in affected_segments:
            events = active_by_segment.get(segment_id, [])
            closed = any(
                event.event_type == "road_closure" for event in events)
            for _, lane_id in self._sumo_lanes_by_segment.get(segment_id, []):
                default_speed, default_allowed = self._sumo_lane_defaults[lane_id]
                speed = min(
                    [default_speed] + [
                        max(0.1, float(event.speed_limit_in_zone) / 3.6)
                        for event in events
                        if event.event_type != "road_closure"
                    ])
                desired = (closed, speed)
                if self._sumo_lane_constraint_state.get(lane_id) == desired:
                    continue
                if closed:
                    self._sumo.lane.setDisallowed(
                        lane_id, list(default_allowed))
                else:
                    self._sumo.lane.setAllowed(
                        lane_id, list(default_allowed))
                self._sumo.lane.setMaxSpeed(lane_id, speed)
                if desired == (False, default_speed):
                    self._sumo_lane_constraint_state.pop(lane_id, None)
                else:
                    self._sumo_lane_constraint_state[lane_id] = desired

    def _sumo_collision_entity(
        self, sumo_id: str,
    ) -> Optional[Tuple[str, str]]:
        if sumo_id in self.vehicles:
            return sumo_id, "vehicle"
        ped_id = self._sumo_pedestrian_owner.get(sumo_id)
        if (ped_id is not None
                and self._sumo_pedestrian_proxy.get(ped_id) == sumo_id):
            return ped_id, "pedestrian"
        return None

    def _apply_entity_collision(
        self, a_id: str, a_type: str, b_id: str, b_type: str,
        *args: Any, **kwargs: Any,
    ) -> None:
        super()._apply_entity_collision(
            a_id, a_type, b_id, b_type, *args, **kwargs)
        for entity_id, entity_type in ((a_id, a_type), (b_id, b_type)):
            if entity_type == "pedestrian":
                self._sumo_pedestrian_leg.pop(entity_id, None)

    def _entity_pose(self, entity_id: str, entity_type: str) -> Any:
        if entity_type == "vehicle":
            return self._lane_geometry.vehicle_pose(self.vehicles[entity_id])
        pose = self._lane_geometry.pedestrian_pose(
            self.pedestrians[entity_id])
        return None if pose is None else tuple(pose)

    def _commit_sumo_collisions(
        self, public_time_s: float,
    ) -> None:
        """Commit vehicle and intermodal contacts reported by SUMO."""
        for collision in self._sumo.simulation.getCollisions():
            collider_raw = str(collision.collider)
            victim_raw = str(collision.victim)
            collider_entity = self._sumo_collision_entity(collider_raw)
            victim_entity = self._sumo_collision_entity(victim_raw)
            if collider_entity is None or victim_entity is None:
                continue
            collider, collider_type = collider_entity
            victim, victim_type = victim_entity
            if {collider_type, victim_type} not in (
                    {"vehicle"}, {"vehicle", "pedestrian"}):
                continue
            pair = frozenset((collider, victim))
            if pair in self._sumo_collision_pairs:
                continue
            self._sumo_collision_pairs.add(pair)
            collider_pose = self._entity_pose(collider, collider_type)
            victim_pose = self._entity_pose(victim, victim_type)
            vehicle_id = (
                collider if collider_type == "vehicle" else victim)
            vehicle = self.vehicles[vehicle_id]
            location = (
                vehicle.active_connector_id or vehicle.current_segment
                or str(collision.lane))
            collision_type = (
                f"sumo_{str(collision.type)}"
                if collider_type == victim_type == "vehicle"
                else "vehicle_pedestrian")
            self._collision_log.append(CollisionEvent(
                entity_a=collider,
                entity_b=victim,
                entity_a_type=collider_type,
                entity_b_type=victim_type,
                location=location,
                collision_type=collision_type,
                tick=round(public_time_s),
                step=0,
                time_s=public_time_s,
                physics_source="sumo",
            ))
            if collider_type == victim_type == "vehicle":
                self._apply_collision(
                    collider, victim, location, collision_type,
                    round(public_time_s), 0,
                    time_s=public_time_s,
                    impact_pose_a=collider_pose,
                    impact_pose_b=victim_pose,
                )
            else:
                self._apply_entity_collision(
                    collider, collider_type, victim, victim_type,
                    location, collision_type, round(public_time_s), 0,
                    time_s=public_time_s,
                    impact_pose_a=collider_pose,
                    impact_pose_b=victim_pose,
                )

    def _install_route(self, vehicle: Any) -> None:
        if vehicle.vehicle_id not in self._sumo_started_ids:
            return
        edges = self._route_edges(vehicle)
        current = self._sumo.vehicle.getRoadID(vehicle.vehicle_id)
        if current.startswith(":"):
            return
        if current in edges:
            edges = edges[edges.index(current):]
        elif current:
            edges.insert(0, current)
        self._sumo.vehicle.setRoute(vehicle.vehicle_id, edges)
        self._sumo_installed_routes[vehicle.vehicle_id] = list(edges)

    def _accept_llm_route_arrival(self, vehicle: Any) -> bool:
        """Validate SUMO arrival even if its final lane fit between two ticks.

        An arrived vehicle is no longer queryable. Keep the last installed
        route as evidence, and require an explicitly selected connector when
        the cached lane has not yet reached the destination approach.
        """
        if self._is_destination_lane(vehicle):
            return True
        connector_id = vehicle.active_connector_id or vehicle.planned_connector_id
        connector = (self._lane_geometry.connector_record(connector_id)
                     if connector_id else None)
        if not connector or connector["from_lane"] != vehicle.current_lane_id:
            return False
        target_id = connector["to_lane"]
        target = self._lane_geometry._lane_by_id.get(target_id)
        route = self._sumo_installed_routes.get(vehicle.vehicle_id, [])
        junctions = self._lane_geometry._node_to_junction
        if (not target or not route or not vehicle.destination_node
                or route[-1] != self.sumo_map.edge_by_lane.get(target_id)
                or junctions.get(str(target["end_node"]), str(target["end_node"]))
                != junctions.get(str(vehicle.destination_node), str(vehicle.destination_node))):
            return False
        # Correct terminal topology without inventing a post-removal physical
        # sample or changing the last measured crossing speed.
        vehicle.current_lane_id = target_id
        vehicle.current_lane = int(target["index"])
        vehicle.current_segment = str(target["segment_id"])
        vehicle.current_node = str(target["end_node"])
        vehicle.edge_progress = 1.0
        vehicle.active_connector_id = ""
        vehicle.active_connector_from_lane_id = ""
        vehicle.active_connector_to_lane_id = ""
        self._clear_authorized_maneuver(vehicle)
        vehicle.active_control_commands.pop("route_maneuver", None)
        return True

    def _is_destination_lane(self, vehicle: Any) -> bool:
        """Whether the current lane legitimately terminates the assigned trip."""
        return self._current_lane_terminates_at_destination(vehicle)

    def _has_unresolved_route_endpoint(self, vehicle: Any) -> bool:
        return bool(
            vehicle.route_control_authority == "llm_maneuver"
            and not vehicle.is_crashed
            and not vehicle.arrived
            and not vehicle.route_failed
            and not vehicle.active_connector_id
            and not vehicle.planned_connector_id
            and not self._is_destination_lane(vehicle)
        )

    def select_vehicle_maneuver(
        self, vehicle_id: str, direction: str,
    ) -> dict:
        """Authorise one connector and extend the live SUMO route by one edge."""
        result = super().select_vehicle_maneuver(vehicle_id, direction)
        if not result.get("success"):
            return result
        vehicle = self.vehicles[vehicle_id]
        self._install_route(vehicle)
        return result

    def set_vehicle_destination(
        self, vehicle_id: str, target_node: str, speed_kmh: float = -1,
    ) -> dict:
        result = super().set_vehicle_destination(
            vehicle_id, target_node, speed_kmh)
        if result.get("success"):
            self._install_route(self.vehicles[vehicle_id])
        return result

    def u_turn_vehicle(self, vehicle_id: str) -> dict:
        """Install an explicitly requested U-turn in SUMO exactly once."""
        result = super().u_turn_vehicle(vehicle_id)
        if result.get("success"):
            self._install_route(self.vehicles[vehicle_id])
        return result

    def execute_pedestrian_action(
        self, ped_id: str, action: str, params: dict = None,
    ) -> dict:
        """Apply an LLM pedestrian decision, then execute it in SUMO."""
        pedestrian = self.pedestrians.get(ped_id)
        if pedestrian is not None and not pedestrian.is_llm:
            return {"success": False, "reason": "pedestrian_is_sumo_controlled"}
        result = super().execute_pedestrian_action(ped_id, action, params)
        if not result.get("success") or pedestrian is None:
            return result

        proxy_id = self._sumo_pedestrian_proxy.get(ped_id)
        active_ids = set(self._sumo.person.getIDList())
        if action == "pedestrian_wait":
            if proxy_id in active_ids:
                self._sumo.person.setSpeed(proxy_id, 0.0)
            return result

        if action in {"pedestrian_walk", "pedestrian_run"}:
            # A pause/resume keeps the existing SUMO walking stage and its
            # exact progress. An idle walker starts one new physical leg.
            if ped_id in self._sumo_pedestrian_leg:
                if proxy_id in active_ids:
                    self._sumo.person.setSpeed(
                        proxy_id, max(0.01, float(pedestrian.speed)))
                return result

        if action not in {
                "pedestrian_walk", "pedestrian_run",
                "pedestrian_cross", "pedestrian_change_route"}:
            return result
        if (action == "pedestrian_change_route"
                and (pedestrian.is_walking
                     or pedestrian.is_on_crosswalk)):
            return result
        spec = self._pedestrian_spec(pedestrian)
        if spec is None:
            self._retire_sumo_pedestrian(ped_id)
            raise ValueError(
                f"pedestrian {ped_id!r} has no SUMO-mapped "
                "sidewalk/crosswalk leg")
        moving = bool(
            action != "pedestrian_change_route"
            and (pedestrian.is_walking or pedestrian.is_on_crosswalk)
            and pedestrian.speed > 0.0)
        self._spawn_sumo_pedestrian(
            pedestrian, spec, moving=moving)
        return result

    def _submit_controls(
        self, pedestrian_depart_before_s: Optional[float] = None,
    ) -> None:
        for vehicle in self.vehicles.values():
            if vehicle.route_failed:
                continue
            if self._prepare_default_continuation(vehicle):
                self._install_route(vehicle)
            self._ensure_sumo_vehicle(vehicle)
        active_ids = set(self._sumo.vehicle.getIDList())
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in active_ids:
                continue
            if not vehicle.is_llm:
                continue
            if vehicle.is_crashed:
                self._sumo.vehicle.setSpeed(vehicle_id, 0.0)
                continue
            self._sumo.vehicle.setAccel(
                vehicle_id,
                min(
                    float(vehicle.max_acceleration_mps2),
                    max(0.1, float(
                        vehicle.control_acceleration_limit_mps2)),
                ),
            )
            self._sumo.vehicle.setDecel(
                vehicle_id,
                min(
                    float(vehicle.max_braking_mps2),
                    max(0.1, float(
                        vehicle.control_deceleration_limit_mps2)),
                ),
            )
            self._sumo.vehicle.setEmergencyDecel(
                vehicle_id, float(vehicle.max_braking_mps2))
            target = max(0.0, float(vehicle.target_speed_kmh) / 3.6)
            self._sumo.vehicle.setSpeed(vehicle_id, target)
            if (vehicle.is_llm and vehicle.is_changing_lane
                    and vehicle.target_lane >= 0
                    and not vehicle.active_connector_id):
                target_lane_id = self._lane_id_for_index(
                    vehicle, int(vehicle.target_lane))
                if not target_lane_id:
                    continue
                sumo_target = self.sumo_map.lane_index_by_lane[
                    target_lane_id]
                change = self._sumo_lane_changes.get(vehicle_id)
                if change is None:
                    duration = max(0.5, float(vehicle.lane_change_duration_s))
                    self._sumo.vehicle.changeLane(
                        vehicle_id, sumo_target, duration)
                    self._sumo_lane_changes[vehicle_id] = (
                        sumo_target, target_lane_id,
                        self._physics_time, duration)
                    self._sumo_lane_change_observations[vehicle_id] = (
                        self._sumo_lane_changes[vehicle_id])
        self._submit_pedestrian_controls(
            self._physics_time if pedestrian_depart_before_s is None
            else float(pedestrian_depart_before_s))

    def _finish_lane_change(self, vehicle: Any, lane_index: int) -> None:
        edge = self._sumo.vehicle.getRoadID(vehicle.vehicle_id)
        lane_id = self.sumo_map.lane_by_edge_index.get(
            edge, {}).get(int(lane_index))
        if lane_id:
            vehicle.current_lane_id = lane_id
            vehicle.current_lane = int(
                self._lane_geometry._lane_by_id[lane_id]["index"])
        vehicle.is_changing_lane = False
        vehicle.target_lane = -1
        vehicle.lane_change_progress = 0.0
        if vehicle.lane_route_action_index < len(vehicle.lane_route_actions):
            action = vehicle.lane_route_actions[vehicle.lane_route_action_index]
            if (action.get("type") == "lane_change"
                    and action.get("to_lane_id") == lane_id):
                vehicle.lane_route_action_index += 1
        self._prepare_lane_transition(vehicle)

    def _connector_for_transition(
        self, vehicle: Any, target_lane_id: str = "",
    ) -> str:
        if vehicle.active_connector_id:
            return vehicle.active_connector_id
        if vehicle.planned_connector_id:
            return vehicle.planned_connector_id
        for action in vehicle.lane_route_actions[
                vehicle.lane_route_action_index:]:
            if action.get("type") != "connector":
                continue
            if (not target_lane_id
                    or action.get("to_lane_id") == target_lane_id):
                return str(action.get("connector_id", ""))
        return ""

    def _sync_external_lane(
        self, vehicle: Any, edge: str, lane_index: int,
    ) -> None:
        previous_lane_id = vehicle.current_lane_id
        lane_id = self.sumo_map.lane_by_edge_index.get(
            edge, {}).get(lane_index)
        if not lane_id:
            return
        lane = self._lane_geometry._lane_by_id[lane_id]
        previous_edge = self._sumo_last_edge.get(vehicle.vehicle_id, "")
        crossed_junction = (previous_edge.startswith(":") or (
            bool(previous_edge) and previous_edge != edge
            and self.sumo_map.edge_by_lane.get(previous_lane_id) != edge))
        if crossed_junction:
            connector_id = self._connector_for_transition(vehicle, lane_id)
            if connector_id:
                for index in range(
                        vehicle.lane_route_action_index,
                        len(vehicle.lane_route_actions)):
                    action = vehicle.lane_route_actions[index]
                    if (action.get("type") == "connector"
                            and action.get("connector_id") == connector_id):
                        vehicle.lane_route_action_index = index + 1
                        break
            vehicle.active_connector_id = ""
            vehicle.active_connector_from_lane_id = ""
            vehicle.active_connector_to_lane_id = ""
        vehicle.current_lane_id = lane_id
        vehicle.current_lane = int(lane["index"])
        vehicle.current_segment = str(lane["segment_id"])
        vehicle.current_node = str(lane["start_node"])
        vehicle.edge_progress = self._project_position_onto_shape(
            (float(vehicle.pose_x_m), float(vehicle.pose_y_m)),
            [tuple(point) for point in lane["centerline_xy"]], 1.0)
        if vehicle.route_control_authority == "llm_maneuver":
            if crossed_junction:
                self._clear_authorized_maneuver(vehicle)
                vehicle.active_control_commands.pop(
                    "route_maneuver", None)
            if previous_lane_id != lane_id:
                self._replan_lane_route(
                    vehicle, round(self._physics_time))
        self._prepare_lane_transition(vehicle)
        if self._prepare_default_continuation(vehicle):
            self._install_route(vehicle)

    def _sync_internal_lane(self, vehicle: Any) -> Optional[TriggerEvent]:
        # Actual SUMO motion, not a pending route plan, owns signal attribution.
        internal_lane_id = str(self._sumo.vehicle.getLaneID(vehicle.vehicle_id))
        connector_id = self._sumo_connector_by_internal_lane.get(internal_lane_id)
        if not connector_id:
            raise RuntimeError(
                f"unmapped SUMO internal lane {internal_lane_id!r} "
                f"for vehicle {vehicle.vehicle_id!r} at {self._physics_time:g}s")
        entered = not vehicle.active_connector_id and bool(connector_id)
        if connector_id:
            connector = self._lane_geometry.connector_record(connector_id)
            if connector:
                vehicle.active_connector_id = connector_id
                vehicle.active_connector_from_lane_id = connector["from_lane"]
                vehicle.active_connector_to_lane_id = connector["to_lane"]
                vehicle.planned_connector_id = ""
                vehicle.planned_turn = connector.get("turn", "straight")
        connector = (
            self._lane_geometry.connector_record(connector_id)
            if connector_id else None)
        if connector:
            vehicle.edge_progress = self._project_position_onto_shape(
                (float(vehicle.pose_x_m), float(vehicle.pose_y_m)),
                [tuple(point) for point in connector["centerline_xy"]], 1.0)
        if not entered:
            return None
        signal = self._lane_geometry.signal_state(
            connector_id, self._physics_time)
        return TriggerEvent(
            type="connector_entered",
            vehicle_id=vehicle.vehicle_id,
            tick=round(self._physics_time),
            step=0,
            time_s=self._physics_time,
            details={
                "connector_id": connector_id,
                "sumo_internal_lane_id": internal_lane_id,
                "turn": vehicle.planned_turn,
                "signal": signal.signal if signal else "unsignalized",
            },
        )

    def _sync_from_sumo(self) -> List[TriggerEvent]:
        events: List[TriggerEvent] = []
        active_ids = set(self._sumo.vehicle.getIDList())
        arrived_ids = set(self._sumo.simulation.getArrivedIDList())
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id in arrived_ids:
                if (vehicle.route_control_authority == "llm_maneuver"
                        and not self._accept_llm_route_arrival(vehicle)):
                    # A native route end is not necessarily the assigned
                    # destination. Do not brake, invent a collision or abort
                    # peer vehicles. Commit failure after native contacts.
                    vehicle.pending_route_failure = True
                    continue
                vehicle.pending_arrival = True
                continue
            if vehicle_id not in active_ids:
                continue
            speed_mps = max(0.0, float(
                self._sumo.vehicle.getSpeed(vehicle_id)))
            vehicle.current_speed_kmh = speed_mps * 3.6
            vehicle.acceleration_mps2 = float(
                self._sumo.vehicle.getAcceleration(vehicle_id))
            if not vehicle.is_llm:
                signal_bits = int(self._sumo.vehicle.getSignals(vehicle_id))
                # SUMO owns every background lamp.  Publish its complete
                # state through the same edge-detecting path used by cabin
                # modules; mutating the dataclass directly made the physical
                # brake-lamp helper repeatedly rediscover one unchanged edge.
                self.update_vehicle_signals(vehicle_id, {
                    "right_indicator": bool(signal_bits & 1),
                    "left_indicator": bool(signal_bits & 2),
                    "hazard": bool(
                        (signal_bits & 1) and (signal_bits & 2)),
                    "brake_light": bool(signal_bits & 8),
                    "low_beam": bool(signal_bits & 16),
                    "high_beam": bool(signal_bits & 64),
                }, self._physics_time)
            sumo_distance_m = max(
                0.0, float(self._sumo.vehicle.getDistance(vehicle_id)))
            # SUMO getDistance() can stay at zero after an initial moveTo()
            # placement on some bindings.  The synchronized pose is the
            # physical authority, so integrate its 0.1-second displacement
            # instead of silently reporting a stationary trajectory.
            x, y = self._sumo.vehicle.getPosition(vehicle_id)
            current_position = (float(x), float(y))
            previous_position = self._sumo_last_position_m.get(
                vehicle_id, current_position)
            pose_delta_m = math.hypot(
                current_position[0] - previous_position[0],
                current_position[1] - previous_position[1])
            if math.isfinite(pose_delta_m):
                vehicle.distance_traveled_m += max(0.0, pose_delta_m)
            self._sumo_last_position_m[vehicle_id] = current_position
            self._sumo_last_distance_m[vehicle_id] = sumo_distance_m
            angle = float(self._sumo.vehicle.getAngle(vehicle_id))
            vehicle.pose_x_m = current_position[0]
            vehicle.pose_y_m = current_position[1]
            vehicle.heading = angle
            vehicle.yaw_rad = math.radians(90.0 - angle)
            vehicle.physical_pose_authority = "sumo"
            # Explicitly controlled vehicles do not receive an autonomous
            # SUMO driving decision, but their physical deceleration still
            # operates the brake lamp. Background lamps remain SUMO-native.
            if vehicle.is_llm:
                self._update_physical_brake_light(
                    vehicle_id, vehicle.acceleration_mps2,
                    self._physics_time)
            edge = str(self._sumo.vehicle.getRoadID(vehicle_id))
            lane_index = int(self._sumo.vehicle.getLaneIndex(vehicle_id))
            if edge.startswith(":"):
                event = self._sync_internal_lane(vehicle)
                if event:
                    events.append(event)
            else:
                self._sync_external_lane(vehicle, edge, lane_index)
            vehicle.lateral_offset_m = float(
                self._sumo.vehicle.getLateralLanePosition(vehicle_id))
            vehicle.lateral_speed_mps = float(
                self._sumo.vehicle.getLateralSpeed(vehicle_id))
            self._sumo_last_edge[vehicle_id] = edge
        return events

    def _observe_lane_change_outcomes(self) -> None:
        """Account for native motion after contacts; never emit a wake event.

        Request expiry is not physical completion. Keep SUMO's original
        request duration and do not reissue, cancel or centre the vehicle.
        """
        for vehicle in self.vehicles.values():
            vid = vehicle.vehicle_id
            change = self._sumo_lane_change_observations.get(vid)
            terminal = (vehicle.is_crashed or vehicle.arrived
                        or vehicle.route_failed or vehicle.pending_arrival
                        or vehicle.pending_route_failure)
            if terminal:
                if vehicle.lane_change_request_outcome == "pending":
                    vehicle.lane_change_uncompleted += 1
                    vehicle.lane_change_request_outcome = "uncompleted"
                self._sumo_lane_changes.pop(vid, None)
                self._sumo_lane_change_observations.pop(vid, None)
                vehicle.is_changing_lane = False
                vehicle.target_lane = -1
                continue
            if change is None:
                continue
            _, target_lane_id, started_at, duration = change
            if (vehicle.lane_change_request_outcome == "pending"
                    and vehicle.current_lane_id == target_lane_id
                    and not vehicle.active_connector_id
                    # SUMO may finish at a non-centred lateral alignment.
                    # Require the whole body within the target lane, not
                    # merely its centre crossing the lane-index boundary.
                    and abs(vehicle.lateral_offset_m) + vehicle.width_m / 2
                    <= self._sumo.lane.getWidth(
                        self._sumo.vehicle.getLaneID(vid)) / 2 + 0.05
                    and abs(vehicle.lateral_speed_mps) <= 0.1
                    and not self._sumo.vehicle.isStopped(vid)):
                vehicle.lane_change_completions += 1
                vehicle.lane_change_request_outcome = "completed"
            if self._physics_time >= started_at + duration - 1e-9:
                # A native maneuver can still be settling after the request
                # expires. Observe it without extending the native command.
                if (vehicle.lane_change_request_outcome == "pending"
                        and (abs(vehicle.lateral_speed_mps) <= 0.1
                             or self._sumo.vehicle.isStopped(vid))):
                    vehicle.lane_change_uncompleted += 1
                    vehicle.lane_change_request_outcome = "uncompleted"
                if vid in self._sumo_lane_changes:
                    self._finish_lane_change(
                        vehicle, int(self._sumo.vehicle.getLaneIndex(vid)))
                    self._sumo_lane_changes.pop(vid, None)
            if (vehicle.lane_change_request_outcome != "pending"
                    and vid not in self._sumo_lane_changes):
                self._sumo_lane_change_observations.pop(vid, None)

    def _sync_pedestrians_from_sumo(self) -> List[TriggerEvent]:
        """Publish SUMO person poses and close completed decision legs."""
        events: List[TriggerEvent] = []
        active_ids = set(self._sumo.person.getIDList())
        for ped_id, proxy_id in list(self._sumo_pedestrian_proxy.items()):
            pedestrian = self.pedestrians.get(ped_id)
            if pedestrian is None or proxy_id not in active_ids:
                continue
            x, y = self._sumo.person.getPosition(proxy_id)
            physical_pose = [float(x), float(y)]
            pedestrian.physical_pose_xy = physical_pose
            angle = float(self._sumo.person.getAngle(proxy_id))
            pedestrian.physical_yaw_rad = math.radians(90.0 - angle)
            pedestrian.physical_pose_authority = "sumo"
            pedestrian.is_spawned = True
            pedestrian.last_update_time = self._physics_time
            leg = self._sumo_pedestrian_leg.get(ped_id)
            if leg is None:
                continue
            lane_position = float(
                self._sumo.person.getLanePosition(proxy_id))
            start_position = float(leg["start_pos"])
            arrival_position = float(leg["arrival_pos"])
            leg_length = abs(arrival_position - start_position)
            progress = (
                1.0 if leg_length <= 1e-9 else
                abs(lane_position - start_position) / leg_length)
            progress = max(0.0, min(1.0, progress))
            start_progress = float(leg.get("start_progress", 0.0))
            route_progress = min(
                1.0,
                start_progress + (1.0 - start_progress) * progress)
            if leg["kind"] == "crosswalk":
                pedestrian.crossing_progress = route_progress
            else:
                pedestrian.walking_progress = route_progress

            if self._sumo.person.getRemainingStages(proxy_id) > 1:
                continue
            target_node = leg.get("target_node")
            if leg["kind"] == "crosswalk":
                pedestrian.complete_crossing(defer_arrival=True)
                event_type = "ped_crossing_complete"
                target_node = pedestrian.position.at_node or target_node
            else:
                pedestrian.arrive_at_node(
                    str(target_node), defer_arrival=True)
                event_type = "ped_arrive_node"
            # Semantic completion updates route indices and flags, while the
            # laterally resolved SUMO endpoint remains the physical truth.
            pedestrian.physical_pose_xy = physical_pose
            pedestrian.physical_pose_authority = "sumo"
            pedestrian.is_waiting = False
            self._sumo_pedestrian_leg.pop(ped_id, None)
            events.append(TriggerEvent(
                type=event_type,
                vehicle_id=ped_id,
                tick=round(self._physics_time),
                step=0,
                time_s=self._physics_time,
                details={"node_id": target_node},
            ))
            if not pedestrian.is_llm:
                # The next physics step installs the next mapped walking
                # stage. There is no VehicleArena behavioural callback.
                self._retire_sumo_pedestrian(ped_id)
        return events

    def _detect_signal_changes(self) -> List[TriggerEvent]:
        events: List[TriggerEvent] = []
        for vehicle_id, vehicle in self.vehicles.items():
            if (not vehicle.is_llm or vehicle.is_crashed or vehicle.arrived
                    or not vehicle.current_segment):
                continue
            movement = self.get_vehicle_movement_signal(
                vehicle_id, time_s=self._physics_time)
            if (movement is None
                    or not movement["controls_current_approach"]
                    or movement["distance_to_stop_line_m"] < -0.25):
                self._prev_light_signal.pop(vehicle_id, None)
                continue
            visible_range = self.perception_model.visual_envelope(
                vehicle_id, bearing_deg=0.0)["effective_range_m"]
            if movement["distance_to_stop_line_m"] > visible_range:
                self._prev_light_signal.pop(vehicle_id, None)
                continue
            current = (movement["connector_id"], movement["signal"])
            previous = self._prev_light_signal.get(vehicle_id)
            if (previous is not None and previous[0] == current[0]
                    and previous[1] != current[1]):
                events.append(TriggerEvent(
                    type="traffic_light_change",
                    vehicle_id=vehicle_id,
                    tick=round(self._physics_time),
                    step=0,
                    time_s=self._physics_time,
                    details={
                        "node": movement["node_id"],
                        "connector_id": movement["connector_id"],
                        "turn": movement["turn"],
                        "prev_signal": previous[1],
                        "signal": current[1],
                        "remaining_seconds": movement["remaining_seconds"],
                        "distance_to_stop_line_m": movement[
                            "distance_to_stop_line_m"],
                        "time_s": round(self._physics_time),
                    },
                ))
            self._prev_light_signal[vehicle_id] = current
        return events

    def advance_world_to(self, target_time_s: float) -> List[TriggerEvent]:
        """Advance one frozen batch through SUMO and publish one state commit."""
        events: List[TriggerEvent] = []
        if not self._initial_state_bootstrapped:
            self._submit_controls()
            self._bootstrap_initial_sumo_state()
            self._publish_bootstrap_poses()
            # Contacts created by authored t=0 placements are reported by the
            # private commit step.  Commit them at public time zero before a
            # later SUMO step can clear the collision result.
            self._commit_sumo_collisions(0.0)
            # The first driver wake may select a maneuver while public time
            # is frozen. Unresolved route ends never install a braking policy.
        while self._physics_time < target_time_s - 1e-6:
            interval_start = self._physics_time
            step_target = min(
                interval_start + self.sumo_config.step_length_s,
                target_time_s,
            )
            self._submit_controls(
                pedestrian_depart_before_s=step_target)
            self._sync_signal_states(interval_start)
            self._sync_lane_constraints(interval_start)
            public_dt_s = step_target - interval_start
            self._sumo.simulationStep(self._sumo_time_s + public_dt_s)
            self._sumo_time_s = float(self._sumo.simulation.getTime())
            self._physics_time = round(step_target, 6)
            events.extend(self._sync_from_sumo())
            events.extend(self._sync_pedestrians_from_sumo())
            self._commit_sumo_collisions(step_target)
            self._observe_lane_change_outcomes()
            events.extend(self._finalize_route_failures(step_target))
            events.extend(
                self.collect_pedestrian_lifecycle_events(step_target))
            if self._pending_events:
                events.extend(self._pending_events)
                self._pending_events = []
            events.extend(self._finalize_pending_arrivals(step_target))
            events.extend(self._detect_signal_changes())
            self._rebuild_seg_index()
            for vehicle in self.vehicles.values():
                self._update_road_network_position(vehicle)
        return events

    def _finalize_route_failures(self, time_s: float) -> List[TriggerEvent]:
        """Record native route removal, preserving the last measured motion.

        SUMO cannot simulate beyond the installed route. This is a per-car
        task failure, not evidence of a wall, impact, or physical stop.
        """
        events = []
        for vehicle in self.vehicles.values():
            if not vehicle.pending_route_failure:
                continue
            vehicle.pending_route_failure = False
            vehicle.present_in_physics_world = False
            vehicle.terminal_crossing_speed_kmh = vehicle.current_speed_kmh
            if vehicle.is_crashed:
                continue  # An actual contact at this boundary takes priority.
            vehicle.route_failed = True
            vehicle.route_failure_reason = "unresolved_route_endpoint"
            vehicle.route_failure_time_s = time_s
            vehicle.is_navigating = False
            events.append(TriggerEvent(
                type="route_failed", vehicle_id=vehicle.vehicle_id,
                tick=round(time_s), step=0, time_s=time_s,
                details={
                    "reason": vehicle.route_failure_reason,
                    "last_observed_lane_id": vehicle.current_lane_id,
                    "installed_route": list(self._sumo_installed_routes.get(
                        vehicle.vehicle_id, [])),
                    "last_observed_speed_kmh": vehicle.current_speed_kmh,
                    "present_in_physics_world": False,
                    "physically_stopped": False,
                }))
        return events
