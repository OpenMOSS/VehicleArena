"""Generic PNG/GIF renderer for any VehicleArena simulation interval.

The recorder reads the exact poses used by lane-level collision detection.
Recorded frames are self-contained and can be saved/reloaded as JSON without
re-running the simulation.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from simulation.lane_level_runtime import VehicleFootprint

Point = Tuple[float, float]

STYLE_COLORS = {
    "cautious": (66, 165, 245),
    "normal": (42, 205, 137),
    "defensive": (174, 125, 245),
    "aggressive": (252, 151, 55),
    "rule_breaker": (252, 69, 83),
}
SIGNAL_COLORS = {
    "green": (42, 211, 119),
    "yellow": (252, 202, 60),
    "red": (128, 48, 55),
}


@dataclass(frozen=True)
class Viewport:
    """Fixed metric world rectangle shared by all frames."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @classmethod
    def full_map(cls, lane_map: dict, padding_m: float = 20.0) -> "Viewport":
        points = [
            point for lane in lane_map["lanes"]
            for point in lane["centerline_xy"]]
        return cls(
            min(p[0] for p in points) - padding_m,
            min(p[1] for p in points) - padding_m,
            max(p[0] for p in points) + padding_m,
            max(p[1] for p in points) + padding_m,
        )

    @classmethod
    def around(
        cls, center_xy: Sequence[float], radius_m: float,
    ) -> "Viewport":
        return cls(
            center_xy[0] - radius_m, center_xy[1] - radius_m,
            center_xy[0] + radius_m, center_xy[1] + radius_m)

    @classmethod
    def around_node(
        cls, lane_map: dict, node_id: str, radius_m: float,
    ) -> "Viewport":
        nodes = lane_map["nodes_xy"]
        node_lookup = (
            {item["id"]: item["xy"] for item in nodes}
            if isinstance(nodes, list) else nodes)
        if node_id not in node_lookup:
            intersection = next(
                (item for item in lane_map.get("intersections", [])
                 if item["id"] == node_id
                 or node_id in item.get("member_nodes", [])), None)
            if not intersection:
                raise ValueError(f"Unknown lane-map node: {node_id}")
            center = intersection["center_xy"]
        else:
            center = node_lookup[node_id]
        return cls.around(center, radius_m)

    def intersects_points(self, points: Sequence[Sequence[float]]) -> bool:
        if any(
                self.min_x <= p[0] <= self.max_x
                and self.min_y <= p[1] <= self.max_y
                for p in points):
            return True
        # A long lane segment can cross a close-up viewport while both stored
        # vertices lie outside it. Point-only culling made the road disappear
        # in otherwise valid arbitrary-interval close-ups.
        for first, second in zip(points, points[1:]):
            if (max(first[0], second[0]) < self.min_x
                    or min(first[0], second[0]) > self.max_x
                    or max(first[1], second[1]) < self.min_y
                    or min(first[1], second[1]) > self.max_y):
                continue
            return True
        return False


@dataclass
class WorldFrame:
    time_s: float
    vehicles: List[dict] = field(default_factory=list)
    pedestrians: List[dict] = field(default_factory=list)
    connector_signals: Dict[str, str] = field(default_factory=dict)
    collisions: List[dict] = field(default_factory=list)
    horn_events: List[dict] = field(default_factory=list)


@dataclass
class WorldRecording:
    network_id: str
    viewport: Viewport
    frame_interval_s: float
    frames: List[WorldFrame] = field(default_factory=list)

    def save_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(asdict(self), stream, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_json(cls, path: str) -> "WorldRecording":
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        return cls(
            network_id=data["network_id"],
            viewport=Viewport(**data["viewport"]),
            frame_interval_s=data["frame_interval_s"],
            frames=[WorldFrame(**frame) for frame in data["frames"]],
        )


class LaneWorldRenderer:
    """Capture, simulate, serialise and render a lane-level world."""

    def __init__(
        self, traffic_manager: Any, viewport: Optional[Viewport] = None,
        width: int = 960, height: int = 960,
        vehicle_marker_radius_px: int = 0,
        title: str = "",
    ):
        self.manager = traffic_manager
        self.network = traffic_manager.road_network
        self.runtime = traffic_manager._lane_geometry
        self.map = self.runtime.data
        self.width = int(width)
        self.height = int(height)
        self.vehicle_marker_radius_px = max(
            0, int(vehicle_marker_radius_px))
        self.title = str(title)
        self.viewport = viewport or Viewport.full_map(self.map)
        self._prepare_geometry()
        self._font = ImageFont.load_default(size=12)
        self._small_font = ImageFont.load_default(size=9)

    def set_viewport(self, viewport: Viewport) -> None:
        """Change view and rebuild spatially culled drawing geometry."""
        self.viewport = viewport
        self._prepare_geometry()

    def _prepare_geometry(self) -> None:
        self._lanes = [
            lane for lane in self.map["lanes"]
            if self.viewport.intersects_points(lane["centerline_xy"])]
        self._lanes.sort(key=lambda lane: lane.get("z_level", 0))
        self._connectors = [
            connector for connector in self.map["connectors"]
            if self.viewport.intersects_points(connector["centerline_xy"])]
        self._connector_by_id = {
            item["id"]: item for item in self.map["connectors"]}
        self._connectors_by_source: Dict[str, List[str]] = {}
        for connector in self.map["connectors"]:
            self._connectors_by_source.setdefault(
                connector["from_lane"], []).append(connector["id"])
        self._stops = [
            stop for stop in self.map.get("stop_lines", [])
            if self.viewport.intersects_points(stop["line_xy"])]
        self._crosswalks = [
            item for item in self.map.get("crosswalks", [])
            if self.viewport.intersects_points(
                item.get("polygon_xy", [item["center_xy"]]))]

    def capture(self, time_s: Optional[float] = None) -> WorldFrame:
        """Capture the current physical state without advancing simulation."""
        if time_s is None:
            time_s = self.manager._physics_time
        vehicles = []
        for vehicle in self.manager.vehicles.values():
            if vehicle.arrived or vehicle.route_failed:
                continue
            pose = self.runtime.vehicle_pose(vehicle)
            if not pose:
                continue
            corners = VehicleFootprint(
                vehicle.length_m, vehicle.width_m).corners(pose[:3])
            vehicles.append({
                "id": vehicle.vehicle_id,
                "style": "llm" if vehicle.is_llm else "sumo_background",
                "pose": [round(value, 6) for value in pose[:3]],
                "z_level": pose[3],
                "corners": [
                    [round(point[0], 6), round(point[1], 6)]
                    for point in corners],
                "speed_kmh": round(vehicle.current_speed_kmh, 3),
                "acceleration_mps2": round(vehicle.acceleration_mps2, 3),
                "lane_id": vehicle.current_lane_id,
                "connector_id": vehicle.active_connector_id,
                "is_changing_lane": vehicle.is_changing_lane,
                "crashed": vehicle.is_crashed,
                "length_m": vehicle.length_m,
                "width_m": vehicle.width_m,
                "signal_state": vehicle.signal_state.as_dict(),
            })
        pedestrians = []
        for pedestrian in self.manager.pedestrians.values():
            if pedestrian.has_arrived:
                continue
            point = self.runtime.pedestrian_pose(pedestrian)
            if point:
                pedestrians.append({
                    "id": pedestrian.ped_id,
                    "center": [round(point[0], 6), round(point[1], 6)],
                    "radius_m": pedestrian.collision_radius_m,
                    "control_authority": pedestrian.control_authority,
                    "crossing": pedestrian.is_on_crosswalk,
                })
        signals = {}
        for connector_id in self.runtime._signal_plan_by_connector:
            state = self.runtime.signal_state(connector_id, time_s)
            if state:
                signals[connector_id] = state.signal
        return WorldFrame(
            time_s=round(time_s, 6),
            vehicles=vehicles,
            pedestrians=pedestrians,
            connector_signals=signals,
            collisions=[
                asdict(event) for event in self.manager.collision_log
                if event.time_s <= time_s + 1e-6],
            horn_events=[{
                "event_id": event.event_id,
                "source_id": event.source_id,
                "center": [event.pose_x_m, event.pose_y_m],
                "intensity": event.intensity,
                "start_time_s": event.start_time_s,
                "end_time_s": event.end_time_s,
            } for event in self.manager.horn_events
                if event.start_time_s - 1e-6 <= time_s
                <= event.end_time_s + 1e-6],
        )

    def simulate_and_record(
        self, start_time_s: float, end_time_s: float,
        frame_interval_s: float = 0.2,
    ) -> WorldRecording:
        """Advance at 0.1 s control/physics cadence and record any interval."""
        if end_time_s < start_time_s:
            raise ValueError("end_time_s must be >= start_time_s")
        if frame_interval_s < 0.1 - 1e-9:
            raise ValueError("frame_interval_s must be at least 0.1 seconds")
        if self.manager._physics_time > start_time_s + 1e-6:
            raise ValueError(
                "Cannot record past state; load a saved recording or start "
                "from a fresh simulation")
        self._advance_to(start_time_s)
        frames = [self.capture(start_time_s)]
        frame_time = start_time_s + frame_interval_s
        while frame_time <= end_time_s + 1e-7:
            target = min(frame_time, end_time_s)
            self._advance_to(target)
            frames.append(self.capture(target))
            frame_time += frame_interval_s
        if frames[-1].time_s < end_time_s - 1e-6:
            self._advance_to(end_time_s)
            frames.append(self.capture(end_time_s))
        return WorldRecording(
            network_id=getattr(self.network, "network_id", ""),
            viewport=self.viewport,
            frame_interval_s=frame_interval_s,
            frames=frames,
        )

    def _advance_to(self, target_time_s: float) -> None:
        while self.manager._physics_time < target_time_s - 1e-7:
            now = self.manager._physics_time
            next_time = min(target_time_s, now + 0.1)
            self.manager.recalculate_speeds(now)
            self.manager.advance_world_to(next_time)

    def render_png(
        self, frame: WorldFrame, path: str,
        show_labels: bool = True,
    ) -> str:
        image = self.render_frame(frame, show_labels=show_labels)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        image.save(path)
        return path

    def render_gif(
        self, recording: WorldRecording, path: str,
        show_labels: bool = False, loop: int = 0,
    ) -> str:
        if not recording.frames:
            raise ValueError("recording contains no frames")
        images = [
            self.render_frame(frame, show_labels=show_labels)
            for frame in recording.frames]
        shared = images[0].convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=96)
        indexed = [shared] + [
            image.quantize(palette=shared, dither=Image.Dither.NONE)
            for image in images[1:]]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        indexed[0].save(
            path, save_all=True, append_images=indexed[1:],
            duration=max(20, round(recording.frame_interval_s * 1000)),
            loop=loop, optimize=False, disposal=1)
        return path

    def render_frame(
        self, frame: WorldFrame, show_labels: bool = True,
    ) -> Image.Image:
        image = Image.new("RGB", (self.width, self.height), (18, 28, 38))
        draw = ImageDraw.Draw(image)
        scale = self._scale

        for lane in self._lanes:
            draw.line(
                [self._screen(point) for point in lane["centerline_xy"]],
                fill=((48, 64, 77) if lane.get("z_level", 0)
                      else (59, 77, 91)),
                width=max(1, round(lane["width_m"] * scale)),
                joint="curve")
        # Road markings belong above the asphalt surface.
        for crosswalk in self._crosswalks:
            draw.polygon(
                [self._screen(point)
                 for point in crosswalk["polygon_xy"]],
                fill=(66, 69, 66), outline=(135, 139, 132))
            for stripe in crosswalk["stripe_polygons_xy"]:
                draw.polygon(
                    [self._screen(point) for point in stripe],
                    fill=(222, 222, 210))
        for lane in self._lanes:
            for key in ("left_boundary_xy", "right_boundary_xy"):
                draw.line(
                    [self._screen(point) for point in lane[key]],
                    fill=(126, 143, 156), width=1)

        for connector in self._connectors:
            signal = frame.connector_signals.get(connector["id"])
            color = SIGNAL_COLORS.get(signal, (88, 83, 72))
            if signal == "red":
                color = (82, 39, 45)
            draw.line(
                [self._screen(point)
                 for point in connector["centerline_xy"]],
                fill=color, width=(3 if signal in ("green", "yellow") else 1),
                joint="curve")
        for stop in self._stops:
            states = [
                frame.connector_signals.get(connector_id)
                for connector_id in self._connectors_by_source.get(
                    stop["lane_id"], [])]
            state = (
                "green" if "green" in states
                else "yellow" if "yellow" in states else "red")
            draw.line(
                [self._screen(point) for point in stop["line_xy"]],
                fill=SIGNAL_COLORS[state], width=4)

        for pedestrian in frame.pedestrians:
            center = self._screen(pedestrian["center"])
            radius = max(2, round(pedestrian["radius_m"] * scale))
            draw.ellipse(
                (center[0] - radius, center[1] - radius,
                 center[0] + radius, center[1] + radius),
                fill=(255, 214, 80), outline=(255, 255, 255))
        for event in frame.horn_events:
            center = self._screen(event["center"])
            elapsed = max(0.0, frame.time_s - event["start_time_s"])
            radius_m = 2.0 + elapsed * 18.0
            radius = max(3, round(radius_m * scale))
            color = (
                (255, 98, 78) if event["intensity"] == "urgent"
                else (255, 202, 72))
            draw.ellipse(
                (center[0] - radius, center[1] - radius,
                 center[0] + radius, center[1] + radius),
                outline=color, width=2)
        for vehicle in sorted(
                frame.vehicles, key=lambda item: item.get("z_level", 0)):
            corners = [self._screen(point) for point in vehicle["corners"]]
            color = (
                (245, 45, 45) if vehicle["crashed"]
                else STYLE_COLORS.get(vehicle["style"], (72, 175, 230)))
            if self.vehicle_marker_radius_px:
                center = self._screen(vehicle["pose"][:2])
                radius = self.vehicle_marker_radius_px
                draw.ellipse(
                    (center[0] - radius, center[1] - radius,
                     center[0] + radius, center[1] + radius),
                    outline=color, width=2)
            draw.polygon(corners, fill=color, outline=(250, 252, 253))
            draw.line(
                (corners[0], corners[1]),
                fill=(255, 238, 100), width=2)
            signals = vehicle.get("signal_state", {})
            pose_x, pose_y, yaw = vehicle["pose"]
            half_length = float(vehicle.get("length_m", 4.6)) / 2.0
            half_width = float(vehicle.get("width_m", 1.9)) / 2.0

            def body_point(longitudinal, lateral):
                return self._screen((
                    pose_x + math.cos(yaw) * longitudinal
                    - math.sin(yaw) * lateral,
                    pose_y + math.sin(yaw) * longitudinal
                    + math.cos(yaw) * lateral,
                ))

            blink_on = int(frame.time_s * 2.0) % 2 == 0
            left_on = bool(
                signals.get("hazard")
                or signals.get("left_indicator")) and blink_on
            right_on = bool(
                signals.get("hazard")
                or signals.get("right_indicator")) and blink_on
            for longitudinal in (-half_length, half_length):
                if left_on:
                    p = body_point(longitudinal, half_width)
                    draw.ellipse(
                        (p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2),
                        fill=(255, 183, 42))
                if right_on:
                    p = body_point(longitudinal, -half_width)
                    draw.ellipse(
                        (p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2),
                        fill=(255, 183, 42))
            rear_color = (
                (255, 30, 35) if signals.get("brake_light")
                else (130, 28, 32) if (
                    signals.get("tail_light")
                    or signals.get("position_light")) else None)
            if rear_color:
                for lateral in (-half_width * 0.72, half_width * 0.72):
                    p = body_point(-half_length, lateral)
                    draw.ellipse(
                        (p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2),
                        fill=rear_color)
            beam_color = None
            beam_length = 0.0
            if signals.get("high_beam"):
                beam_color, beam_length = (202, 235, 255), 12.0
            elif signals.get("low_beam"):
                beam_color, beam_length = (255, 244, 184), 7.0
            elif signals.get("front_fog_light"):
                beam_color, beam_length = (255, 220, 122), 4.0
            if beam_color:
                for lateral in (-half_width * 0.65, half_width * 0.65):
                    start = body_point(half_length, lateral)
                    end = body_point(half_length + beam_length, lateral)
                    draw.line((start, end), fill=beam_color, width=1)
            if show_labels:
                x, y = self._screen(vehicle["pose"][:2])
                draw.text(
                    (x + 4, y - 12),
                    f"{vehicle['id']} {vehicle['speed_kmh']:.0f}",
                    font=self._small_font, fill=(238, 244, 248))

        draw.rounded_rectangle(
            (8, 8, 185, 40), radius=6,
            fill=(7, 14, 20), outline=(70, 89, 102))
        draw.text(
            (16, 17), f"t = {frame.time_s:.1f} s",
            font=self._font, fill=(245, 249, 252))
        if self.title:
            title_width = draw.textlength(self.title, font=self._font)
            left = max(196, self.width - round(title_width) - 24)
            draw.rounded_rectangle(
                (left, 8, self.width - 8, 40), radius=6,
                fill=(7, 14, 20), outline=(70, 89, 102))
            draw.text(
                (left + 8, 17), self.title,
                font=self._font, fill=(245, 249, 252))
        return image

    @property
    def _scale(self) -> float:
        world_width = max(1e-6, self.viewport.max_x - self.viewport.min_x)
        world_height = max(1e-6, self.viewport.max_y - self.viewport.min_y)
        return min((self.width - 20) / world_width,
                   (self.height - 20) / world_height)

    def _screen(self, point: Sequence[float]) -> Tuple[int, int]:
        scale = self._scale
        world_width = self.viewport.max_x - self.viewport.min_x
        world_height = self.viewport.max_y - self.viewport.min_y
        used_width, used_height = world_width * scale, world_height * scale
        offset_x = (self.width - used_width) / 2
        offset_y = (self.height - used_height) / 2
        return (
            round(offset_x + (point[0] - self.viewport.min_x) * scale),
            round(self.height - offset_y
                  - (point[1] - self.viewport.min_y) * scale),
        )
