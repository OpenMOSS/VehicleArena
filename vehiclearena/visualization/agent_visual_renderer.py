"""Deterministic LiDAR BEV and navigation images for driving agents.

The renderer deliberately has two separate products:

* ``render_lidar_bev`` is an ego-centred processed LiDAR view. It preserves
  road and actor geometry but contains no traffic-light state, lamp effect,
  weather effect, IDs, distances, boxes, or route overlay.
* ``render_minimap`` is a navigation display.  It contains static lane
  geometry and the selected lane-level route, but no live traffic entities or
  signal state.

Both products are generated from the same authoritative lane geometry used by
physics and are deterministic for a fixed world state.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from simulation.lane_level_runtime import VehicleFootprint, _point_segment_projection


Point = Tuple[float, float]
_ROUTE_PREVIEW_UNSET = object()


@dataclass(frozen=True)
class RenderedAgentImage:
    """One PNG plus compact metadata safe to expose beside the image."""

    kind: str
    sim_time_s: float
    width: int
    height: int
    png_bytes: bytes
    view: str = ""
    route_available: Optional[bool] = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.png_bytes).hexdigest()

    @property
    def data_url(self) -> str:
        payload = base64.b64encode(self.png_bytes).decode("ascii")
        return f"data:image/png;base64,{payload}"

    def metadata(self) -> dict:
        result = {
            "kind": self.kind,
            "sim_time_s": round(float(self.sim_time_s), 6),
            "view": self.view,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
        }
        if self.route_available is not None:
            result["route_available"] = bool(self.route_available)
        return result


def multimodal_image_message(
    rendered: RenderedAgentImage,
    *, label: str,
) -> dict:
    """Build one OpenAI-compatible user image message."""
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"[{label}] This image is the authoritative "
                    f"{rendered.kind} observation at simulation time "
                    f"{rendered.sim_time_s:.3f}s. Infer spatial facts from "
                    "the image; no duplicate object list is provided."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": rendered.data_url,
                    "detail": "high",
                },
            },
        ],
    }


class AgentVisualRenderer:
    """Rule renderer over one TrafficCoordinator and its lane-level runtime."""

    _DAY_BACKGROUND = (151, 174, 185)
    # Keep roadside terrain chromatically neutral in the driving observation.
    # A green ground strip beside a narrow road can otherwise be mistaken for
    # a lane marking by a vision model.
    _DAY_GROUND = (132, 136, 137)
    _DAY_ROAD = (82, 91, 98)
    _DAY_CONNECTOR = (77, 86, 92)
    _DAY_BOUNDARY = (229, 233, 234)
    _DAY_CONNECTOR_BOUNDARY = (151, 160, 165)

    # Navigation and LiDAR images use fixed neutral display themes.
    _BACKGROUND = (10, 18, 25)
    _MINIMAP_ROAD = (38, 49, 58)
    _MINIMAP_EDGE = (70, 84, 94)
    _MINIMAP_LANE = (86, 101, 111)
    _MINIMAP_ROUTE_OUTLINE = (7, 27, 39)
    _ROUTE = (39, 199, 255)
    _MINIMAP_DESTINATION = (242, 251, 255)
    _MINIMAP_SCALE = (207, 222, 231)
    _EGO_VEHICLE = (30, 168, 226)
    _OTHER_VEHICLE = (235, 132, 52)
    _CRASHED_VEHICLE = (219, 78, 73)
    _VEHICLE_ARROW = (25, 54, 72)
    _CROSSWALK_SURFACE = (66, 70, 71)
    _CROSSWALK_STRIPE = (188, 184, 164)
    _STOP_LINE = (250, 252, 248)
    def __init__(
        self,
        traffic_manager: Any,
        *,
        width: int = 1024,
        height: int = 1024,
    ):
        self.manager = traffic_manager
        self.network = traffic_manager.road_network
        self.runtime = traffic_manager._lane_geometry
        self.map = self.runtime.data
        self.width = int(width)
        self.height = int(height)
        self._lanes = list(self.map.get("lanes", []))
        self._connectors = list(self.map.get("connectors", []))
        self._stops = list(self.map.get("stop_lines", []))
        self._crosswalks = list(self.map.get("crosswalks", []))
        self._connector_by_id = {
            item["id"]: item for item in self._connectors}
        self._lane_by_id = {
            item["id"]: item for item in self._lanes}
        # Display-only progress, isolated per vehicle and saved plan instance.
        # Never mutate navigation's saved route or ask the planner to advance it.
        self._minimap_progress: Dict[str, tuple] = {}
        self._lane_groups: Dict[Tuple[str, str, int], List[dict]] = {}
        for lane in self._lanes:
            key = (
                str(lane.get("segment_id", "")),
                str(lane.get("direction", "")),
                int(lane.get("z_level", 0)),
            )
            self._lane_groups.setdefault(key, []).append(lane)
        for lanes in self._lane_groups.values():
            lanes.sort(key=lambda item: int(item.get(
                "directional_index", item.get("index", 0))))
        self._stop_lane_ids = {
            str(item.get("lane_id", "")) for item in self._stops}
        self._connectors_by_source: Dict[str, List[str]] = {}
        self._connectors_by_target: Dict[str, List[str]] = {}
        junction_connector_groups: Dict[Tuple[str, int], List[dict]] = {}
        for connector in self._connectors:
            self._connectors_by_source.setdefault(
                connector["from_lane"], []).append(connector["id"])
            self._connectors_by_target.setdefault(
                connector["to_lane"], []).append(connector["id"])
            junction_connector_groups.setdefault((
                str(connector.get("node_id", "")),
                int(connector.get("z_level", 0)),
            ), []).append(connector)
        self._junction_surfaces = []
        for (node_id, z_level), connectors in (
                junction_connector_groups.items()):
            # Connector ribbons describe individual legal movements, not the
            # complete paved junction. Build an envelope from their physical
            # mouth endpoints so the spaces between movements are not shown
            # as roadside terrain.
            mouth_points = []
            for connector in connectors:
                for key in ("left_boundary_xy", "right_boundary_xy"):
                    boundary = connector.get(key, [])
                    if boundary:
                        mouth_points.extend((boundary[0], boundary[-1]))
            polygon = self._convex_hull(mouth_points)
            if len(polygon) >= 3:
                self._junction_surfaces.append({
                    "node_id": node_id,
                    "z_level": z_level,
                    "polygon_xy": polygon,
                })
        self._connectors_by_node: Dict[str, List[dict]] = {}
        for connector in self._connectors:
            self._connectors_by_node.setdefault(
                str(connector.get("node_id", "")), []).append(connector)

    @staticmethod
    def _png(image: Image.Image) -> bytes:
        stream = io.BytesIO()
        image.save(stream, format="PNG", optimize=True)
        return stream.getvalue()

    @staticmethod
    def _mix_color(
        dark: Tuple[int, int, int],
        light: Tuple[int, int, int],
        amount: float,
    ) -> Tuple[int, int, int]:
        amount = max(0.0, min(1.0, float(amount)))
        return tuple(round(
            dark[index] + (light[index] - dark[index]) * amount)
            for index in range(3))

    def _lidar_palette(self) -> Dict[str, Tuple[int, int, int]]:
        """Return the fixed surface colours used by the LiDAR display."""
        return {
            "background": self._DAY_BACKGROUND,
            "ground": self._DAY_GROUND,
            "road": self._DAY_ROAD,
            "connector": self._DAY_CONNECTOR,
            "boundary": self._DAY_BOUNDARY,
            "connector_boundary": self._DAY_CONNECTOR_BOUNDARY,
        }

    @staticmethod
    def _local_transform(ego_pose: Sequence[float]):
        ex, ey, yaw = map(float, ego_pose[:3])

        def transform(point: Sequence[float]) -> Point:
            dx, dy = float(point[0]) - ex, float(point[1]) - ey
            return (
                # Image X grows to the driver's right.  The previous formula
                # returned left-positive lateral distance and then added it
                # to screen X, mirroring every left/right movement.
                math.sin(yaw) * dx - math.cos(yaw) * dy,
                math.cos(yaw) * dx + math.sin(yaw) * dy,
            )

        return transform

    @staticmethod
    def _intersects_local(
        points: Sequence[Sequence[float]], transform,
        *, front_m: float, rear_m: float, half_width_m: float,
        margin_m: float = 8.0,
    ) -> bool:
        if not points:
            return False
        values = [transform(point) for point in points]
        return not (
            max(point[1] for point in values) < -rear_m - margin_m
            or min(point[1] for point in values) > front_m + margin_m
            or max(point[0] for point in values) < -half_width_m - margin_m
            or min(point[0] for point in values) > half_width_m + margin_m
        )

    @staticmethod
    def _convex_hull(points: Sequence[Sequence[float]]) -> List[Point]:
        """Return a deterministic paved envelope for one junction."""
        values = sorted({
            (float(point[0]), float(point[1]))
            for point in points if len(point) >= 2
        })
        if len(values) <= 1:
            return values

        def cross(origin: Point, first: Point, second: Point) -> float:
            return ((first[0] - origin[0]) * (second[1] - origin[1])
                    - (first[1] - origin[1]) * (second[0] - origin[0]))

        lower: List[Point] = []
        for point in values:
            while len(lower) >= 2 and cross(
                    lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper: List[Point] = []
        for point in reversed(values):
            while len(upper) >= 2 and cross(
                    upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    def _relevant_approach_lane_ids(self, ego: Any) -> set[str]:
        """Return the current lane and its immediately reachable neighbours.

        The local LiDAR image is not a topology debugger. Drawing movement
        arrows for every lane visible through a wide junction produces visual
        noise and exposes unrelated approaches. The current lane plus one lane
        on either side contains every lane ego can physically occupy next.
        """
        anchor_id = str(getattr(ego, "current_lane_id", "") or "")
        if getattr(ego, "active_connector_id", ""):
            target_id = str(getattr(
                ego, "active_connector_to_lane_id", "") or "")
            if target_id in self._lane_by_id:
                anchor_id = target_id
        anchor = self._lane_by_id.get(anchor_id)
        if anchor is None:
            return set(self._lane_by_id)
        key = (
            str(anchor.get("segment_id", "")),
            str(anchor.get("direction", "")),
            int(anchor.get("z_level", 0)),
        )
        group = self._lane_groups.get(key, [anchor])
        position = next(
            (index for index, lane in enumerate(group)
             if lane.get("id") == anchor_id),
            0,
        )
        return {
            str(group[index]["id"])
            for index in range(
                max(0, position - 1), min(len(group), position + 2))
        }

    def _relevant_road_paint(self, ego: Any) -> Tuple[set[str], set[str]]:
        """Select paint for ego's road and its next physical junction.

        Nearby but topologically unrelated OSM segments remain visible as
        road surfaces, while their markings no longer project across the
        driver's corridor.  Every approach at the next junction is included,
        so cross traffic is still spatially legible.
        """
        approach_ids = self._relevant_approach_lane_ids(ego)
        lane_ids = set(approach_ids)
        node_ids = set()
        for lane_id in approach_ids:
            for connector_id in self._connectors_by_source.get(lane_id, []):
                connector = self._connector_by_id.get(connector_id)
                if connector is not None:
                    node_ids.add(str(connector.get("node_id", "")))
        active_id = str(getattr(ego, "active_connector_id", "") or "")
        active = self._connector_by_id.get(active_id)
        if active is not None:
            node_ids.add(str(active.get("node_id", "")))
        for node_id in node_ids:
            for connector in self._connectors_by_node.get(node_id, []):
                for connected_lane_id in (
                        str(connector.get("from_lane", "")),
                        str(connector.get("to_lane", ""))):
                    lane = self._lane_by_id.get(connected_lane_id)
                    if lane is None:
                        continue
                    group_key = (
                        str(lane.get("segment_id", "")),
                        str(lane.get("direction", "")),
                        int(lane.get("z_level", 0)),
                    )
                    lane_ids.update(
                        str(item["id"])
                        for item in self._lane_groups.get(group_key, [lane]))
        return lane_ids, node_ids

    @staticmethod
    def _vehicle_corners(vehicle: Any, pose: Sequence[float]) -> List[Point]:
        return VehicleFootprint(
            float(vehicle.length_m), float(vehicle.width_m),
        ).corners(tuple(map(float, pose[:3])))

    @staticmethod
    def _draw_direction_arrow(
        draw: ImageDraw.ImageDraw,
        corners: Sequence[Sequence[float]],
        *, fill: Tuple[int, int, int],
    ) -> None:
        """Draw a compact arrow from the rear toward the physical front."""
        if len(corners) != 4:
            return
        front = (
            (float(corners[0][0]) + float(corners[1][0])) / 2.0,
            (float(corners[0][1]) + float(corners[1][1])) / 2.0,
        )
        rear = (
            (float(corners[2][0]) + float(corners[3][0])) / 2.0,
            (float(corners[2][1]) + float(corners[3][1])) / 2.0,
        )
        dx, dy = front[0] - rear[0], front[1] - rear[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        center = (
            sum(float(point[0]) for point in corners) / 4.0,
            sum(float(point[1]) for point in corners) / 4.0,
        )
        arrow_length = max(6.0, min(14.0, length * 0.78))
        head_length = max(3.0, arrow_length * 0.38)
        half_width = max(2.5, min(5.0, arrow_length * 0.27))
        tail = (
            center[0] - ux * arrow_length * 0.45,
            center[1] - uy * arrow_length * 0.45,
        )
        tip = (
            center[0] + ux * arrow_length * 0.55,
            center[1] + uy * arrow_length * 0.55,
        )
        head_base = (
            tip[0] - ux * head_length,
            tip[1] - uy * head_length,
        )
        draw.line((tail, head_base), fill=fill, width=2)
        draw.polygon([
            tip,
            (head_base[0] + px * half_width,
             head_base[1] + py * half_width),
            (head_base[0] - px * half_width,
             head_base[1] - py * half_width),
        ], fill=fill)

    @staticmethod
    def _trim_polyline_end(
        points: Sequence[Sequence[float]], trim_m: float,
    ) -> List[Point]:
        """Return a polyline shortened by ``trim_m`` at its travel end."""
        result = [tuple(map(float, point[:2])) for point in points]
        remaining = max(0.0, float(trim_m))
        while len(result) >= 2 and remaining > 1e-9:
            first, second = result[-2], result[-1]
            length = math.hypot(second[0] - first[0], second[1] - first[1])
            if length <= remaining + 1e-9:
                remaining -= length
                result.pop()
                continue
            keep = (length - remaining) / length
            result[-1] = (
                first[0] + (second[0] - first[0]) * keep,
                first[1] + (second[1] - first[1]) * keep,
            )
            remaining = 0.0
        return result if len(result) >= 2 else []

    @classmethod
    def _trim_polyline_start(
        cls, points: Sequence[Sequence[float]], trim_m: float,
    ) -> List[Point]:
        """Return a polyline shortened by ``trim_m`` at its travel start."""
        trimmed = cls._trim_polyline_end(list(reversed(points)), trim_m)
        return list(reversed(trimmed))

    def _trim_lane_paint(
        self, lane_id: str, points: Sequence[Sequence[float]],
        *, stop_line_clearance: bool = False,
    ) -> List[Point]:
        """Keep lane paint out of the paved connector mouth."""
        result = [tuple(map(float, point[:2])) for point in points]
        if self._connectors_by_target.get(lane_id):
            result = self._trim_polyline_start(result, 1.0)
        if self._connectors_by_source.get(lane_id):
            end_trim_m = (
                2.0 if stop_line_clearance
                and lane_id in self._stop_lane_ids else 1.0)
            result = self._trim_polyline_end(result, end_trim_m)
        return result

    @staticmethod
    def _draw_dashed_polyline(
        draw: ImageDraw.ImageDraw,
        points: Sequence[Sequence[float]],
        *,
        fill: Tuple[int, int, int],
        width: int,
        dash_px: float,
        gap_px: float,
    ) -> None:
        """Draw a dash pattern continuously across a screen polyline."""
        if len(points) < 2:
            return
        pattern = max(1.0, float(dash_px)) + max(1.0, float(gap_px))
        walked = 0.0
        for raw_first, raw_second in zip(points, points[1:]):
            first = (float(raw_first[0]), float(raw_first[1]))
            second = (float(raw_second[0]), float(raw_second[1]))
            dx, dy = second[0] - first[0], second[1] - first[1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            local = 0.0
            while local < length - 1e-9:
                phase = (walked + local) % pattern
                drawing = phase < dash_px
                run = min(
                    (dash_px - phase) if drawing else (pattern - phase),
                    length - local,
                )
                if drawing and run > 1e-9:
                    start_ratio = local / length
                    end_ratio = (local + run) / length
                    draw.line((
                        (first[0] + dx * start_ratio,
                         first[1] + dy * start_ratio),
                        (first[0] + dx * end_ratio,
                         first[1] + dy * end_ratio),
                    ), fill=fill, width=width)
                local += max(run, 1e-6)
            walked += length

    @staticmethod
    def _point_and_tangent_from_end(
        points: Sequence[Sequence[float]], distance_m: float,
    ) -> Optional[Tuple[Point, Point]]:
        """Sample a point and forward tangent a distance before lane end."""
        if len(points) < 2:
            return None
        segments = []
        total = 0.0
        for first, second in zip(points, points[1:]):
            length = math.hypot(
                float(second[0]) - float(first[0]),
                float(second[1]) - float(first[1]),
            )
            segments.append(length)
            total += length
        target = max(0.0, total - max(0.0, float(distance_m)))
        walked = 0.0
        for index, ((first, second), length) in enumerate(zip(
                zip(points, points[1:]), segments)):
            if walked + length >= target or index == len(segments) - 1:
                if length < 1e-9:
                    continue
                ratio = max(0.0, min(1.0, (target - walked) / length))
                dx = float(second[0]) - float(first[0])
                dy = float(second[1]) - float(first[1])
                return (
                    (float(first[0]) + dx * ratio,
                     float(first[1]) + dy * ratio),
                    (dx / length, dy / length),
                )
            walked += length
        return None

    @staticmethod
    def _draw_movement_arrow(
        draw: ImageDraw.ImageDraw,
        center: Sequence[float],
        travel_direction_px: Sequence[float],
        turn: str,
        *,
        length_px: float,
        lateral_offset_px: float = 0.0,
        fill: Tuple[int, int, int] = (238, 239, 226),
    ) -> None:
        """Draw one compact painted left/straight/right pavement arrow."""
        ux, uy = map(float, travel_direction_px[:2])
        magnitude = math.hypot(ux, uy)
        if magnitude < 1e-9:
            return
        ux, uy = ux / magnitude, uy / magnitude
        # Screen Y points down, so this is physical left of travel.
        lx, ly = uy, -ux
        cx = float(center[0]) + lx * lateral_offset_px
        cy = float(center[1]) + ly * lateral_offset_px
        arrow_length = max(10.0, float(length_px))
        start = (cx - ux * arrow_length * 0.52,
                 cy - uy * arrow_length * 0.52)
        bend = (cx + ux * arrow_length * 0.08,
                cy + uy * arrow_length * 0.08)
        direction = (ux, uy)
        if turn == "left":
            direction = (lx, ly)
        elif turn == "right":
            direction = (-lx, -ly)
        tip = (
            bend[0] + direction[0] * arrow_length * 0.42,
            bend[1] + direction[1] * arrow_length * 0.42,
        )
        width = max(2, round(arrow_length * 0.11))
        path = [start, bend, tip]
        draw.line(path, fill=fill, width=width, joint="curve")
        head_length = max(3.5, arrow_length * 0.24)
        head_half_width = max(2.8, arrow_length * 0.16)
        dx, dy = direction
        px, py = -dy, dx
        base = (tip[0] - dx * head_length, tip[1] - dy * head_length)
        draw.polygon([
            tip,
            (base[0] + px * head_half_width,
             base[1] + py * head_half_width),
            (base[0] - px * head_half_width,
             base[1] - py * head_half_width),
        ], fill=fill)

    def render_lidar_bev(
        self,
        vehicle_id: str,
        sim_time_s: float,
        sensor_overrides: Optional[dict] = None,
    ) -> RenderedAgentImage:
        """Render a processed LiDAR BEV without optical/semantic hints."""
        ego = self.manager.vehicles.get(vehicle_id)
        if ego is None:
            raise KeyError(f"unknown vehicle: {vehicle_id}")
        ego_pose = self.runtime.vehicle_pose(ego)
        if ego_pose is None:
            ego_pose = (ego.pose_x_m, ego.pose_y_m, ego.yaw_rad, ego.z_level)

        from module.lidar import resolve_lidar_spec
        lidar_spec = resolve_lidar_spec("lidar", sensor_overrides)
        # Keep the automatic urban-road LiDAR frame fixed.
        base_front_m = float(lidar_spec.range_m)
        rear_m = 8.0
        # Six ordinary urban lanes fit within this local crop. The former
        # 68 m-wide slab rendered whole opposing carriageways and unrelated
        # junction approaches as if they were dashboard information.
        half_width_m = min(
            base_front_m,
            base_front_m * math.tan(math.radians(
                min(178.0, float(lidar_spec.horizontal_fov_deg)) / 2.0)))
        # Preserve the readable original local crop instead of zooming out to
        # the complete theoretical sensor cone.
        half_width_m = min(18.0, max(8.0, half_width_m))
        transform = self._local_transform(ego_pose)
        relevant_lane_ids = self._relevant_approach_lane_ids(ego)
        road_paint_lane_ids, relevant_junction_node_ids = (
            self._relevant_road_paint(ego))
        relevant_connector_ids = {
            connector_id
            for lane_id in relevant_lane_ids
            for connector_id in self._connectors_by_source.get(lane_id, [])
        }
        active_connector_id = str(getattr(
            ego, "active_connector_id", "") or "")
        if active_connector_id:
            relevant_connector_ids.add(active_connector_id)
        scale = min(
            (self.width - 24) / (2.0 * half_width_m),
            (self.height - 24) / (base_front_m + rear_m),
        )
        ego_px = (
            self.width / 2.0,
            self.height - 12.0 - rear_m * scale,
        )

        def screen(point: Sequence[float]) -> Tuple[int, int]:
            lateral, longitudinal = transform(point)
            return (
                round(ego_px[0] + lateral * scale),
                round(ego_px[1] - longitudinal * scale),
            )

        def visible(points: Sequence[Sequence[float]]) -> bool:
            return self._intersects_local(
                points, transform, front_m=base_front_m,
                rear_m=rear_m, half_width_m=half_width_m)

        palette = self._lidar_palette()
        # Fixed display colours: optical lighting and weather never enter this
        # processed LiDAR view.
        scene = Image.new(
            "RGB", (self.width, self.height), palette["ground"])
        draw = ImageDraw.Draw(scene)
        # Paint each directional carriageway as one polygon.  Painting every
        # lane as an independently rounded thick centerline left one-pixel
        # seams between adjacent lanes, exposing the green ground underneath
        # as fake longitudinal markings.
        for lanes in self._lane_groups.values():
            if not any(
                    lane["id"] in road_paint_lane_ids for lane in lanes):
                continue
            right_boundary = lanes[0].get("right_boundary_xy", [])
            left_boundary = lanes[-1].get("left_boundary_xy", [])
            carriageway = list(left_boundary) + list(reversed(right_boundary))
            if len(carriageway) >= 3 and visible(carriageway):
                draw.polygon(
                    [screen(point) for point in carriageway],
                    fill=palette["road"],
                )
                continue
            # Retain a safe fallback for imported maps without lane bounds.
            for lane in lanes:
                points = lane.get("centerline_xy", [])
                if visible(points):
                    draw.line(
                        [screen(point) for point in points],
                        fill=palette["road"],
                        width=max(2, round(
                            float(lane.get("width_m", 3.5)) * scale) + 2),
                        joint="curve",
                    )
        # Fill the complete physical junction before painting individual
        # connector ribbons. Without this layer, valid space between turning
        # trajectories appeared as a large gray hole.
        for junction in self._junction_surfaces:
            if junction["node_id"] not in relevant_junction_node_ids:
                continue
            polygon = junction["polygon_xy"]
            if visible(polygon):
                draw.polygon(
                    [screen(point) for point in polygon],
                    fill=palette["connector"],
                )
        for connector in self._connectors:
            if str(connector.get("node_id", "")) not in (
                    relevant_junction_node_ids):
                continue
            points = connector["centerline_xy"]
            if visible(points):
                draw.line(
                    [screen(point) for point in points],
                    fill=palette["connector"],
                    width=max(2, round(
                        float(connector.get("width_m", 3.5)) * scale) + 2),
                    joint="curve")
        # Keep topology-only connector centerlines hidden.  Instead, draw the
        # two lane-extension boundaries as short white junction guide dashes
        # for connectors reachable from ego's local approach.  This restores
        # lane continuity without turning every possible movement in a busy
        # intersection into a dense centerline mesh.
        junction_guide_color = self._mix_color(
            palette["connector"], palette["boundary"], 0.88)
        for connector in self._connectors:
            if connector["id"] not in relevant_connector_ids:
                continue
            for key in ("left_boundary_xy", "right_boundary_xy"):
                boundary = connector.get(key, [])
                if not boundary or not visible(boundary):
                    continue
                self._draw_dashed_polyline(
                    draw,
                    [screen(point) for point in boundary],
                    fill=junction_guide_color,
                    width=max(2, round(0.14 * scale)),
                    dash_px=max(7, round(1.2 * scale)),
                    gap_px=max(8, round(1.4 * scale)),
                )
        # A checkered transverse cap represents a traversable mapped-world
        # exit boundary, not a barrier or another road user.
        # It prevents a short terminal road fragment from being mistaken for
        # another vehicle while avoiding any fabricated drivable extension.
        for lane in self._lanes:
            points = lane.get("centerline_xy", [])
            if (len(points) < 2 or not visible(points)
                    or lane["id"] not in road_paint_lane_ids
                    or self._connectors_by_source.get(lane["id"])):
                continue
            previous, endpoint = points[-2], points[-1]
            dx = float(endpoint[0]) - float(previous[0])
            dy = float(endpoint[1]) - float(previous[1])
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            nx, ny = -dy / length, dx / length
            half_width = float(lane.get("width_m", 3.5)) / 2.0
            first = (
                float(endpoint[0]) - nx * half_width,
                float(endpoint[1]) - ny * half_width,
            )
            second = (
                float(endpoint[0]) + nx * half_width,
                float(endpoint[1]) + ny * half_width,
            )
            for index in range(6):
                start_ratio = index / 6.0
                end_ratio = (index + 1) / 6.0
                start = (
                    first[0] + (second[0] - first[0]) * start_ratio,
                    first[1] + (second[1] - first[1]) * start_ratio,
                )
                end = (
                    first[0] + (second[0] - first[0]) * end_ratio,
                    first[1] + (second[1] - first[1]) * end_ratio,
                )
                draw.line(
                    (screen(start), screen(end)),
                    fill=(235, 245, 248) if index % 2 == 0
                    else (42, 185, 213),
                    width=max(2, round(0.35 * scale)),
                )
        for crosswalk in self._crosswalks:
            polygon = crosswalk.get("polygon_xy", [])
            if (str(crosswalk.get("node_id", ""))
                    not in relevant_junction_node_ids
                    or not polygon or not visible(polygon)):
                continue
            draw.polygon(
                [screen(point) for point in polygon],
                fill=self._CROSSWALK_SURFACE,
            )
            for stripe in crosswalk.get("stripe_polygons_xy", []):
                draw.polygon(
                    [screen(point) for point in stripe],
                    fill=self._CROSSWALK_STRIPE,
                    outline=self._CROSSWALK_SURFACE,
                    width=1,
                )
        # Draw only real road paint.  Use a minimum three-pixel stroke and
        # high contrast so lane markings remain legible after masking and
        # image resizing. Connector topology stays hidden.
        marking_width = max(4, round(0.24 * scale))
        edge_width = max(4, round(0.28 * scale))
        dash_px = max(10, round(3.0 * scale))
        gap_px = max(12, round(5.0 * scale))
        divider_color = palette["boundary"]
        for lanes in self._lane_groups.values():
            if not any(
                    lane["id"] in road_paint_lane_ids for lane in lanes):
                continue
            visible_lanes = [
                lane for lane in lanes
                if visible(lane.get("centerline_xy", []))]
            if not visible_lanes:
                continue
            # Road paint is physical geometry, not privileged route data. Draw
            # every physically visible carriageway, including crossing and
            # oncoming roads; only topology-only connector centerlines stay
            # hidden.
            # Directional lane 0 is rightmost; increasing indexes move left.
            rightmost = lanes[0]
            leftmost = lanes[-1]
            for lane, key in (
                    (leftmost, "left_boundary_xy"),
                    (rightmost, "right_boundary_xy")):
                boundary = self._trim_lane_paint(
                    lane["id"], lane.get(key, []))
                if boundary and visible(boundary):
                    draw.line(
                        [screen(point) for point in boundary],
                        fill=palette["boundary"],
                        width=edge_width,
                        joint="curve",
                    )
            # One dashed shared divider between every adjacent lane pair.
            for lane in lanes[:-1]:
                boundary = self._trim_lane_paint(
                    lane["id"], lane.get("left_boundary_xy", []),
                    stop_line_clearance=True)
                if not boundary or not visible(boundary):
                    continue
                self._draw_dashed_polyline(
                    draw,
                    [screen(point) for point in boundary],
                    fill=divider_color,
                    width=marking_width,
                    dash_px=dash_px,
                    gap_px=gap_px,
                )

        # Infer painted movement arrows from the lane's legal connector set.
        # This reveals only ordinary road markings, never the hidden selected
        # route or all connector trajectories.
        turn_order = ("left", "straight", "right")
        for lane_id in self._stop_lane_ids & relevant_lane_ids:
            lane = self._lane_by_id.get(lane_id)
            if lane is None or not visible(lane.get("centerline_xy", [])):
                continue
            turns = {
                str(self._connector_by_id[connector_id].get("turn", ""))
                for connector_id in self._connectors_by_source.get(lane_id, [])
                if connector_id in self._connector_by_id
            }
            movements = [turn for turn in turn_order if turn in turns]
            if not movements:
                continue
            sampled = self._point_and_tangent_from_end(
                lane["centerline_xy"], 8.0)
            if sampled is None:
                continue
            center_world, tangent_world = sampled
            center_px = screen(center_world)
            ahead_px = screen((
                center_world[0] + tangent_world[0],
                center_world[1] + tangent_world[1],
            ))
            direction_px = (
                ahead_px[0] - center_px[0],
                ahead_px[1] - center_px[1],
            )
            spacing = max(3.0, min(6.0, 0.55 * scale))
            offsets = [
                (index - (len(movements) - 1) / 2.0) * spacing
                for index in range(len(movements))
            ]
            for movement, offset in zip(movements, offsets):
                self._draw_movement_arrow(
                    draw,
                    center_px,
                    direction_px,
                    movement,
                    length_px=max(13.0, min(25.0, 3.2 * scale)),
                    lateral_offset_px=offset,
                )
        for stop in self._stops:
            if stop["lane_id"] not in relevant_lane_ids:
                continue
            if not visible(stop["line_xy"]):
                continue
            points = [screen(point) for point in stop["line_xy"]]
            draw.line(
                points,
                fill=self._STOP_LINE,
                width=max(5, round(0.34 * scale)),
            )
        # Processed LiDAR displays physical bodies but never their lamp state.
        for other_id, other in self.manager.vehicles.items():
            if other_id == vehicle_id or other.arrived or other.route_failed:
                continue
            pose = self.runtime.vehicle_pose(other)
            if pose is None or int(pose[3]) != int(ego_pose[3]):
                continue
            corners_world = self._vehicle_corners(other, pose)
            if not visible(corners_world):
                continue
            corners = [screen(point) for point in corners_world]
            body = (
                self._OTHER_VEHICLE
                if not other.is_crashed else self._CRASHED_VEHICLE)
            draw.polygon(corners, fill=body, outline=(255, 255, 250))
            self._draw_direction_arrow(
                draw, corners, fill=self._VEHICLE_ARROW)

        for pedestrian_id, pedestrian in self.manager.pedestrians.items():
            if not pedestrian.is_spawned or pedestrian.has_arrived:
                continue
            center_world = self.runtime.pedestrian_pose(pedestrian)
            if center_world is None or not visible([center_world]):
                continue
            center = screen(center_world)
            radius = max(
                3, round(float(pedestrian.collision_radius_m) * scale))
            draw.ellipse(
                (center[0] - radius, center[1] - radius,
                 center[0] + radius, center[1] + radius),
                fill=(255, 204, 72), outline=(255, 245, 208), width=1)

        # Apply only the configured LiDAR field of view. No weather, ambient
        # light, signal state or vehicle lamp state participates in this mask.
        fov_mask = Image.new("L", (self.width, self.height), 0)
        mask_draw = ImageDraw.Draw(fov_mask)
        half_fov_deg = min(
            89.0, float(lidar_spec.horizontal_fov_deg) / 2.0)
        top_half_m = min(
            half_width_m,
            base_front_m * math.tan(math.radians(half_fov_deg)))
        top_half_px = top_half_m * scale
        near_half_px = min(10.0, rear_m + 2.0) * scale
        mask_draw.polygon([
            (ego_px[0] - near_half_px, ego_px[1] + 2),
            (ego_px[0] - top_half_px,
             ego_px[1] - base_front_m * scale),
            (ego_px[0] + top_half_px,
             ego_px[1] - base_front_m * scale),
            (ego_px[0] + near_half_px, ego_px[1] + 2),
        ], fill=255)
        image = Image.new("RGB", (self.width, self.height), self._BACKGROUND)
        image.paste(scene, (0, 0), fov_mask)

        # The ego body is only a spatial anchor.  It is drawn after the
        # field-of-view mask and has no label, collision outline, or dashboard
        # data.
        final_draw = ImageDraw.Draw(image)
        ego_corners = [
            screen(point) for point in self._vehicle_corners(ego, ego_pose)]
        final_draw.polygon(
            ego_corners, fill=self._EGO_VEHICLE,
            outline=(236, 251, 255))
        self._draw_direction_arrow(
            final_draw, ego_corners, fill=(246, 252, 255))

        return RenderedAgentImage(
            kind="lidar_bev",
            sim_time_s=float(sim_time_s),
            width=self.width,
            height=self.height,
            png_bytes=self._png(image),
            view="ego_top_down",
        )

    @staticmethod
    def _bounds(points: Iterable[Sequence[float]], padding_m: float) -> tuple:
        values = [tuple(map(float, point[:2])) for point in points]
        if not values:
            return (-100.0, -100.0, 100.0, 100.0)
        return (
            min(point[0] for point in values) - padding_m,
            min(point[1] for point in values) - padding_m,
            max(point[0] for point in values) + padding_m,
            max(point[1] for point in values) + padding_m,
        )

    @staticmethod
    def _intersects_bounds(
        points: Sequence[Sequence[float]], bounds: Sequence[float],
    ) -> bool:
        if not points:
            return False
        min_x, min_y, max_x, max_y = bounds
        for point in points:
            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                return True
        for first, second in zip(points, points[1:]):
            if (max(first[0], second[0]) < min_x
                    or min(first[0], second[0]) > max_x
                    or max(first[1], second[1]) < min_y
                    or min(first[1], second[1]) > max_y):
                continue
            return True
        return False

    def _route_parts(self, preview: dict) -> List[tuple]:
        """Keep travel order, including repeated lanes in a loop."""
        parts = []

        def append(kind: str, item_id: str):
            items = (self._connector_by_id if kind == "connector"
                     else self._lane_by_id)
            item = items.get(item_id)
            if item and (not parts or parts[-1][:2] != (kind, item_id)):
                parts.append((kind, item_id, item["centerline_xy"]))

        append("lane", preview.get("start_lane_id", ""))
        for action in preview.get("actions", []):
            connector = self._connector_by_id.get(action.get("connector_id"), {})
            append("lane", action.get("from_lane_id") or connector.get("from_lane", ""))
            if action.get("type") == "connector":
                append("connector", action.get("connector_id", ""))
            append("lane", action.get("to_lane_id") or connector.get("to_lane", ""))
        return parts

    @staticmethod
    def _route_distance_at_pose(points: Sequence[Point], pose: Sequence[float]) -> float:
        best_distance, best_s, base = float("inf"), 0.0, 0.0
        for a, b in zip(points, points[1:]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            distance, ratio = _point_segment_projection(pose[:2], a, b)
            if distance < best_distance:
                best_distance, best_s = distance, base + ratio * length
            base += length
        return best_s

    @staticmethod
    def _route_line_suffix(points: Sequence[Point], distance_s: float) -> List[Point]:
        """Cut exactly at a projected pose, not at the next map vertex."""
        remaining = max(0.0, distance_s)
        for index, (a, b) in enumerate(zip(points, points[1:])):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if length > 1e-9 and remaining < length:
                ratio = remaining / length
                start = (a[0] + ratio * (b[0] - a[0]),
                         a[1] + ratio * (b[1] - a[1]))
                return [start, *[tuple(p) for p in points[index + 1:]]]
            remaining -= length
        return [tuple(points[-1])] if points else []

    def _remaining_route_lines(
        self, vehicle: Any, preview: Optional[dict], pose: Sequence[float],
    ) -> tuple[List[List[Point]], str]:
        vehicle_id = vehicle.vehicle_id
        if not preview:
            self._minimap_progress.pop(vehicle_id, None)
            return [], ""
        parts = self._route_parts(preview)
        saved, index, cuts = self._minimap_progress.get(
            vehicle_id, (None, 0, {}))
        if saved is not preview:
            # An explicit plan replacement resets even if its geometry is equal.
            index, cuts = 0, {}
        active = vehicle.active_connector_id
        current = ("connector", active) if active else ("lane", vehicle.current_lane_id)
        matched = next((i for i in range(index, len(parts))
                        if parts[i][:2] == current), None)
        if matched is not None:
            projected = self._route_distance_at_pose(parts[matched][2], pose)
            cuts[matched] = max(cuts.get(matched, 0.0), projected)
            index = matched
        elif active:
            # A wrong turn may still have consumed a saved approach lane.
            # Keep the original future route, never switch to that wrong turn.
            source = self._connector_by_id.get(active, {}).get("from_lane")
            consumed = next((i for i in range(index, len(parts))
                             if parts[i][:2] == ("lane", source)), None)
            if consumed is not None:
                index = consumed + 1
        self._minimap_progress[vehicle_id] = (preview, index, cuts)
        lines = []
        adjacent_lanes = matched is not None and current[0] == "lane"
        for i in range(index, len(parts)):
            kind, _, points = parts[i]
            if kind != "lane":
                adjacent_lanes = False
            cut = cuts.get(i, 0.0)
            if adjacent_lanes:
                # Consecutive lane parts are a pending lane-change chain.
                # Their parallel tails behind the ego must disappear as well.
                cut = max(cut, self._route_distance_at_pose(points, pose))
                cuts[i] = cut
            lines.append(self._route_line_suffix(points, cut))
        return lines, str(preview.get("goal_lane_id", ""))

    def render_minimap(
        self,
        vehicle_id: str,
        sim_time_s: float,
        *,
        scope: str = "route",
        route_preview: Any = _ROUTE_PREVIEW_UNSET,
    ) -> RenderedAgentImage:
        """Render a heading-up local navigation map.

        ``route`` keeps a wider look-ahead than ``local`` but deliberately
        does not fit the whole trip into one frame.  Whole-route fitting made
        the next junction only a few pixels wide on long roads, so an agent
        could not reliably distinguish an intersection from a lane branch.
        The route remains highlighted and simply exits the image when its
        destination lies outside the current navigation window.
        """
        vehicle = self.manager.vehicles.get(vehicle_id)
        if vehicle is None:
            raise KeyError(f"unknown vehicle: {vehicle_id}")
        # Direct callers may omit a preview to resolve the assigned route.
        # Tool callers explicitly pass None after a failed plan: do not
        # replace that failure with cached steering or another destination.
        if route_preview is _ROUTE_PREVIEW_UNSET:
            route_preview = self.manager.navigation_route_preview(vehicle_id)
        pose = self.runtime.vehicle_pose(vehicle)
        if pose is None:
            pose = (
                vehicle.pose_x_m, vehicle.pose_y_m,
                vehicle.yaw_rad, vehicle.z_level)
        route_lines, goal_lane_id = self._remaining_route_lines(
            vehicle, route_preview, pose)
        route_available = any(route_lines)
        if scope not in ("route", "local"):
            raise ValueError("scope must be 'route' or 'local'")

        # Navigation follows the driver's frame: forward is always up and
        # the ego marker sits low enough to reserve most pixels for the road
        # ahead.  ``route`` is the normal driving view; ``local`` is a closer
        # inspection of the immediate junction/lane geometry.
        if scope == "route":
            front_m, rear_m, half_width_m = 180.0, 35.0, 90.0
        else:
            front_m, rear_m, half_width_m = 100.0, 25.0, 60.0
        transform = self._local_transform(pose)
        # Render at twice the requested resolution and reduce with Lanczos.
        # This removes the stair-stepping that otherwise dominates curved
        # route connectors without changing the public image dimensions.
        supersample = 2
        canvas_width = self.width * supersample
        canvas_height = self.height * supersample
        padding_px = 18 * supersample
        scale = min(
            (canvas_width - 2 * padding_px) / (2.0 * half_width_m),
            (canvas_height - 2 * padding_px) / (front_m + rear_m))
        ego_x = canvas_width / 2.0
        ego_y = canvas_height - padding_px - rear_m * scale

        def screen(point: Sequence[float]) -> Tuple[int, int]:
            lateral_m, forward_m = transform(point)
            return (
                round(ego_x + lateral_m * scale),
                round(ego_y - forward_m * scale),
            )

        def visible(points: Sequence[Sequence[float]]) -> bool:
            return self._intersects_local(
                points, transform, front_m=front_m, rear_m=rear_m,
                half_width_m=half_width_m)

        image = Image.new(
            "RGB", (canvas_width, canvas_height), self._BACKGROUND)
        draw = ImageDraw.Draw(image)

        # A navigation map describes roads as corridors. Drawing every lane
        # centreline and every legal connector produces a spider-web at urban
        # junctions, so merge each directional carriageway into one surface
        # and fill the connector mouths as a single junction surface.
        visible_road_edges = []
        for lanes in self._lane_groups.values():
            right_boundary = lanes[0].get("right_boundary_xy", [])
            left_boundary = lanes[-1].get("left_boundary_xy", [])
            carriageway = list(left_boundary) + list(reversed(right_boundary))
            if len(carriageway) >= 3 and visible(carriageway):
                draw.polygon(
                    [screen(point) for point in carriageway],
                    fill=self._MINIMAP_ROAD,
                )
                visible_road_edges.extend(
                    boundary for boundary in (
                        left_boundary, right_boundary)
                    if visible(boundary)
                )
                continue
            for lane in lanes:
                points = lane.get("centerline_xy", [])
                if visible(points):
                    draw.line(
                        [screen(point) for point in points],
                        fill=self._MINIMAP_ROAD,
                        width=max(
                            4 * supersample,
                            round(float(lane.get("width_m", 3.5)) * scale),
                        ),
                        joint="curve",
                    )
        for junction in self._junction_surfaces:
            polygon = junction["polygon_xy"]
            if visible(polygon):
                draw.polygon(
                    [screen(point) for point in polygon],
                    fill=self._MINIMAP_ROAD,
                )
        # Draw only longitudinal road edges after the surfaces are merged.
        # Polygon outlines would also draw a transverse cap at every imported
        # segment boundary, making one continuous road look tiled.
        for boundary in visible_road_edges:
            draw.line(
                [screen(point) for point in boundary],
                fill=self._MINIMAP_EDGE,
                width=max(2, supersample),
                joint="curve",
            )

        # Only the close inspection mode exposes ordinary lane dividers.
        # Unselected connector paths remain hidden in both modes; the chosen
        # route itself is the only curve drawn through a junction.
        if scope == "local":
            for lanes in self._lane_groups.values():
                for lane in lanes[:-1]:
                    boundary = lane.get("left_boundary_xy", [])
                    if not visible(boundary):
                        continue
                    self._draw_dashed_polyline(
                        draw,
                        [screen(point) for point in boundary],
                        fill=self._MINIMAP_LANE,
                        width=max(2, supersample),
                        dash_px=5 * supersample,
                        gap_px=6 * supersample,
                    )

        route_width = max(9 * supersample, round(1.7 * scale))
        route_outline_width = route_width + 5 * supersample
        visible_route_parts = [points for points in route_lines
                               if len(points) >= 2 and visible(points)]
        for points in visible_route_parts:
            draw.line(
                [screen(point) for point in points],
                fill=self._MINIMAP_ROUTE_OUTLINE,
                width=route_outline_width,
                joint="curve",
            )
        for points in visible_route_parts:
            draw.line(
                [screen(point) for point in points],
                fill=self._ROUTE,
                width=route_width,
                joint="curve",
            )

        # A visible destination gets both a perpendicular route-end cap and
        # a larger bullseye.  A ring alone was easy for vision models to read
        # as an ordinary waypoint beyond the next junction.
        if goal_lane_id in self.runtime._lane_by_id:
            goal_points = self.runtime._lane_by_id[
                goal_lane_id]["centerline_xy"]
            destination = goal_points[-1]
            destination_local = transform(destination)
            if (-half_width_m <= destination_local[0] <= half_width_m
                    and -rear_m <= destination_local[1] <= front_m):
                dx, dy = screen(destination)
                if len(goal_points) >= 2:
                    before_x, before_y = screen(goal_points[-2])
                    tangent_x = dx - before_x
                    tangent_y = dy - before_y
                    tangent_length = math.hypot(tangent_x, tangent_y)
                    if tangent_length > 1e-6:
                        normal_x = -tangent_y / tangent_length
                        normal_y = tangent_x / tangent_length
                        cap_half = 20 * supersample
                        cap = (
                            round(dx - normal_x * cap_half),
                            round(dy - normal_y * cap_half),
                            round(dx + normal_x * cap_half),
                            round(dy + normal_y * cap_half),
                        )
                        draw.line(
                            cap,
                            fill=self._MINIMAP_ROUTE_OUTLINE,
                            width=10 * supersample,
                        )
                        draw.line(
                            cap,
                            fill=self._MINIMAP_DESTINATION,
                            width=5 * supersample,
                        )
                marker_radius = 18 * supersample
                draw.ellipse(
                    (dx - marker_radius, dy - marker_radius,
                     dx + marker_radius, dy + marker_radius),
                    fill=self._MINIMAP_ROUTE_OUTLINE,
                    outline=self._MINIMAP_DESTINATION,
                    width=4 * supersample,
                )
                inner_radius = 7 * supersample
                draw.ellipse(
                    (dx - inner_radius, dy - inner_radius,
                     dx + inner_radius, dy + inner_radius),
                    fill=self._ROUTE,
                    outline=self._MINIMAP_DESTINATION,
                    width=2 * supersample,
                )

        # Scale is map-only information.  Keeping it in the image lets the
        # driver estimate braking distance without injecting a textual route
        # distance or next-manoeuvre answer into the wake.
        scale_length_m = 50 if scope == "route" else 25
        scale_length_px = round(scale_length_m * scale)
        scale_x = padding_px + 16 * supersample
        scale_y = canvas_height - padding_px - 18 * supersample
        panel = (
            scale_x - 12 * supersample,
            scale_y - 34 * supersample,
            scale_x + scale_length_px + 12 * supersample,
            scale_y + 13 * supersample,
        )
        draw.rounded_rectangle(
            panel,
            radius=8 * supersample,
            fill=self._MINIMAP_ROUTE_OUTLINE,
        )
        draw.line(
            (scale_x, scale_y, scale_x + scale_length_px, scale_y),
            fill=self._MINIMAP_SCALE,
            width=4 * supersample,
        )
        tick_height = 8 * supersample
        for tick_x in (scale_x, scale_x + scale_length_px):
            draw.line(
                (tick_x, scale_y - tick_height,
                 tick_x, scale_y + tick_height),
                fill=self._MINIMAP_SCALE,
                width=4 * supersample,
            )
        try:
            scale_font = ImageFont.truetype(
                "DejaVuSans.ttf", 15 * supersample)
        except OSError:
            scale_font = ImageFont.load_default()
        draw.text(
            (scale_x, scale_y - 31 * supersample),
            f"{scale_length_m} m",
            fill=self._MINIMAP_SCALE,
            font=scale_font,
        )
        cx, cy = screen(pose[:2])
        # The heading-up transform makes the vehicle arrow invariant: its
        # point always indicates the top of the image.
        halo_radius = 17 * supersample
        draw.ellipse(
            (cx - halo_radius, cy - halo_radius,
             cx + halo_radius, cy + halo_radius),
            fill=self._MINIMAP_ROUTE_OUTLINE,
            outline=self._ROUTE,
            width=2 * supersample,
        )
        outer_arrow = [
            (round(cx), round(cy - 14 * supersample)),
            (round(cx - 8 * supersample), round(cy + 9 * supersample)),
            (round(cx + 8 * supersample), round(cy + 9 * supersample)),
        ]
        draw.polygon(outer_arrow, fill=(236, 251, 255))
        inner_arrow = [
            (round(cx), round(cy - 10 * supersample)),
            (round(cx - 5 * supersample), round(cy + 6 * supersample)),
            (round(cx + 5 * supersample), round(cy + 6 * supersample)),
        ]
        draw.polygon(inner_arrow, fill=self._EGO_VEHICLE)
        image = image.resize(
            (self.width, self.height), Image.Resampling.LANCZOS)
        return RenderedAgentImage(
            kind="navigation_minimap",
            sim_time_s=float(sim_time_s),
            width=self.width,
            height=self.height,
            png_bytes=self._png(image),
            view=f"ego_heading_up_{scope}",
            route_available=route_available,
        )
