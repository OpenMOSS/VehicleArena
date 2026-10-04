"""Layer-based physical scenario authoring for VehicleArena.

The generator creates maps, actors and exogenous world conditions. Passenger
requests are intentionally absent: formal requests come from a recorded
Personal Agent tape. Generated JSON is the reproducibility artifact.
"""

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Set, Type

from simulation.scenario import WeatherKeyframe, DayNightKeyframe


def _seg_has_lane_in_direction(seg, direction: str) -> bool:
    """True if segment has at least one lane in the given travel direction.

    For oneway segments without explicit lane_config, every lane is forward.
    For two-way segments, checks lane_config explicitly. Catches the case
    where a road is marked oneway=False but lane_config only has one
    direction listed (data quirk seen on some OSM-derived networks).
    """
    if not seg.lane_config:
        # No lane_config → assume forward only when oneway, both otherwise.
        return True if (not seg.oneway or direction == "forward") else False
    return any(lc.direction == direction for lc in seg.lane_config)


def _route_traversable(rn, route) -> bool:
    """Verify every consecutive hop in route has a lane in travel direction."""
    if not route or len(route) < 2:
        return False
    for a, b in zip(route, route[1:]):
        seg = rn.get_edge_by_nodes(a, b)
        if seg is None:
            return False
        # Travel direction is "forward" iff a == seg.from_node
        direction = "forward" if seg.from_node == a else "backward"
        if not _seg_has_lane_in_direction(seg, direction):
            return False
    return True


# =====================================================================
# Element pools (constants)
# =====================================================================

WEATHER_CONDITIONS = [
    "sunny", "cloudy", "rainy", "heavy_rain", "foggy", "snowy",
    "heavy_snow", "hail",
]
WEATHER_ACTIVE = [
    "rainy", "heavy_rain", "foggy", "snowy", "heavy_snow", "hail",
]
WEATHER_CALM = ["sunny", "cloudy"]
WEATHER_SEVERE = [
    "heavy_rain", "snowy", "heavy_snow", "hail", "foggy",
]

DAYNIGHT_CYCLE = ["dawn", "morning", "noon", "afternoon", "dusk", "night"]

# ── Weather intensity defaults ──
WEATHER_INTENSITY = {
    "sunny":      {"rain": 0.0, "snow": 0.0, "fog": 0.0},
    "cloudy":     {"rain": 0.0, "snow": 0.0, "fog": 0.05},
    "rainy":      {"rain": 0.5, "snow": 0.0, "fog": 0.1},
    "heavy_rain": {"rain": 0.9, "snow": 0.0, "fog": 0.2},
    "foggy":      {"rain": 0.0, "snow": 0.0, "fog": 0.9},
    "snowy":      {"rain": 0.0, "snow": 0.4, "fog": 0.1},
    "heavy_snow": {"rain": 0.0, "snow": 0.9, "fog": 0.25},
    "hail":       {"rain": 0.8, "snow": 0.3, "fog": 0.1},
}

TEMP_RANGES = {
    "sunny": (18, 35), "cloudy": (12, 28), "rainy": (8, 22),
    "heavy_rain": (5, 18), "foggy": (2, 15), "snowy": (-5, 2),
    "heavy_snow": (-12, -2), "hail": (-2, 8),
}

WEATHER_DESCRIPTIONS = {
    "rainy": ["It starts to rain, the road is getting wet.", "Rain begins to fall.", "Light rain is starting."],
    "heavy_rain": ["A sudden heavy rainstorm hits, wind is picking up.",
                   "Rain intensifies to a heavy downpour with strong wind.",
                   "Heavy rain and strong wind suddenly hit."],
    "foggy": ["Fog is rolling in, visibility is dropping rapidly.",
              "Dense fog is forming, visibility very low.",
              "Fog is settling in, visibility dropping fast."],
    "snowy": ["Snow starts falling, temperature drops below zero.",
              "Snow begins falling, roads may become icy.",
              "Light snow is starting to fall."],
    "heavy_snow": ["Heavy snow sharply reduces visibility.",
                   "Snowfall intensifies and the road is rapidly covering."],
    "hail": ["Hail is pelting the car, dangerous conditions.",
             "Hailstorm hits, seek shelter if possible."],
    "sunny": ["The sky is clearing up, sun is coming out.",
              "Weather has improved, sunny skies ahead.",
              "The rain has stopped, the sky is clearing up."],
    "cloudy": ["Clouds are gathering overhead.", "The sky is becoming overcast."],
}

DAYNIGHT_DESCRIPTIONS = {
    "dawn": "First light of dawn is breaking.",
    "morning": "Morning has arrived, daylight is good.",
    "noon": "It's now midday, bright sunshine.",
    "afternoon": "It's afternoon, still good daylight.",
    "dusk": "The sun is setting, dusk is approaching.",
    "night": "It is now fully dark.",
}

# ── Initial state options (agent needs to detect and correct) ──
# ── Road events ──
ROAD_EVENT_TYPES = [
    {"event_type": "construction", "severity": "moderate",
     "description": "Road resurfacing, right lane closed",
     "speed_limit_in_zone": 30, "lanes_affected": 1,
     "detour_available": False, "estimated_delay_minutes": 10},
    {"event_type": "accident", "severity": "severe",
     "description": "Multi-vehicle collision, lanes blocked",
     "speed_limit_in_zone": 20, "lanes_affected": 2,
     "detour_available": True, "estimated_delay_minutes": 25},
    {"event_type": "road_closure", "severity": "severe",
     "description": "Water main break, road closed",
     "speed_limit_in_zone": 0, "lanes_affected": 4,
     "detour_available": True, "estimated_delay_minutes": 30},
]

CONGESTION_LEVELS = ["moderate", "heavy", "gridlock"]


# =====================================================================
# ScenarioContext — shared mutable state across layers
# =====================================================================

@dataclass
class ScenarioContext:
    """Shared state between ScenarioLayer.apply() calls."""
    rng: random.Random
    network_id: str
    road_network: Any  # RoadNetwork
    key_nodes: dict
    vehicles: List[dict] = field(default_factory=list)
    duration_minutes: int = 60
    tick_interval: int = 5
    max_tick: int = 12
    used_starts: Set[str] = field(default_factory=set)
    used_times: Set[int] = field(default_factory=set)      # minutes
    used_edges: Set[str] = field(default_factory=set)
    # Layer outputs
    weather_kfs: List[WeatherKeyframe] = field(default_factory=list)
    daynight_kfs: List[DayNightKeyframe] = field(default_factory=list)
    congestion_events: list = field(default_factory=list)
    road_events: list = field(default_factory=list)
    speed_cameras: list = field(default_factory=list)
    pedestrians: list = field(default_factory=list)
    inits_code: Dict[str, str] = field(default_factory=dict)  # {vid: code}

    def available_ticks(self) -> List[int]:
        """Return tick indices (1..max_tick) whose times are not yet used."""
        return sorted(t for t in range(1, self.max_tick + 1)
                      if t * self.tick_interval not in self.used_times)

    def all_edges(self) -> List[str]:
        """Return all edge IDs from the road network."""
        edges = []
        for src, dsts in self.road_network.adjacency.items():
            for dst in dsts:
                edges.append(f"{src}_{dst}")
        return edges

# =====================================================================
# ScenarioLayer base
# =====================================================================

_LAYER_REGISTRY: Dict[str, Type["ScenarioLayer"]] = {}
_LAYER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def register_scenario_layer(name: str, *, order: int = 100):
    """Register a deterministic scenario-composition layer."""
    def decorator(cls: Type["ScenarioLayer"]):
        if not isinstance(name, str) or not _LAYER_NAME_RE.fullmatch(name):
            raise ValueError(
                "Scenario layer name must match [a-z][a-z0-9_]*")
        if not isinstance(cls, type) or not issubclass(cls, ScenarioLayer):
            raise TypeError("Scenario layer must inherit ScenarioLayer")
        if name in _LAYER_REGISTRY:
            existing = _LAYER_REGISTRY[name]
            raise ValueError(
                f"Scenario layer {name!r} is already registered by "
                f"{existing.__module__}.{existing.__name__}")
        cls.layer_name = name
        cls.order = int(order)
        _LAYER_REGISTRY[name] = cls
        return cls
    return decorator


def create_scenario_layer(
    name: str, intensity: str = "mid", **options,
) -> "ScenarioLayer":
    try:
        layer_cls = _LAYER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown scenario layer {name!r}; "
            f"available={sorted(_LAYER_REGISTRY)}") from exc
    return layer_cls(intensity=intensity, **options)


def list_scenario_layers() -> List[str]:
    return sorted(_LAYER_REGISTRY)


class ScenarioLayer(ABC):
    """Abstract base for one physical scenario dimension."""

    INTENSITIES = ("off", "low", "mid", "high")
    layer_name = "base"
    order = 100

    def __init__(self, intensity: str = "mid"):
        if intensity not in self.INTENSITIES:
            raise ValueError(
                f"Invalid intensity {intensity!r}; "
                f"available={list(self.INTENSITIES)}")
        self.intensity = intensity

    @abstractmethod
    def apply(self, ctx: ScenarioContext) -> None:
        """Inject data into ctx. May read prior layers' outputs."""
        ...

    @property
    def is_active(self) -> bool:
        return self.intensity != "off"

    def __repr__(self):
        return f"{type(self).__name__}({self.intensity!r})"


# =====================================================================
# WeatherLayer
# =====================================================================

@register_scenario_layer("weather", order=10)
class WeatherLayer(ScenarioLayer):
    """Weather transitions and conditions."""

    CONFIGS = {
        "off":  {"changes": (0, 0)},
        "low":  {"changes": (0, 1), "transition": False, "reversal": False, "force_severe": False},
        "mid":  {"changes": (1, 2), "transition": True,  "reversal": True,  "force_severe": False},
        "high": {"changes": (3, 5), "transition": True,  "reversal": True,  "force_severe": True},
    }

    def apply(self, ctx: ScenarioContext) -> None:
        if not self.is_active:
            ctx.weather_kfs = [WeatherKeyframe(
                t=0, condition="sunny",
                temperature=ctx.rng.randint(20, 30),
                humidity=ctx.rng.randint(35, 55),
                wind_speed=ctx.rng.randint(3, 10),
            )]
            return

        cfg = self.CONFIGS[self.intensity]
        rng = ctx.rng
        num_changes = rng.randint(*cfg["changes"])

        # Start condition (always calm)
        start_cond = rng.choice(WEATHER_CALM)
        start_temp = rng.randint(*TEMP_RANGES[start_cond])
        kfs = [WeatherKeyframe(
            t=0, condition=start_cond,
            temperature=start_temp,
            humidity=rng.randint(35, 60),
            wind_speed=rng.randint(3, 12),
        )]

        if num_changes == 0:
            ctx.weather_kfs = kfs
            return

        avail = ctx.available_ticks()
        if len(avail) < num_changes:
            num_changes = len(avail)
        if num_changes == 0:
            ctx.weather_kfs = kfs
            return

        change_ticks = sorted(rng.sample(avail[:max(len(avail), num_changes * 3)],
                                          min(num_changes, len(avail))))

        # Pick conditions
        prev_cond = start_cond
        conditions = []
        has_severe = False
        for i in range(len(change_ticks)):
            pool = [c for c in WEATHER_ACTIVE if c != prev_cond]
            # Last change may revert to calm
            if cfg["reversal"] and i == len(change_ticks) - 1 and len(change_ticks) >= 2:
                if rng.random() < 0.5:
                    pool = [c for c in WEATHER_CALM if c != prev_cond]
            cond = rng.choice(pool)
            if cond in WEATHER_SEVERE:
                has_severe = True
            conditions.append(cond)
            prev_cond = cond

        # Force severe if required and none selected
        if cfg["force_severe"] and not has_severe and conditions:
            idx = rng.randrange(len(conditions))
            conditions[idx] = rng.choice(WEATHER_SEVERE)

        for tick_idx, cond in zip(change_ticks, conditions):
            t = tick_idx * ctx.tick_interval
            temp = rng.randint(*TEMP_RANGES[cond])
            desc_list = WEATHER_DESCRIPTIONS.get(cond, [f"Weather changed to {cond}."])
            intensities = WEATHER_INTENSITY.get(cond, {"rain": 0.0, "snow": 0.0, "fog": 0.0})
            transition_min = 0
            if cfg["transition"]:
                transition_min = rng.choice([0, 5, 10, 15])
            kfs.append(WeatherKeyframe(
                t=t, condition=cond, temperature=temp,
                humidity=rng.randint(40, 95),
                wind_speed=rng.randint(2, 35),
                description=rng.choice(desc_list),
                rain_intensity=intensities["rain"],
                snow_intensity=intensities["snow"],
                fog_density=intensities["fog"],
                transition_minutes=transition_min,
            ))
            ctx.used_times.add(t)

        ctx.weather_kfs = kfs


# =====================================================================
# DayNightLayer
# =====================================================================

@register_scenario_layer("daynight", order=20)
class DayNightLayer(ScenarioLayer):
    """Day-night cycle transitions."""

    CONFIGS = {
        "off":  {"changes": (0, 0)},
        "low":  {"changes": (0, 1)},
        "mid":  {"changes": (1, 2)},
        "high": {"changes": (2, 3)},
    }

    def apply(self, ctx: ScenarioContext) -> None:
        rng = ctx.rng

        if not self.is_active:
            ctx.daynight_kfs = [DayNightKeyframe(t=0, period="morning")]
            return

        cfg = self.CONFIGS[self.intensity]
        num_changes = rng.randint(*cfg["changes"])

        start_idx = rng.randint(0, len(DAYNIGHT_CYCLE) - 1)
        start_period = DAYNIGHT_CYCLE[start_idx]
        kfs = [DayNightKeyframe(t=0, period=start_period)]

        if num_changes == 0:
            ctx.daynight_kfs = kfs
            return

        avail = ctx.available_ticks()
        if len(avail) < num_changes:
            num_changes = len(avail)
        if num_changes == 0:
            ctx.daynight_kfs = kfs
            return

        change_ticks = sorted(rng.sample(avail[:max(len(avail), num_changes * 3)],
                                          min(num_changes, len(avail))))

        current_idx = start_idx
        for tick_idx in change_ticks:
            step = rng.randint(1, 2)
            current_idx += step
            if current_idx >= len(DAYNIGHT_CYCLE):
                break
            period = DAYNIGHT_CYCLE[current_idx]
            t = tick_idx * ctx.tick_interval
            desc = DAYNIGHT_DESCRIPTIONS.get(period, f"Time changed to {period}.")
            kfs.append(DayNightKeyframe(t=t, period=period, description=desc))
            ctx.used_times.add(t)

        ctx.daynight_kfs = kfs


# =====================================================================
# TrafficLayer
# =====================================================================

@register_scenario_layer("traffic", order=40)
class TrafficLayer(ScenarioLayer):
    """Congestion, road events, and speed cameras."""

    CONFIGS = {
        "off":  {"congestion": (0, 0), "road_events": (0, 0), "cameras": (0, 0), "targeted": False},
        "low":  {"congestion": (0, 1), "road_events": (0, 1), "cameras": (0, 0), "targeted": False},
        "mid":  {"congestion": (1, 2), "road_events": (1, 2), "cameras": (1, 2), "targeted": False},
        "high": {"congestion": (2, 4), "road_events": (2, 3), "cameras": (2, 4), "targeted": True},
    }

    def __init__(self, intensity: str = "mid", targeted: bool = None):
        super().__init__(intensity)
        self._targeted = targeted

    def apply(self, ctx: ScenarioContext) -> None:
        if not self.is_active:
            return

        cfg = self.CONFIGS[self.intensity]
        rng = ctx.rng
        targeted = self._targeted if self._targeted is not None else cfg["targeted"]
        evaluated = [v for v in ctx.vehicles if v.get("is_evaluated", True)]

        all_edges = ctx.all_edges()

        # ── Congestion ──
        num_cong = rng.randint(*cfg["congestion"])
        if targeted and evaluated:
            self._targeted_congestion(ctx, evaluated, min(num_cong, len(evaluated)))
            num_cong = max(0, num_cong - len(evaluated))

        for _ in range(num_cong):
            avail = [e for e in all_edges if e not in ctx.used_edges]
            if not avail:
                break
            edge = rng.choice(avail)
            ctx.used_edges.add(edge)
            start_tick = rng.randint(2, max(3, ctx.max_tick // 2))
            end_tick = start_tick + rng.randint(3, 8)
            level = rng.choice(CONGESTION_LEVELS)
            speed = {"moderate": 30, "heavy": 8, "gridlock": 0}[level]
            wait = {"moderate": 1, "heavy": 1, "gridlock": 2}[level]
            ctx.congestion_events.append({
                "edge_id": edge,
                "start_tick": start_tick,
                "end_tick": min(end_tick, ctx.max_tick),
                "level": level, "wait_ticks": wait,
                "estimated_speed": speed,
                "description": f"Congestion: {level}",
            })

        # ── Road events ──
        num_road = rng.randint(*cfg["road_events"])
        if targeted and evaluated and num_road > 0:
            self._targeted_road_event(ctx, evaluated[0])
            num_road -= 1

        for _ in range(num_road):
            avail = [e for e in all_edges if e not in ctx.used_edges]
            if not avail:
                break
            edge = rng.choice(avail)
            ctx.used_edges.add(edge)
            template = rng.choice(ROAD_EVENT_TYPES)
            start_tick = rng.randint(0, max(1, ctx.max_tick // 3))
            end_tick = start_tick + rng.randint(5, 12)
            ctx.road_events.append({
                "edge_id": edge,
                "start_tick": start_tick,
                "end_tick": min(end_tick, ctx.max_tick),
                **template,
            })

        # ── Speed cameras ──
        num_cams = rng.randint(*cfg["cameras"])
        cam_edges: Set[str] = set()
        for _ in range(num_cams):
            avail = [e for e in all_edges if e not in cam_edges]
            if not avail:
                break
            edge = rng.choice(avail)
            cam_edges.add(edge)
            ctx.speed_cameras.append({
                "edge_id": edge,
                "position_ratio": round(rng.uniform(0.3, 0.8), 1),
                "speed_limit": rng.choice([30, 40, 50, 60]),
                "camera_type": rng.choice(["fixed", "temporary"]),
            })

    def _targeted_congestion(self, ctx: ScenarioContext,
                             evaluated: List[dict], count: int):
        """Place congestion on evaluated vehicles' routes."""
        rng = ctx.rng
        for vcfg in evaluated[:count]:
            route = ctx.road_network.plan_route(
                vcfg["initial_node"], vcfg["destination_node"], time_sec=0)
            if not route or len(route) < 5:
                continue
            mid = len(route) // 2
            edge = f"{route[mid]}_{route[mid + 1]}"
            if edge in ctx.used_edges:
                continue
            ctx.used_edges.add(edge)
            start_tick = rng.randint(1, max(2, ctx.max_tick // 3))
            end_tick = start_tick + rng.randint(4, 10)
            level = rng.choice(["heavy", "gridlock"])
            speed = {"heavy": 8, "gridlock": 0}[level]
            wait = {"heavy": 1, "gridlock": 2}[level]
            ctx.congestion_events.append({
                "edge_id": edge,
                "start_tick": start_tick,
                "end_tick": min(end_tick, ctx.max_tick),
                "level": level, "wait_ticks": wait,
                "estimated_speed": speed,
                "description": f"Targeted congestion: {level}",
            })

    def _targeted_road_event(self, ctx: ScenarioContext, vcfg: dict):
        """Place road closure on the first third of a vehicle's route."""
        route = ctx.road_network.plan_route(
            vcfg["initial_node"], vcfg["destination_node"], time_sec=0)
        if not route or len(route) < 6:
            return
        idx = len(route) // 3
        edge = f"{route[idx]}_{route[idx + 1]}"
        if edge in ctx.used_edges:
            return
        ctx.used_edges.add(edge)
        start_tick = ctx.rng.randint(0, max(1, ctx.max_tick // 4))
        end_tick = start_tick + ctx.rng.randint(6, 14)
        ctx.road_events.append({
            "edge_id": edge,
            "start_tick": start_tick,
            "end_tick": min(end_tick, ctx.max_tick),
            "event_type": "road_closure", "severity": "severe",
            "description": "Targeted road closure, forcing reroute",
            "speed_limit_in_zone": 0, "lanes_affected": 4,
            "detour_available": True, "estimated_delay_minutes": 30,
        })


@register_scenario_layer("pedestrian", order=60)
class PedestrianLayer(ScenarioLayer):
    """Spawn SUMO-managed pedestrians at crosswalk-equipped intersections.

    Each pedestrian gets ``initial_node`` (a node with has_crosswalk) and
    a ``destination_node`` (a random neighbor). Start times are spread
    over the first half of the run so pedestrians actually meet vehicles.
    """

    CONFIGS = {
        "off":  {"count": (0, 0)},
        "low":  {"count": (2, 4)},
        "mid":  {"count": (5, 10)},
        "high": {"count": (10, 20)},
    }

    def apply(self, ctx: ScenarioContext) -> None:
        if not self.is_active:
            return

        cfg = self.CONFIGS[self.intensity]
        rng = ctx.rng
        count = rng.randint(*cfg["count"])
        if count == 0:
            return

        rn = ctx.road_network
        if rn is None:
            return

        # Eligible spawn nodes: signalled intersections + at least one neighbor.
        # (has_crosswalk only gets set during scenario_dict load via
        # _auto_generate_traffic_lights, which hasn't run at generation time;
        # signal=True nodes are exactly the ones that will gain crosswalks.)
        crosswalk_nodes = [
            nid for nid, node in rn.nodes.items()
            if getattr(node, "signal", False)
            and rn.adjacency.get(nid)
        ]
        if not crosswalk_nodes:
            return

        rng.shuffle(crosswalk_nodes)
        used_nodes: Set[str] = set()
        # Pedestrians appear within the first half of the scenario.
        half_minutes = max(ctx.tick_interval,
                           (ctx.max_tick // 2) * ctx.tick_interval)
        for i in range(count):
            avail = [n for n in crosswalk_nodes if n not in used_nodes]
            if not avail:
                break
            init_node = avail[0]
            used_nodes.add(init_node)
            neighbors = list(rn.adjacency.get(init_node, []))
            if not neighbors:
                continue
            dest = rng.choice(neighbors)
            start_min = rng.randint(0, max(0, half_minutes))
            ctx.pedestrians.append({
                "ped_id": f"ped_{i+1}",
                "initial_node": init_node,
                "destination_node": dest,
                "agent_config": {"type": "sumo"},
                "speed": round(rng.uniform(1.1, 1.6), 2),
                "start_time": float(start_min),
                "is_evaluated": False,
            })


# ScenarioGenerator
# =====================================================================

class ScenarioGenerator:
    """Layer-based scenario generator for VehicleArena benchmark."""

    def __init__(self):
        # Generated scenarios are authoring artifacts. Reproducibility comes
        # from freezing the emitted JSON, not from a runtime seed parameter.
        self.rng = random.Random()
        self._counter = 0

    def compose(
        self,
        network: str = "beijing_zhongguancun",
        num_vehicles: int = 4,
        duration: int = 60,
        tick_interval: int = 5,
        layers: List[ScenarioLayer] = None,
        vehicle_capabilities: Dict[str, dict] = None,
    ) -> dict:
        """Compose a multi-vehicle scenario using layer stacking.

        Args:
            network: Road network ID.
            num_vehicles: Number of vehicles.
            duration: Duration in minutes.
            tick_interval: Minutes between ticks.
            layers: List of ScenarioLayer instances.
            vehicle_capabilities: Per-vehicle equipment/chassis overrides.
                Use ``"*"`` for defaults and a vehicle ID for overrides.
        Returns:
            Scenario dict accepted by MultiScenario.from_dict().
        """
        from simulation.road_networks import list_key_nodes, load_road_network

        self._counter += 1
        rn = load_road_network(network)
        key_nodes = list_key_nodes(network)
        max_tick = duration // tick_interval

        # 1. Pick vehicle routes
        vehicles = self._pick_vehicle_routes(rn, key_nodes, num_vehicles)
        used_starts = {v["initial_node"] for v in vehicles}

        # 2. Create shared context
        ctx = ScenarioContext(
            rng=self.rng,
            network_id=network,
            road_network=rn,
            key_nodes=key_nodes,
            vehicles=vehicles,
            duration_minutes=duration,
            tick_interval=tick_interval,
            max_tick=max_tick,
            used_starts=used_starts,
        )

        # 3. Apply layers in dependency order
        if layers is None:
            layers = []

        sorted_layers = sorted(
            layers,
            key=lambda layer: (layer.order, layer.layer_name),
        )

        for layer in sorted_layers:
            if layer.is_active:
                layer.apply(ctx)
        self._apply_vehicle_capabilities(
            ctx.vehicles, vehicle_capabilities or {})

        # Ensure defaults if no Weather/DayNight layer
        if not ctx.weather_kfs:
            ctx.weather_kfs = [WeatherKeyframe(
                t=0, condition="sunny",
                temperature=self.rng.randint(20, 30),
                humidity=self.rng.randint(35, 55),
                wind_speed=self.rng.randint(3, 10),
            )]
        if not ctx.daynight_kfs:
            ctx.daynight_kfs = [DayNightKeyframe(t=0, period="morning")]

        # 4. Assemble scenario dict
        scenario = self._build_scenario_dict(ctx)
        # Generation and hand-written scenarios share one authoritative
        # parser. A custom layer therefore fails at composition time instead
        # of much later during engine initialization.
        from simulation.multi_sim_engine import MultiScenario
        MultiScenario.from_dict(scenario)
        return scenario

    # ── Private helpers ──────────────────────────────────────

    @staticmethod
    def _apply_vehicle_capabilities(
        vehicles: List[dict], configurations: Dict[str, dict],
    ) -> None:
        allowed = {
            "equipment_profile",
            "chassis_profile",
            "enable_modules",
            "disable_modules",
            "chassis_overrides",
        }
        defaults = configurations.get("*", {}) or {}
        for vehicle in vehicles:
            specific = configurations.get(
                vehicle.get("vehicle_id", ""), {}) or {}
            merged = {**defaults, **specific}
            unknown = set(merged) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown vehicle capability fields: "
                    f"{sorted(unknown)}")
            vehicle.setdefault("equipment_profile", "executive")
            vehicle.setdefault("chassis_profile", "sedan")
            vehicle.update(merged)

    def _pick_vehicle_routes(self, rn, key_nodes: dict,
                             num_vehicles: int) -> List[dict]:
        """Pick start/destination pairs with route reachability.

        Uses key_nodes as preferred node pool; falls back to all road-network
        nodes (for maps without registered key_nodes).
        """
        all_nodes = list(rn.nodes.keys())
        poi_nodes = list(key_nodes.values()) if key_nodes else []
        if num_vehicles <= len(poi_nodes):
            node_ids = poi_nodes
        else:
            # Merge POI + all nodes when requested count exceeds POI pool.
            node_ids = list(set(poi_nodes) | set(all_nodes))
        if len(node_ids) < 2:
            raise ValueError(
                f"Network has only {len(node_ids)} usable node(s); "
                "cannot pick start/destination pair."
            )
        vehicles = []
        used_starts: Set[str] = set()
        vid_names = ["ego"] + [f"v{i}" for i in range(2, num_vehicles + 1)]

        for vid in vid_names:
            route_found = False
            attempts = 0
            start, dest = "", ""
            while not route_found and attempts < 30:
                start = self.rng.choice(node_ids)
                dest = self.rng.choice(node_ids)
                if start == dest or start in used_starts:
                    attempts += 1
                    continue
                route = rn.plan_route(start, dest, time_sec=0)
                if (route and len(route) >= 3
                        and _route_traversable(rn, route)):
                    route_found = True
                attempts += 1

            if not route_found:
                # Random sampling failed — exhaustively probe candidate
                # destinations from a fresh start to guarantee reachability.
                # Skip this vehicle slot if no reachable pair exists at all.
                shuffled_starts = list(node_ids)
                self.rng.shuffle(shuffled_starts)
                shuffled_dests = list(node_ids)
                self.rng.shuffle(shuffled_dests)
                for s in shuffled_starts:
                    if s in used_starts:
                        continue
                    for d in shuffled_dests:
                        if d == s:
                            continue
                        r = rn.plan_route(s, d, time_sec=0)
                        if (r and len(r) >= 3
                                and _route_traversable(rn, r)):
                            start, dest = s, d
                            route_found = True
                            break
                    if route_found:
                        break
                if not route_found:
                    # Network has no reachable triplet for any unused start —
                    # drop this vehicle slot rather than emitting a stuck car
                    continue

            used_starts.add(start)
            dest_name = ""
            for kname, knid in key_nodes.items():
                if knid == dest:
                    dest_name = kname.replace("_", " ").title()
                    break

            vehicles.append({
                "vehicle_id": vid,
                "initial_node": start,
                "destination_node": dest,
                "destination_name": dest_name,
                "initial_lane": 0,
                "is_evaluated": vid == "ego",
                "equipment_profile": "full",
                "chassis_profile": "sedan",
                "agent_config": {
                    "type": "llm" if vid == "ego" else "sumo"},
            })

        return vehicles

    def _build_scenario_dict(self, ctx: ScenarioContext) -> dict:
        """Assemble final scenario dict from context."""
        # Build name
        weather_tags = [kf.condition for kf in ctx.weather_kfs
                        if kf.condition not in ("sunny", "cloudy")]
        daynight_tags = [kf.period for kf in ctx.daynight_kfs
                         if kf.period in ("night", "dusk", "dawn")]
        name_parts = weather_tags[:2] + daynight_tags[:1]
        name = " + ".join(name_parts).title() if name_parts else "Drive"
        name += f" — {len(ctx.vehicles)}V"

        scenario_id = f"gen_{self._counter:04d}"

        # Inits code: merge per-vehicle inits into a single string
        inits_code = ""
        if ctx.inits_code:
            inits_code = "\n".join(ctx.inits_code.values())

        return {
            "scenario_id": scenario_id,
            "name": name,
            "road_network_id": ctx.network_id,
            "total_time_s": ctx.duration_minutes * 60,
            "tick_interval_s": ctx.tick_interval * 60,
            "vehicles": ctx.vehicles,
            "weather_keyframes": [
                {k: v for k, v in asdict(kf).items() if k != "description" or v}
                for kf in ctx.weather_kfs
            ],
            "daynight_keyframes": [
                {k: v for k, v in asdict(kf).items() if k != "description" or v}
                for kf in ctx.daynight_kfs
            ],
            "congestion_events": ctx.congestion_events,
            "road_events": ctx.road_events,
            "speed_cameras": ctx.speed_cameras,
            "pedestrians": ctx.pedestrians,
            "inits_code": inits_code,
        }
