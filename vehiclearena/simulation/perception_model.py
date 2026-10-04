"""Entity-specific optical/acoustic perception over authoritative world truth.

The physics engine may know every pose.  Agents only receive detections that
pass this layer.  The model is deterministic for replay: confidence is a
continuous score and the configured threshold decides visibility; no hidden
per-step random draw is used.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass
class VehicleSignalState:
    """World-visible signal state for one vehicle."""

    left_indicator: bool = False
    right_indicator: bool = False
    hazard: bool = False
    brake_light: bool = False
    low_beam: bool = False
    high_beam: bool = False
    front_fog_light: bool = False
    rear_fog_light: bool = False
    position_light: bool = False
    tail_light: bool = False
    updated_at_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HornEmission:
    event_id: str
    source_id: str
    start_time_s: float
    duration_s: float
    intensity: str
    source_db: float
    pose_x_m: float
    pose_y_m: float

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s


@dataclass(frozen=True)
class PerceptionProfile:
    """Physical sensing capability, separate from control authority."""

    name: str = "human_driver_standard"
    forward_visual_range_m: float = 120.0
    peripheral_visual_range_m: float = 65.0
    rear_visual_range_m: float = 45.0
    horizontal_fov_deg: float = 160.0
    detection_threshold: float = 0.08
    night_sensitivity: float = 0.48
    glare_resistance: float = 0.55
    fog_penetration: float = 0.48
    rain_resistance: float = 0.78
    acoustic_threshold_db: float = 35.0
    cabin_sound_attenuation_db: float = 18.0

    def with_overrides(self, overrides: Optional[dict]) -> "PerceptionProfile":
        if not overrides:
            return self
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown perception profile fields: {sorted(unknown)}")
        profile = replace(self, **overrides)
        positive = (
            "forward_visual_range_m", "peripheral_visual_range_m",
            "rear_visual_range_m", "horizontal_fov_deg",
            "acoustic_threshold_db",
        )
        if any(float(getattr(profile, key)) <= 0.0 for key in positive):
            raise ValueError("perception ranges/FOV/threshold must be positive")
        return profile

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


PERCEPTION_PROFILES: Dict[str, PerceptionProfile] = {
    "human_driver_standard": PerceptionProfile(),
    "human_driver_limited": PerceptionProfile(
        name="human_driver_limited", forward_visual_range_m=85.0,
        peripheral_visual_range_m=45.0, rear_visual_range_m=30.0,
        night_sensitivity=0.35, glare_resistance=0.35,
        fog_penetration=0.35, rain_resistance=0.62,
        cabin_sound_attenuation_db=22.0),
    "enhanced_vision": PerceptionProfile(
        name="enhanced_vision", forward_visual_range_m=180.0,
        peripheral_visual_range_m=95.0, rear_visual_range_m=70.0,
        horizontal_fov_deg=190.0, night_sensitivity=0.72,
        glare_resistance=0.78, fog_penetration=0.65,
        rain_resistance=0.9, cabin_sound_attenuation_db=16.0),
    "pedestrian_standard": PerceptionProfile(
        name="pedestrian_standard", forward_visual_range_m=90.0,
        peripheral_visual_range_m=90.0, rear_visual_range_m=90.0,
        horizontal_fov_deg=360.0, detection_threshold=0.07,
        night_sensitivity=0.42, glare_resistance=0.45,
        fog_penetration=0.42, rain_resistance=0.7,
        acoustic_threshold_db=28.0, cabin_sound_attenuation_db=0.0),
}


@dataclass(frozen=True)
class RadarSpec:
    """One installed short/medium-range millimetre-wave radar."""

    range_m: float
    horizontal_fov_deg: float
    range_error_m: float = 0.15
    speed_error_mps: float = 0.10
    max_tracks: int = 16

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


RADAR_SPECS: Dict[str, RadarSpec] = {
    "frontRadar": RadarSpec(range_m=180.0, horizontal_fov_deg=24.0),
    "rearRadar": RadarSpec(
        range_m=80.0, horizontal_fov_deg=42.0,
        range_error_m=0.20, speed_error_mps=0.12),
}


def resolve_radar_spec(
    module_name: str, overrides: Optional[dict] = None,
) -> RadarSpec:
    """Resolve and strictly validate one per-vehicle radar configuration."""
    if module_name not in RADAR_SPECS:
        raise ValueError(
            f"unknown radar module: {module_name!r}; "
            f"available={sorted(RADAR_SPECS)}")
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(RadarSpec.__dataclass_fields__)
    if unknown:
        raise ValueError(
            f"unknown {module_name} override fields: {sorted(unknown)}")
    spec = replace(RADAR_SPECS[module_name], **overrides)
    positive = ("range_m", "horizontal_fov_deg", "max_tracks")
    if any(float(getattr(spec, key)) <= 0.0 for key in positive):
        raise ValueError(f"{module_name} ranges/FOV/rate/tracks must be positive")
    if spec.horizontal_fov_deg > 180.0:
        raise ValueError(f"{module_name} horizontal_fov_deg must be <= 180")
    if spec.range_error_m < 0.0 or spec.speed_error_mps < 0.0:
        raise ValueError(f"{module_name} measurement errors cannot be negative")
    if int(spec.max_tracks) != spec.max_tracks:
        raise ValueError(f"{module_name} max_tracks must be an integer")
    return replace(spec, max_tracks=int(spec.max_tracks))


def resolve_perception_profile(
    name: str, overrides: Optional[dict] = None, *,
    entity_type: str = "vehicle",
) -> PerceptionProfile:
    default = (
        "pedestrian_standard" if entity_type == "pedestrian"
        else "human_driver_standard")
    profile_name = name or default
    if profile_name not in PERCEPTION_PROFILES:
        raise ValueError(f"unknown perception profile: {profile_name}")
    return PERCEPTION_PROFILES[profile_name].with_overrides(overrides)


@dataclass(frozen=True)
class Detection:
    entity_id: str
    entity_type: str
    distance_m: float
    bearing_deg: float
    confidence: float
    effective_range_m: float
    modalities: List[str] = field(default_factory=lambda: ["visual"])
    observed_signals: Dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerceptionModel:
    """Deterministic local observation model owned by TrafficCoordinator."""

    _WEATHER_VISIBILITY = {
        "sunny": 1.0, "clear": 1.0, "cloudy": 0.94,
        "overcast": 0.9, "rain": 0.76, "rainy": 0.76,
        "heavy_rain": 0.55, "storm": 0.48, "snow": 0.72,
        "snowy": 0.72, "heavy_snow": 0.48,
        "fog": 0.44, "foggy": 0.44, "heavy_fog": 0.25,
        "sandstorm": 0.22,
    }
    _HORN_DB = {"soft": 96.0, "normal": 105.0, "urgent": 110.0}

    def __init__(self, traffic_manager):
        self._tm = traffic_manager

    def profile_for(self, entity_id: str) -> PerceptionProfile:
        vehicle = self._tm.vehicles.get(entity_id)
        if vehicle is not None:
            return resolve_perception_profile(
                getattr(vehicle, "perception_profile_name", ""),
                getattr(vehicle, "perception_overrides", {}),
                entity_type="vehicle")
        pedestrian = self._tm.pedestrians.get(entity_id)
        if pedestrian is not None:
            return resolve_perception_profile(
                getattr(pedestrian, "perception_profile_name", ""),
                getattr(pedestrian, "perception_overrides", {}),
                entity_type="pedestrian")
        return resolve_perception_profile("", entity_type="vehicle")

    def entity_pose(self, entity_id: str) -> Optional[Tuple[float, float, float]]:
        vehicle = self._tm.vehicles.get(entity_id)
        if vehicle is not None:
            return (float(vehicle.pose_x_m), float(vehicle.pose_y_m),
                    float(vehicle.yaw_rad))
        pedestrian = self._tm.pedestrians.get(entity_id)
        if pedestrian is None or not pedestrian.is_spawned:
            return None
        pose = self._tm._lane_geometry.pedestrian_pose(pedestrian)
        if pose is None:
            return None
        return (float(pose[0]), float(pose[1]), 0.0)

    @staticmethod
    def _angle_difference(a: float, b: float) -> float:
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi

    def relative_geometry(
        self, observer_id: str, target_id: str,
    ) -> Optional[Tuple[float, float]]:
        observer = self.entity_pose(observer_id)
        target = self.entity_pose(target_id)
        if observer is None or target is None:
            return None
        dx, dy = target[0] - observer[0], target[1] - observer[1]
        distance = math.hypot(dx, dy)
        absolute = math.atan2(dy, dx)
        relative = self._angle_difference(absolute, observer[2])
        return distance, math.degrees(relative)

    def _weather_factor(self, profile: PerceptionProfile) -> float:
        raw = str(getattr(self._tm, "_current_weather", "sunny"))
        condition = raw.lower().replace(" ", "_")
        base = self._WEATHER_VISIBILITY.get(condition, 0.85)
        if "fog" in condition:
            return 1.0 - (1.0 - base) * (1.0 - profile.fog_penetration)
        if "rain" in condition or "storm" in condition:
            return 1.0 - (1.0 - base) * (1.0 - profile.rain_resistance)
        return base

    def _visible_vehicle_signals(
        self, observer_id: str, target_id: str,
    ) -> Dict[str, bool]:
        """Return lamps physically visible from the observer's side.

        Indicators/hazards and position lamps have side visibility.  Front
        lamps cannot masquerade as rear lamps and a vehicle in front of a
        target cannot observe that target's brake/tail/rear-fog lamps.
        """
        target = self._tm.vehicles.get(target_id)
        observer_pose = self.entity_pose(observer_id)
        target_pose = self.entity_pose(target_id)
        if target is None or observer_pose is None or target_pose is None:
            return {}
        state = getattr(target, "signal_state", VehicleSignalState())
        angle = abs(self._angle_difference(
            math.atan2(
                observer_pose[1] - target_pose[1],
                observer_pose[0] - target_pose[0]),
            target_pose[2]))
        observer_in_front = angle <= math.radians(110.0)
        observer_behind = angle >= math.radians(70.0)
        keys = {
            "left_indicator", "right_indicator", "hazard",
            "position_light",
        }
        if observer_in_front:
            keys.update({"low_beam", "high_beam", "front_fog_light"})
        if observer_behind:
            keys.update({"brake_light", "tail_light", "rear_fog_light"})
        values = state.as_dict()
        return {key: bool(values[key]) for key in keys}

    def visual_envelope(
        self, observer_id: str, *, bearing_deg: float = 0.0,
    ) -> Dict[str, float]:
        profile = self.profile_for(observer_id)
        half_fov = profile.horizontal_fov_deg / 2.0
        abs_bearing = abs(float(bearing_deg))
        if abs_bearing <= half_fov:
            base_range = (
                profile.forward_visual_range_m
                if abs_bearing <= 55.0
                else profile.peripheral_visual_range_m)
        else:
            base_range = profile.rear_visual_range_m
        weather_factor = self._weather_factor(profile)
        night_factor = 1.0
        lighting_factor = 1.0
        observer = self._tm.vehicles.get(observer_id)
        if getattr(self._tm, "_is_night", False):
            night_factor = profile.night_sensitivity
            if observer is not None and abs_bearing <= 55.0:
                signals = getattr(
                    observer, "signal_state", VehicleSignalState())
                condition = str(getattr(
                    self._tm, "_current_weather", "")).lower()
                if signals.low_beam:
                    lighting_factor += 0.65
                if signals.high_beam:
                    lighting_factor += 1.0
                    if "fog" in condition:
                        lighting_factor *= 0.55
                if signals.front_fog_light and "fog" in condition:
                    lighting_factor += 0.35
        effective = base_range * weather_factor * night_factor
        effective *= lighting_factor
        effective = _clamp(effective, 3.0, base_range * 1.35)
        return {
            "base_range_m": round(base_range, 2),
            "effective_range_m": round(effective, 2),
            "weather_factor": round(weather_factor, 4),
            "night_factor": round(night_factor, 4),
            "lighting_factor": round(lighting_factor, 4),
        }

    def detect_entity(
        self, observer_id: str, target_id: str,
        *, claimed_distance_m: Optional[float] = None,
    ) -> Optional[Detection]:
        # A caller-provided distance must not bypass the spawn gate in
        # entity_pose when geometry is unavailable for a scheduled pedestrian.
        pedestrian = self._tm.pedestrians.get(target_id)
        if pedestrian is not None and not pedestrian.is_spawned:
            return None
        geometry = self.relative_geometry(observer_id, target_id)
        if geometry is None:
            if claimed_distance_m is None:
                return None
            distance, bearing = float(claimed_distance_m), 0.0
        else:
            geometric_distance, bearing = geometry
            distance = (
                geometric_distance if geometric_distance > 1e-6
                else float(claimed_distance_m or 0.0))
        envelope = self.visual_envelope(
            observer_id, bearing_deg=bearing)
        effective = envelope["effective_range_m"]
        confidence = _clamp(1.0 - distance / max(effective, 1e-6), 0.0, 1.0)
        profile = self.profile_for(observer_id)
        if distance > effective or confidence < profile.detection_threshold:
            self._record_detection(
                observer_id, target_id, distance, bearing, confidence,
                effective, detected=False, modality="visual")
            return None
        target_vehicle = self._tm.vehicles.get(target_id)
        signals = (
            self._visible_vehicle_signals(observer_id, target_id)
            if target_vehicle is not None else {})
        target_type = (
            "vehicle" if target_vehicle is not None else "pedestrian")
        detection = Detection(
            entity_id=target_id, entity_type=target_type,
            distance_m=round(distance, 2), bearing_deg=round(bearing, 2),
            confidence=round(confidence, 4),
            effective_range_m=round(effective, 2),
            observed_signals={
                key: bool(value) for key, value in signals.items()
                if isinstance(value, bool)},
        )
        self._record_detection(
            observer_id, target_id, distance, bearing, confidence,
            effective, detected=True, modality="visual")
        return detection

    @staticmethod
    def _deterministic_sensor_error(
        observer_id: str, target_id: str, sample_index: int,
        channel: str,
    ) -> float:
        """Return replay-stable noise in [-1, 1] without global RNG state."""
        payload = (
            f"{observer_id}|{target_id}|{sample_index}|{channel}"
        ).encode("utf-8")
        raw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return (raw / float((1 << 64) - 1)) * 2.0 - 1.0

    @staticmethod
    def _vehicle_velocity(vehicle: Any) -> Tuple[float, float]:
        """Return longitudinal plus active lane-change velocity in world XY."""
        yaw = float(vehicle.yaw_rad)
        speed = max(0.0, float(vehicle.current_speed_kmh) / 3.6)
        lateral_speed = float(getattr(vehicle, "lateral_speed_mps", 0.0))
        return (
            speed * math.cos(yaw) - lateral_speed * math.sin(yaw),
            speed * math.sin(yaw) + lateral_speed * math.cos(yaw),
        )

    def _driving_corridor_relevance(
        self,
        observer: Any,
        target: Any,
        lateral_offset_m: float,
        relative_lateral_speed_mps: float,
    ) -> Tuple[bool, str]:
        """Classify whether a radar target occupies or enters ego's path.

        Raw radar still reports every target in its beam.  This classification
        is used only by the collision-warning monitor, preventing a stable car
        in an adjacent lane from being treated as an imminent rear-end hazard.
        Active cut-ins and an ego lane change remain relevant from the moment
        their source/target lane occupancy overlaps.
        """
        corridor_half_width = (
            float(getattr(observer, "width_m", 1.9))
            + float(getattr(target, "width_m", 1.9))) / 2.0 + 0.25
        if abs(lateral_offset_m) <= corridor_half_width:
            return True, "current_body_corridor"

        def occupied_paths(vehicle: Any) -> set:
            paths = set()
            current_lane_id = str(getattr(
                vehicle, "current_lane_id", "") or "")
            active_connector_id = str(getattr(
                vehicle, "active_connector_id", "") or "")
            if current_lane_id:
                paths.add(f"lane:{current_lane_id}")
            if active_connector_id:
                paths.add(f"connector:{active_connector_id}")
            if (getattr(vehicle, "is_changing_lane", False)
                    and int(getattr(vehicle, "target_lane", -1)) >= 0
                    and hasattr(self._tm, "_lane_id_for_index")):
                target_lane_id = self._tm._lane_id_for_index(
                    vehicle, int(vehicle.target_lane))
                if target_lane_id:
                    paths.add(f"lane:{target_lane_id}")
            return paths

        observer_paths = occupied_paths(observer)
        target_paths = occupied_paths(target)
        if observer_paths and target_paths and observer_paths & target_paths:
            if (getattr(observer, "is_changing_lane", False)
                    or getattr(target, "is_changing_lane", False)):
                return True, "lane_change_path_overlap"
            return True, "same_map_path"

        # Stable vehicles on distinct directed lanes of the same segment are
        # separated by lane boundaries.  Do not let a long-range angular beam
        # turn that adjacent-lane observation into a collision warning.
        same_segment = (
            bool(getattr(observer, "current_segment", ""))
            and getattr(observer, "current_segment", "")
            == getattr(target, "current_segment", ""))
        stable = not (
            getattr(observer, "is_changing_lane", False)
            or getattr(target, "is_changing_lane", False))
        if same_segment and stable and observer_paths and target_paths:
            runtime = getattr(self._tm, "_lane_geometry", None)
            lane_by_id = getattr(runtime, "_lane_by_id", {})
            observer_lane = lane_by_id.get(
                str(getattr(observer, "current_lane_id", "")), {})
            target_lane = lane_by_id.get(
                str(getattr(target, "current_lane_id", "")), {})
            shared_corridor = bool(
                observer_lane.get("shared_bidirectional")
                and target_lane.get("shared_bidirectional"))
            if not shared_corridor:
                return False, "separate_stable_lane"

        # For unresolved paths and active lateral maneuvers, project relative
        # lateral motion across the warning horizon.  A target is relevant if
        # its body corridor will intersect ego's, even if it is adjacent now.
        horizon_s = 4.5
        if abs(relative_lateral_speed_mps) > 1e-3:
            closest_time_s = max(0.0, min(
                horizon_s,
                -lateral_offset_m / relative_lateral_speed_mps,
            ))
            projected_offset = (
                lateral_offset_m
                + relative_lateral_speed_mps * closest_time_s)
            if abs(projected_offset) <= corridor_half_width:
                return True, "predicted_corridor_entry"
        return False, "outside_driving_corridor"

    def radar_scan(
        self, observer_id: str, module_name: str,
        overrides: Optional[dict] = None,
        *, include_monitor_metadata: bool = False,
    ) -> Dict[str, Any]:
        """Scan anonymous vehicle targets using an installed front/rear radar.

        Radar is deliberately independent from the optical detector: fog and
        darkness do not turn it into a camera, while heavy precipitation can
        shorten its useful range. Measurements are deterministic at the
        configured sampling rate so a replay receives the same observations.
        """
        spec = resolve_radar_spec(module_name, overrides)
        observer = self._tm.vehicles.get(observer_id)
        if observer is None:
            return {
                "success": False,
                "error": "unknown_observer_vehicle",
                "sensor": module_name,
            }

        direction = "front" if module_name == "frontRadar" else "rear"
        time_s = float(getattr(self._tm, "_physics_time", 0.0))
        # Radar sampling is synchronized with the authoritative 0.1-second
        # world clock. The engine caches this frame; tool calls read it
        # instead of recomputing against a later/private world state.
        sample_index = int(math.floor(time_s * 10.0 + 1e-9))
        sample_time_s = sample_index / 10.0
        weather = str(getattr(self._tm, "_current_weather", "sunny"))
        condition = weather.lower().replace(" ", "_")
        weather_factor = (
            0.62 if condition in {"storm", "sandstorm"}
            else 0.72 if condition in {"heavy_rain", "heavy_snow"}
            else 0.84 if "rain" in condition or "snow" in condition
            else 0.97 if "fog" in condition
            else 1.0
        )
        effective_range = spec.range_m * weather_factor
        half_fov = spec.horizontal_fov_deg / 2.0
        ego_velocity = self._vehicle_velocity(observer)
        tracks = []
        for target_id, target in self._tm.vehicles.items():
            if (target_id == observer_id or target.arrived
                    or getattr(target, "route_failed", False)):
                continue
            if int(getattr(target, "z_level", 0)) != int(
                    getattr(observer, "z_level", 0)):
                continue
            geometry = self.relative_geometry(observer_id, target_id)
            if geometry is None:
                continue
            center_distance, bearing_deg = geometry
            if direction == "front":
                angular_offset = abs(bearing_deg)
            else:
                angular_offset = abs(180.0 - abs(bearing_deg))
            if angular_offset > half_fov or center_distance > effective_range:
                continue

            dx = float(target.pose_x_m) - float(observer.pose_x_m)
            dy = float(target.pose_y_m) - float(observer.pose_y_m)
            safe_center_distance = max(center_distance, 1e-6)
            ray_x, ray_y = dx / safe_center_distance, dy / safe_center_distance
            target_velocity = self._vehicle_velocity(target)
            range_rate = (
                (target_velocity[0] - ego_velocity[0]) * ray_x
                + (target_velocity[1] - ego_velocity[1]) * ray_y
            )
            clearance = max(
                0.0,
                center_distance
                - float(observer.length_m) / 2.0
                - float(target.length_m) / 2.0,
            )
            range_noise = self._deterministic_sensor_error(
                observer_id, target_id, sample_index, "range")
            speed_noise = self._deterministic_sensor_error(
                observer_id, target_id, sample_index, "speed")
            measured_distance = max(
                0.0, clearance + range_noise * spec.range_error_m)
            measured_range_rate = (
                range_rate + speed_noise * spec.speed_error_mps)
            closing_speed = max(0.0, -measured_range_rate)
            ttc_s = (
                measured_distance / closing_speed
                if closing_speed > 0.2 else None)
            left_x, left_y = (
                -math.sin(float(observer.yaw_rad)),
                math.cos(float(observer.yaw_rad)),
            )
            lateral_offset = dx * left_x + dy * left_y
            relative_lateral_speed = (
                (target_velocity[0] - ego_velocity[0]) * left_x
                + (target_velocity[1] - ego_velocity[1]) * left_y)
            corridor_relevant, corridor_relation = \
                self._driving_corridor_relevance(
                    observer, target, lateral_offset,
                    relative_lateral_speed)
            track_hash = hashlib.sha256(
                f"{observer_id}|{target_id}".encode("utf-8")
            ).hexdigest()[:10]
            confidence = _clamp(
                1.0 - center_distance / max(effective_range, 1e-6),
                0.0, 1.0)
            track = {
                "track_id": f"trk_{track_hash}",
                "object_type": "vehicle",
                "direction": direction,
                "distance_m": round(measured_distance, 2),
                "lateral_offset_m": round(lateral_offset, 2),
                "relative_lateral_speed_mps": round(
                    relative_lateral_speed, 2),
                "range_rate_mps": round(measured_range_rate, 2),
                "closing_speed_mps": round(closing_speed, 2),
                "ttc_s": (round(ttc_s, 2) if ttc_s is not None else None),
                "confidence": round(confidence, 4),
            }
            if include_monitor_metadata:
                # Private fusion state for the engine's warning monitor.  It
                # is stripped from the public cached frame returned by the
                # radar tool, which remains an anonymous measurement sensor.
                track["_warning_corridor_relevant"] = corridor_relevant
                track["_warning_corridor_relation"] = corridor_relation
            tracks.append(track)
            self._record_detection(
                observer_id, target_id, measured_distance, bearing_deg,
                confidence, effective_range, detected=True,
                modality=f"radar_{direction}")

        tracks.sort(key=lambda item: item["distance_m"])
        tracks = tracks[:spec.max_tracks]
        return {
            "success": True,
            "sensor": module_name,
            "direction": direction,
            "sample_time_s": round(sample_time_s, 6),
            "measurement_age_s": round(max(0.0, time_s - sample_time_s), 6),
            "weather": weather,
            "effective_range_m": round(effective_range, 2),
            "spec": spec.as_dict(),
            "tracks": tracks,
        }

    def heard_horns(
        self, observer_id: str, time_s: float, *, lookback_s: float = 0.0,
    ) -> List[Dict[str, Any]]:
        profile = self.profile_for(observer_id)
        observer_pose = self.entity_pose(observer_id)
        if observer_pose is None:
            return []
        weather = str(getattr(self._tm, "_current_weather", "")).lower()
        weather_noise = (
            9.0 if "storm" in weather or "heavy_rain" in weather
            else 5.0 if "rain" in weather else 3.0 if "wind" in weather
            else 0.0)
        wind_speed = float(getattr(
            self._tm, "_current_wind_speed_mps", 0.0) or 0.0)
        weather_noise += min(12.0, max(0.0, wind_speed) * 0.6)
        observer_vehicle = self._tm.vehicles.get(observer_id)
        cabin_attenuation = profile.cabin_sound_attenuation_db
        if observer_vehicle is not None:
            opening = _clamp(getattr(
                observer_vehicle, "cabin_open_fraction", 0.0), 0.0, 1.0)
            cabin_attenuation *= 1.0 - 0.85 * opening
        heard = []
        for event in self._tm.horn_events:
            if event.source_id == observer_id:
                continue
            if not (event.start_time_s - 1e-9 <= time_s
                    and event.end_time_s + max(0.0, lookback_s) + 1e-9
                    >= time_s):
                continue
            distance = math.hypot(
                observer_pose[0] - event.pose_x_m,
                observer_pose[1] - event.pose_y_m)
            received = event.source_db - 20.0 * math.log10(
                max(1.0, distance)) - 11.0
            received -= cabin_attenuation + weather_noise
            if received < profile.acoustic_threshold_db:
                continue
            bearing = math.degrees(self._angle_difference(
                math.atan2(
                    event.pose_y_m - observer_pose[1],
                    event.pose_x_m - observer_pose[0]), observer_pose[2]))
            salience = _clamp(
                (received - profile.acoustic_threshold_db) / 35.0,
                0.0, 1.0)
            heard.append({
                "event_id": event.event_id,
                "source_id": event.source_id,
                "distance_m": round(distance, 2),
                "bearing_deg": round(bearing, 2),
                "received_db": round(received, 2),
                "salience": round(salience, 4),
                "intensity": event.intensity,
                "start_time_s": event.start_time_s,
                "end_time_s": event.end_time_s,
            })
        return sorted(heard, key=lambda item: item["distance_m"])

    def acoustic_envelope(
        self, observer_id: str, *, intensity: str = "normal",
    ) -> Dict[str, float]:
        """Maximum free-field horn range under the current conditions."""
        profile = self.profile_for(observer_id)
        weather = str(getattr(self._tm, "_current_weather", "")).lower()
        noise = (
            9.0 if "storm" in weather or "heavy_rain" in weather
            else 5.0 if "rain" in weather else 3.0 if "wind" in weather
            else 0.0)
        wind_speed = float(getattr(
            self._tm, "_current_wind_speed_mps", 0.0) or 0.0)
        noise += min(12.0, max(0.0, wind_speed) * 0.6)
        attenuation = profile.cabin_sound_attenuation_db
        vehicle = self._tm.vehicles.get(observer_id)
        opening = 0.0
        if vehicle is not None:
            opening = _clamp(getattr(
                vehicle, "cabin_open_fraction", 0.0), 0.0, 1.0)
            attenuation *= 1.0 - 0.85 * opening
        source_db = self.horn_source_db(intensity)
        range_m = 10.0 ** max(
            0.0, (source_db - 11.0 - attenuation - noise
                  - profile.acoustic_threshold_db) / 20.0)
        return {
            "intensity": intensity,
            "effective_range_m": round(range_m, 2),
            "cabin_attenuation_db": round(attenuation, 2),
            "ambient_noise_db": round(noise, 2),
            "cabin_open_fraction": round(opening, 3),
        }

    def horn_source_db(self, intensity: str) -> float:
        if intensity not in self._HORN_DB:
            raise ValueError(
                f"horn intensity must be one of {sorted(self._HORN_DB)}")
        return self._HORN_DB[intensity]

    def _record_detection(
        self, observer_id: str, target_id: str, distance: float,
        bearing: float, confidence: float, effective_range: float,
        *, detected: bool, modality: str,
    ) -> None:
        if len(self._tm.perception_log) >= self._tm.perception_log_limit:
            self._tm.perception_log_dropped += 1
            return
        self._tm.perception_log.append({
            "time_s": round(float(self._tm._physics_time), 6),
            "observer_id": observer_id,
            "target_id": target_id,
            "modality": modality,
            "distance_m": round(float(distance), 3),
            "bearing_deg": round(float(bearing), 3),
            "confidence": round(float(confidence), 5),
            "effective_range_m": round(float(effective_range), 3),
            "detected": bool(detected),
        })
