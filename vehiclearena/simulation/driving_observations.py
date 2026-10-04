"""Immutable local traffic observations shared by perception and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ConnectorConflictObservation:
    other_vehicle_id: str
    other_connector_id: str
    distance_to_conflict_m: float
    other_distance_to_conflict_m: float
    ego_ttc_s: float
    other_ttc_s: float
    other_speed_kmh: float
    other_in_conflict_zone: bool = False


@dataclass(frozen=True)
class PedestrianHazard:
    pedestrian_id: str
    crosswalk_id: str
    distance_to_crosswalk_m: float
    vehicle_ttc_s: float
    crossing_progress: float


@dataclass(frozen=True)
class DriverVehicleObservation:
    vehicle_id: str
    current_speed_kmh: float
    current_lane: int
    is_changing_lane: bool
    lane_change_duration_s: float
    lane_change_progress: float
    length_m: float
    width_m: float
    max_acceleration_mps2: float
    max_braking_mps2: float
    target_lane: int = -1
    observed_signals: Dict[str, bool] = field(default_factory=dict)
    is_crashed: bool = False
    desired_speed_kmh: float = -1.0

    @classmethod
    def from_state(cls, vehicle: Any) -> "DriverVehicleObservation":
        signal_state = getattr(vehicle, "signal_state", None)
        signals = (
            {
                key: bool(value)
                for key, value in signal_state.as_dict().items()
                if isinstance(value, bool)
            }
            if hasattr(signal_state, "as_dict") else {}
        )
        return cls(
            vehicle_id=str(vehicle.vehicle_id),
            current_speed_kmh=float(vehicle.current_speed_kmh),
            current_lane=int(vehicle.current_lane),
            is_changing_lane=bool(vehicle.is_changing_lane),
            lane_change_duration_s=float(vehicle.lane_change_duration_s),
            lane_change_progress=float(vehicle.lane_change_progress),
            length_m=float(vehicle.length_m),
            width_m=float(vehicle.width_m),
            max_acceleration_mps2=float(vehicle.max_acceleration_mps2),
            max_braking_mps2=float(vehicle.max_braking_mps2),
            target_lane=int(getattr(vehicle, "target_lane", -1)),
            observed_signals=signals,
            is_crashed=bool(vehicle.is_crashed),
            desired_speed_kmh=float(vehicle.desired_speed_kmh),
        )


@dataclass(frozen=True)
class DriverContext:
    vehicle: DriverVehicleObservation
    speed_limit_kmh: float
    time_s: float = 0.0
    leader: Optional[DriverVehicleObservation] = None
    leader_gap_m: float = float("inf")
    traffic_light: str = ""
    distance_to_light_m: float = float("inf")
    lane_gaps_m: Dict[int, float] = field(default_factory=dict)
    current_lane: int = 0
    route_blocked: bool = False
    signal_remaining_s: float = 0.0
    distance_to_lane_end_m: float = float("inf")
    route_target_lane: int = -1
    route_lane_change_required: bool = False
    connector_conflict_occupied: bool = False
    downstream_gap_m: float = float("inf")
    connector_id: str = ""
    on_connector: bool = False
    distance_to_connector_end_m: float = float("inf")
    connector_conflicts: List[ConnectorConflictObservation] = field(
        default_factory=list)
    pedestrian_hazards: List[PedestrianHazard] = field(default_factory=list)
    oncoming_vehicle: Optional[DriverVehicleObservation] = None
    oncoming_gap_m: float = float("inf")
    heard_horns: List[Dict[str, Any]] = field(default_factory=list)
