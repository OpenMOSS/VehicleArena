"""Deterministic, trajectory-based evaluation for the physical road world.

The cabin benchmark is intentionally evaluated from declarative YAML rules.
Driving is different: a driver may choose many valid control sequences, so this
module scores the physical outcome sampled from the authoritative 0.1 s world.

The evaluator never writes vehicle state and never prescribes a personality.
It observes:

* safety: collisions and local time-to-collision hazards;
* compliance: signal entry and sustained speeding episodes;
* comfort: acceleration, braking, longitudinal/lateral jerk;
* interaction: unsafe lane-change gaps and intersection spillback;
* communication: turn indicators, high-beam glare and unnecessary horn use;
* efficiency: route progress and unexplained idle time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from evaluation.layer_scoring import single_vehicle_layer_score_100
from evaluation.signal_context import has_restrictive_approach_signal


@dataclass(frozen=True)
class DrivingEvaluationConfig:
    """Thresholds used by :class:`DrivingEvaluator`.

    These are evaluation thresholds, not a driving policy.  Changing them
    changes how a trajectory is graded; it does not modify vehicle control.
    """

    moving_speed_kmh: float = 0.5
    overspeed_tolerance_kmh: float = 3.0
    near_miss_ttc_s: float = 2.0
    # Match the front-radar caution transition.  Starting the grading clock
    # earlier would penalize an LLM before the physical warning can wake it.
    risk_episode_ttc_s: float = 4.5
    # At simulation start every driver is awake and can observe an already
    # closing leader.  This narrow startup-only threshold captures proactive
    # speed matching without moving the normal mid-route warning threshold.
    initial_following_risk_ttc_s: float = 6.0
    connector_time_separation_s: float = 1.25
    connector_horizon_s: float = 5.0
    pedestrian_near_miss_clearance_m: float = 1.5
    pedestrian_near_miss_min_speed_kmh: float = 5.0
    pedestrian_risk_episode_ttc_s: float = 4.0
    hard_acceleration_mps2: float = 2.8  # fallback when chassis capability is unknown
    hard_braking_mps2: float = 4.0
    high_jerk_mps3: float = 5.0
    high_lateral_jerk_mps2: float = 2.0
    unsafe_lane_change_gap_m: float = 6.0
    sustained_overspeed_s: float = 1.0
    max_unexplained_stop_s: float = 3.0
    stop_oncoming_gap_m: float = 100.0
    risk_response_deadline_s: float = 0.5
    green_resume_deadline_s: float = 3.0
    rational_episode_pass_rate: float = 0.90
    minimum_trajectory_observation_s: float = 5.0


@dataclass
class _VehicleAccumulator:
    vehicle_id: str
    first_time_s: Optional[float] = None
    last_time_s: Optional[float] = None
    last_acceleration_mps2: Optional[float] = None
    last_lateral_speed_mps: Optional[float] = None
    last_connector_id: str = ""
    last_lane_change_active: bool = False
    last_lane_change_attempts: int = 0
    last_speed_kmh: Optional[float] = None
    last_traffic_light: str = ""
    terminal_observed: bool = False
    route_destination_node: Optional[str] = None
    initial_remaining_distance_m: Optional[float] = None
    last_remaining_distance_m: Optional[float] = None
    route_distance_available: bool = False
    max_route_progress: float = 0.0

    sample_count: int = 0
    observed_time_s: float = 0.0
    moving_time_s: float = 0.0
    unexplained_idle_s: float = 0.0
    overspeed_time_s: float = 0.0
    hard_acceleration_s: float = 0.0
    hard_braking_s: float = 0.0
    high_jerk_s: float = 0.0
    high_lateral_jerk_s: float = 0.0
    intersection_blocking_s: float = 0.0

    overspeed_episodes: int = 0
    near_miss_episodes: int = 0
    lane_changes: int = 0
    unsafe_lane_changes: int = 0
    red_light_entries: int = 0
    signaled_lane_changes: int = 0
    unsignaled_lane_changes: int = 0
    horn_events: int = 0
    unnecessary_horn_events: int = 0
    high_beam_misuse_s: float = 0.0

    overspeed_active: bool = False
    overspeed_streak_s: float = 0.0
    overspeed_episode_recorded: bool = False
    unexplained_idle_streak_s: float = 0.0
    unexplained_idle_episode_recorded: bool = False
    green_resume_started_s: Optional[float] = None
    green_resume_recorded: bool = False
    green_resume_pending_clearance: bool = False
    active_near_miss_keys: Set[str] = field(default_factory=set)
    # A pedestrian encounter remains active while that person is still a
    # physical route hazard, even if ego stops and TTC temporarily becomes
    # infinite.  Without this lifecycle state, one crossing can be charged
    # again when ego starts moving toward the same person.
    recorded_pedestrian_near_miss_keys: Set[str] = field(
        default_factory=set)
    active_pedestrian_hazard_keys: Set[str] = field(default_factory=set)
    pedestrian_encounter_generations: Dict[str, int] = field(
        default_factory=dict)
    active_risk_episodes: Dict[str, dict] = field(default_factory=dict)
    resolved_risk_keys: Set[str] = field(default_factory=set)
    observed_red_entries: Set[Tuple[str, int]] = field(default_factory=set)
    observed_horn_ids: Set[str] = field(default_factory=set)

    max_speed_kmh: float = 0.0
    max_overspeed_kmh: float = 0.0
    max_acceleration_mps2: float = 0.0
    max_braking_mps2: float = 0.0
    max_jerk_mps3: float = 0.0
    min_ttc_s: float = float("inf")
    min_vehicle_gap_m: float = float("inf")
    min_pedestrian_distance_m: float = float("inf")

    events: List[dict] = field(default_factory=list)
    decision_episodes: List[dict] = field(default_factory=list)


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _round_or_none(value: float, digits: int = 3):
    return round(value, digits) if math.isfinite(value) else None


class DrivingEvaluator:
    """Observe and score physical trajectories without changing the world."""

    def __init__(
        self,
        traffic_manager,
        road_network,
        vehicle_ids: Iterable[str],
        config: Optional[DrivingEvaluationConfig] = None,
    ):
        self.traffic_manager = traffic_manager
        self.road_network = road_network
        self.config = config or DrivingEvaluationConfig()
        self._vehicles: Dict[str, _VehicleAccumulator] = {
            vehicle_id: _VehicleAccumulator(vehicle_id)
            for vehicle_id in vehicle_ids
        }

    def observe(self, time_s: float) -> None:
        """Sample every evaluated vehicle at the current physics time."""
        for vehicle_id, acc in self._vehicles.items():
            if acc.terminal_observed:
                continue
            vehicle = self.traffic_manager.get_state(vehicle_id)
            if vehicle is None:
                continue
            self._observe_vehicle(acc, vehicle, float(time_s))

    def _observe_vehicle(self, acc: _VehicleAccumulator, vehicle, time_s: float):
        previous_time = acc.last_time_s
        dt = (
            max(0.0, time_s - previous_time)
            if previous_time is not None else 0.0
        )
        if acc.first_time_s is None:
            acc.first_time_s = time_s
        acc.last_time_s = time_s
        acc.sample_count += 1
        acc.observed_time_s += dt

        env = self.traffic_manager._build_env_view(
            vehicle, int(round(time_s)))
        awareness = self.traffic_manager.get_driving_awareness(
            vehicle.vehicle_id, ground_truth=True)
        navigation = (
            self.traffic_manager.get_navigation_status(vehicle.vehicle_id)
            if hasattr(self.traffic_manager, "get_navigation_status")
            else {})
        self._observe_navigation(acc, vehicle, navigation)

        speed = max(0.0, float(vehicle.current_speed_kmh))
        acceleration = float(vehicle.acceleration_mps2)
        lateral_speed = float(getattr(vehicle, "lateral_speed_mps", 0.0))
        acc.max_speed_kmh = max(acc.max_speed_kmh, speed)
        acc.max_acceleration_mps2 = max(
            acc.max_acceleration_mps2, acceleration)
        acc.max_braking_mps2 = max(
            acc.max_braking_mps2, max(0.0, -acceleration))

        terminal = bool(vehicle.arrived or vehicle.route_failed or vehicle.is_crashed)
        moving = speed > self.config.moving_speed_kmh
        legitimate_stop = self._has_legitimate_stop_reason(
            vehicle, env, awareness)
        if dt > 0 and not terminal:
            if moving:
                acc.moving_time_s += dt
            elif not legitimate_stop:
                acc.unexplained_idle_s += dt
        self._observe_unexplained_stop(
            acc, moving, legitimate_stop, terminal, dt, time_s)

        self._observe_speed_compliance(acc, env, speed, dt, time_s)
        chassis_acceleration = _finite(
            getattr(vehicle, "max_acceleration_mps2", None))
        hard_acceleration_threshold = (
            0.9 * chassis_acceleration
            if chassis_acceleration is not None and chassis_acceleration > 0
            else self.config.hard_acceleration_mps2)
        self._observe_comfort(
            acc, acceleration, lateral_speed, dt, moving,
            hard_acceleration_threshold)
        self._observe_near_misses(
            acc, vehicle, awareness, time_s, dt)
        self._observe_lane_change(acc, vehicle, awareness, time_s)
        self._observe_signal_entry(acc, vehicle, time_s, dt)
        self._observe_green_resume(
            acc, vehicle, env, awareness, speed, terminal, time_s)
        self._observe_intersection_blocking(
            acc, vehicle, awareness, dt)
        self._observe_communication(
            acc, vehicle, awareness, time_s, dt)

        acc.last_acceleration_mps2 = acceleration
        acc.last_lateral_speed_mps = lateral_speed
        acc.last_connector_id = vehicle.active_connector_id
        acc.last_lane_change_active = bool(vehicle.is_changing_lane)
        acc.last_lane_change_attempts = vehicle.lane_change_attempts
        acc.last_speed_kmh = speed
        if terminal:
            acc.terminal_observed = True

    def _observe_speed_compliance(
        self, acc: _VehicleAccumulator, env, speed: float,
        dt: float, time_s: float,
    ) -> None:
        limit = _finite(getattr(env, "speed_limit_kmh", None))
        over = (
            max(0.0, speed - limit)
            if limit is not None and limit > 0 else 0.0
        )
        acc.max_overspeed_kmh = max(acc.max_overspeed_kmh, over)
        speeding = over > self.config.overspeed_tolerance_kmh
        if speeding:
            acc.overspeed_time_s += dt
            acc.overspeed_streak_s += dt
            if not acc.overspeed_active:
                acc.overspeed_episodes += 1
                acc.events.append({
                    "type": "overspeed_episode",
                    "time_s": round(time_s, 3),
                    "speed_kmh": round(speed, 2),
                    "speed_limit_kmh": round(limit, 2),
                })
            if (not acc.overspeed_episode_recorded
                    and acc.overspeed_streak_s
                    >= self.config.sustained_overspeed_s):
                self._record_decision_episode(
                    acc, "speed_compliance",
                    max(0.0, time_s - acc.overspeed_streak_s), time_s,
                    False, "sustained_overspeed", {
                        "speed_kmh": round(speed, 2),
                        "speed_limit_kmh": round(limit, 2),
                        "duration_s": round(acc.overspeed_streak_s, 3),
                    })
                acc.overspeed_episode_recorded = True
        else:
            acc.overspeed_streak_s = 0.0
            acc.overspeed_episode_recorded = False
        acc.overspeed_active = speeding

    def _observe_unexplained_stop(
        self, acc: _VehicleAccumulator, moving: bool,
        legitimate_stop: bool, terminal: bool, dt: float, time_s: float,
    ) -> None:
        if dt <= 0:
            return
        unexplained = not moving and not legitimate_stop and not terminal
        if unexplained:
            acc.unexplained_idle_streak_s += dt
            if (not acc.unexplained_idle_episode_recorded
                    and acc.unexplained_idle_streak_s
                    > self.config.max_unexplained_stop_s):
                self._record_decision_episode(
                    acc, "traffic_progress",
                    max(0.0, time_s - acc.unexplained_idle_streak_s),
                    time_s, False, "unexplained_stop_too_long", {
                        "duration_s": round(
                            acc.unexplained_idle_streak_s, 3),
                        "limit_s": self.config.max_unexplained_stop_s,
                    })
                acc.unexplained_idle_episode_recorded = True
            return
        acc.unexplained_idle_streak_s = 0.0
        acc.unexplained_idle_episode_recorded = False

    def _observe_comfort(
        self, acc: _VehicleAccumulator, acceleration: float,
        lateral_speed: float, dt: float, moving: bool,
        hard_acceleration_threshold: float,
    ) -> None:
        if dt <= 0:
            return
        if acceleration > hard_acceleration_threshold:
            acc.hard_acceleration_s += dt
        if acceleration < -self.config.hard_braking_mps2:
            acc.hard_braking_s += dt
        if acc.last_acceleration_mps2 is not None:
            jerk = abs(
                acceleration - acc.last_acceleration_mps2) / dt
            acc.max_jerk_mps3 = max(acc.max_jerk_mps3, jerk)
            if moving and jerk > self.config.high_jerk_mps3:
                acc.high_jerk_s += dt
        if acc.last_lateral_speed_mps is not None:
            lateral_change = abs(
                lateral_speed - acc.last_lateral_speed_mps) / dt
            if moving and lateral_change > self.config.high_lateral_jerk_mps2:
                acc.high_lateral_jerk_s += dt

    def _observe_near_misses(
        self, acc: _VehicleAccumulator, vehicle,
        awareness: dict, time_s: float, dt: float,
    ) -> None:
        near_miss_keys: Set[str] = set()
        pedestrian_near_miss_evidence: Dict[str, dict] = {}
        decision_risk_keys: Set[str] = set()
        decision_risk_details: Dict[str, dict] = {}
        pedestrian_hazard_keys: Set[str] = set()

        leader = awareness.get("leader")
        if leader:
            gap = _finite(leader.get("gap_m"))
            other_speed = _finite(leader.get("speed_kmh")) or 0.0
            if gap is not None:
                acc.min_vehicle_gap_m = min(
                    acc.min_vehicle_gap_m, gap)
                closing_mps = max(
                    0.0,
                    (vehicle.current_speed_kmh - other_speed) / 3.6,
                )
                ttc = gap / closing_mps if closing_mps > 1e-6 else None
                if ttc is not None:
                    acc.min_ttc_s = min(acc.min_ttc_s, ttc)
                    if ttc <= self.config.near_miss_ttc_s:
                        near_miss_keys.add(
                            f"leader:{leader.get('vehicle_id', '?')}")
                    key = f"leader:{leader.get('vehicle_id', '?')}"
                    startup_elapsed = (
                        time_s - acc.first_time_s
                        if acc.first_time_s is not None else float("inf"))
                    startup_risk = (
                        0.0 <= startup_elapsed
                        <= self.config.risk_response_deadline_s + 1e-9
                        and ttc
                        <= self.config.initial_following_risk_ttc_s)
                    if (ttc <= self.config.risk_episode_ttc_s
                            or startup_risk):
                        decision_risk_keys.add(key)
                        decision_risk_details[key] = {
                            "entry_ttc_s": round(ttc, 3),
                            "trigger": (
                                "startup_closing_leader"
                                if startup_risk
                                and ttc > self.config.risk_episode_ttc_s
                                else "risk_threshold"),
                        }

        for conflict in awareness.get("connector_conflicts", []):
            ego_ttc = _finite(conflict.get("ego_ttc_s"))
            other_ttc = _finite(conflict.get("other_ttc_s"))
            risky = False
            ttc_for_min = None
            if conflict.get("other_in_conflict_zone"):
                risky = (
                    ego_ttc is not None
                    and ego_ttc <= self.config.near_miss_ttc_s)
                ttc_for_min = ego_ttc
            elif ego_ttc is not None and other_ttc is not None:
                risky = (
                    min(ego_ttc, other_ttc)
                    <= self.config.connector_horizon_s
                    and abs(ego_ttc - other_ttc)
                    <= self.config.connector_time_separation_s)
                ttc_for_min = max(ego_ttc, other_ttc)
            if ttc_for_min is not None:
                acc.min_ttc_s = min(acc.min_ttc_s, ttc_for_min)
            if risky:
                key = (
                    "connector:"
                    + str(conflict.get("other_vehicle_id", "?")))
                near_miss_keys.add(key)
                decision_risk_keys.add(key)

        for hazard in awareness.get("pedestrian_hazards", []):
            pedestrian_id = str(hazard.get("pedestrian_id", "?"))
            if not self._pedestrian_is_physically_active(pedestrian_id):
                # Awareness is normally constructed from the live pedestrian
                # registry, but keep evaluation defensive against a stale
                # cached hazard after SUMO has retired the physical person.
                continue
            key = "pedestrian:" + pedestrian_id
            first_hazard_record = key not in pedestrian_hazard_keys
            pedestrian_hazard_keys.add(key)
            if (first_hazard_record
                    and key not in acc.active_pedestrian_hazard_keys):
                acc.pedestrian_encounter_generations[key] = (
                    acc.pedestrian_encounter_generations.get(key, 0) + 1)
            encounter_id = (
                f"{key}:encounter:"
                f"{acc.pedestrian_encounter_generations[key]}")
            distance = _finite(hazard.get("distance_to_crosswalk_m"))
            ttc = _finite(hazard.get("vehicle_ttc_s"))
            if distance is not None:
                acc.min_pedestrian_distance_m = min(
                    acc.min_pedestrian_distance_m, distance)
            if ttc is not None:
                acc.min_ttc_s = min(acc.min_ttc_s, ttc)
            clearance = self._pedestrian_body_clearance_m(
                vehicle, pedestrian_id)
            if (clearance is not None
                    and clearance <= self.config.pedestrian_near_miss_clearance_m
                    and float(vehicle.current_speed_kmh)
                    > self.config.pedestrian_near_miss_min_speed_kmh):
                near_miss_keys.add(key)
                pedestrian_near_miss_evidence[key] = {
                    "body_clearance_m": round(clearance, 3),
                    "vehicle_speed_kmh": round(
                        float(vehicle.current_speed_kmh), 3),
                    "vehicle_ttc_s": (
                        round(ttc, 3) if ttc is not None else None),
                }
            if (ttc is not None
                    and ttc
                    <= self.config.pedestrian_risk_episode_ttc_s):
                decision_risk_keys.add(key)
                decision_risk_details[key] = {
                    "pedestrian_id": pedestrian_id,
                    "pedestrian_encounter_id": encounter_id,
                }

        for key in sorted(near_miss_keys - acc.active_near_miss_keys):
            if (key.startswith("pedestrian:")
                    and key in acc.recorded_pedestrian_near_miss_keys):
                continue
            acc.near_miss_episodes += 1
            acc.events.append({
                "type": "near_miss",
                "time_s": round(time_s, 3),
                "hazard": key,
                **({
                    "pedestrian_encounter_id": (
                        f"{key}:encounter:"
                        f"{acc.pedestrian_encounter_generations[key]}"),
                    **pedestrian_near_miss_evidence[key],
                } if key.startswith("pedestrian:") else {}),
            })
            if key.startswith("pedestrian:"):
                acc.recorded_pedestrian_near_miss_keys.add(key)

        # A risk episode asks one narrow question: did this vehicle start a
        # physical response within the declared deadline?  It grades the
        # observed trajectory and never prescribes how a driver must decide.
        newly_active = (
            decision_risk_keys
            - set(acc.active_risk_episodes)
            - acc.resolved_risk_keys)
        for key in sorted(newly_active):
            acc.active_risk_episodes[key] = {
                "start_time_s": time_s,
                "episode_type": key.split(":", 1)[0] + "_response",
                "entry_details": decision_risk_details.get(key, {}),
            }

        braking_response = (
            float(vehicle.acceleration_mps2) <= -0.35
            or float(getattr(
                vehicle, "target_speed_kmh",
                vehicle.current_speed_kmh))
            < float(vehicle.current_speed_kmh) - 0.5
            or vehicle.current_speed_kmh <= self.config.moving_speed_kmh
            or (
                dt > 0
                and acc.last_speed_kmh is not None
                and acc.last_speed_kmh - vehicle.current_speed_kmh >= 0.2
            )
        )
        for key, episode in list(acc.active_risk_episodes.items()):
            started = float(episode["start_time_s"])
            elapsed = max(0.0, time_s - started)
            # A timely physical response may move TTC back above the entry
            # threshold in the same 0.1 s sample.  Score that recovery before
            # treating the risk as a transient that disappeared unaided.
            if (braking_response
                    and elapsed
                    <= self.config.risk_response_deadline_s + 1e-9):
                self._record_decision_episode(
                    acc, episode["episode_type"], started, time_s, True,
                    "timely_risk_response", {
                        "hazard": key,
                        "response_time_s": round(elapsed, 3),
                        "deadline_s": self.config.risk_response_deadline_s,
                        **episode.get("entry_details", {}),
                    })
                del acc.active_risk_episodes[key]
                acc.resolved_risk_keys.add(key)
            elif elapsed > self.config.risk_response_deadline_s:
                self._record_decision_episode(
                    acc, episode["episode_type"], started, time_s, False,
                    "risk_response_too_late", {
                        "hazard": key,
                        "response_time_s": None,
                        "deadline_s": self.config.risk_response_deadline_s,
                        **episode.get("entry_details", {}),
                    })
                del acc.active_risk_episodes[key]
                acc.resolved_risk_keys.add(key)
            elif key not in decision_risk_keys:
                # A sub-deadline transient that cleared without intervention
                # is not promoted into a scored decision episode.
                del acc.active_risk_episodes[key]

        acc.resolved_risk_keys = {
            key for key in acc.resolved_risk_keys
            if (
                key in pedestrian_hazard_keys
                if key.startswith("pedestrian:")
                else key in decision_risk_keys
            )
        }
        acc.recorded_pedestrian_near_miss_keys.intersection_update(
            pedestrian_hazard_keys)
        acc.active_pedestrian_hazard_keys = pedestrian_hazard_keys
        acc.active_near_miss_keys = near_miss_keys

    def _pedestrian_body_clearance_m(
        self, vehicle, pedestrian_id: str,
    ) -> Optional[float]:
        """Distance from a pedestrian circle to the vehicle's physical body.

        SUMO reports a vehicle pose at the centre of its front bumper. Its
        rectangle therefore extends one full vehicle length behind that pose.
        TTC to a crosswalk remains useful for warning, but is not evidence of
        a close physical encounter with the pedestrian.
        """
        pedestrians = getattr(self.traffic_manager, "pedestrians", None)
        pedestrian = (
            pedestrians.get(pedestrian_id)
            if pedestrians is not None else None)
        if pedestrian is None:
            return None
        pose = getattr(pedestrian, "physical_pose_xy", None)
        if not isinstance(pose, (tuple, list)) or len(pose) < 2:
            return None
        values = [
            _finite(value) for value in (
                getattr(vehicle, "pose_x_m", None),
                getattr(vehicle, "pose_y_m", None),
                getattr(vehicle, "yaw_rad", None),
                getattr(vehicle, "length_m", None),
                getattr(vehicle, "width_m", None),
                pose[0], pose[1],
                getattr(pedestrian, "collision_radius_m", None),
            )
        ]
        if (any(value is None for value in values)
                or values[3] <= 0 or values[4] <= 0 or values[7] <= 0):
            return None
        front_x, front_y, yaw, length, width, ped_x, ped_y, radius = values
        dx, dy = ped_x - front_x, ped_y - front_y
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
        longitudinal_gap = max(0.0, forward, -length - forward)
        lateral_gap = max(0.0, abs(lateral) - width / 2.0)
        return math.hypot(longitudinal_gap, lateral_gap) - radius

    def _pedestrian_is_physically_active(self, pedestrian_id: str) -> bool:
        """Reject stale pedestrian hazards at their lifecycle boundary.

        Managers used by focused unit tests may not expose a pedestrian
        registry; in that case the supplied awareness remains authoritative.
        A SUMO manager additionally exposes its live proxy registry, allowing
        the evaluator to reject a semantic object whose physical person has
        already been removed between mapped legs.
        """
        pedestrians = getattr(self.traffic_manager, "pedestrians", None)
        if pedestrians is None:
            return True
        pedestrian = pedestrians.get(pedestrian_id)
        if pedestrian is None:
            return False
        if (not bool(getattr(pedestrian, "is_spawned", False))
                or bool(getattr(pedestrian, "has_arrived", False))
                or bool(getattr(pedestrian, "pending_arrival", False))
                or bool(getattr(pedestrian, "is_crashed", False))):
            return False
        proxies = getattr(
            self.traffic_manager, "_sumo_pedestrian_proxy", None)
        if proxies is not None and pedestrian_id not in proxies:
            return False
        return True

    def _observe_lane_change(
        self, acc: _VehicleAccumulator, vehicle, awareness: dict,
        time_s: float,
    ) -> None:
        # A scene may intentionally place an actor partway through a lane
        # change.  That authored initial condition is not a decision made
        # during the evaluated episode.
        if vehicle.lane_change_attempts <= acc.last_lane_change_attempts:
            return
        acc.lane_changes += 1
        target_lane = int(getattr(vehicle, "target_lane", -1))
        signals = vehicle.signal_state
        correctly_signaled = (
            bool(signals.left_indicator)
            if target_lane > int(vehicle.current_lane)
            else bool(signals.right_indicator)
            if target_lane < int(vehicle.current_lane) else False)
        if correctly_signaled:
            acc.signaled_lane_changes += 1
        else:
            acc.unsignaled_lane_changes += 1
            acc.events.append({
                "type": "unsignaled_lane_change",
                "time_s": round(time_s, 3),
                "target_lane": target_lane,
            })
        gap = float("inf")
        follower_ttc = float("inf")
        if vehicle.current_segment and target_lane >= 0:
            gap = self.traffic_manager._lane_neighbor_gap(
                vehicle, vehicle.current_segment, target_lane)
            follower, follower_gap, closing_mps = (
                self.traffic_manager._adjacent_lane_follower(
                    vehicle, vehicle.current_segment, target_lane))
            if follower is not None and closing_mps > 1e-6:
                follower_ttc = follower_gap / closing_mps
        unsafe = (
            gap < self.config.unsafe_lane_change_gap_m
            or follower_ttc < self.config.near_miss_ttc_s)
        action_reason = str(
            (vehicle.llm_control_command or {}).get("reason", ""))
        reason_supported = bool(
            (action_reason and action_reason not in ("cruise", "none"))
            or awareness.get("leader")
            or awareness.get("route_blocked")
            or getattr(vehicle, "lane_route_blocked", False))
        reasonable = correctly_signaled and not unsafe and reason_supported
        self._record_decision_episode(
            acc, "lane_change", time_s, time_s, reasonable,
            "safe_signaled_lane_change" if reasonable
            else "lane_change_requirements_not_met", {
                "target_lane": target_lane,
                "reason": action_reason,
                "reason_supported": reason_supported,
                "correctly_signaled": correctly_signaled,
                "nearest_gap_m": _round_or_none(gap),
                "rear_ttc_s": _round_or_none(follower_ttc),
            })
        if unsafe:
            acc.unsafe_lane_changes += 1
            acc.events.append({
                "type": "unsafe_lane_change",
                "time_s": round(time_s, 3),
                "target_lane": target_lane,
                "nearest_gap_m": _round_or_none(gap),
                "rear_ttc_s": _round_or_none(follower_ttc),
            })

    def _observe_signal_entry(
        self, acc: _VehicleAccumulator, vehicle,
        time_s: float, dt: float,
    ) -> None:
        connector_id = vehicle.active_connector_id
        if not connector_id or connector_id == acc.last_connector_id:
            return
        query_time = max(0.0, time_s - min(max(dt, 0.0), 0.05))
        signal = self.traffic_manager._lane_geometry.signal_state(
            connector_id, query_time)
        if not signal or signal.signal != "red":
            return
        key = (connector_id, int(math.floor(time_s * 10)))
        if key in acc.observed_red_entries:
            return
        acc.observed_red_entries.add(key)
        acc.red_light_entries += 1
        acc.events.append({
            "type": "red_light_entry",
            "time_s": round(time_s, 3),
            "connector_id": connector_id,
        })
        self._record_decision_episode(
            acc, "signal_compliance", time_s, time_s, False,
            "red_light_entry", {"connector_id": connector_id})

    def _observe_green_resume(
        self, acc: _VehicleAccumulator, vehicle, env, awareness: dict,
        speed: float, terminal: bool, time_s: float,
    ) -> None:
        light = getattr(env, "traffic_light", None)
        signal = str(getattr(light, "signal", "") or "")
        distance = _finite(getattr(env, "dist_to_light_m", None))
        close_to_signal = distance is not None and distance <= 30.0
        blocked = self._has_legitimate_stop_reason(
            vehicle, env, awareness)
        if (signal == "green"
                and acc.last_traffic_light in ("red", "yellow")
                and close_to_signal and not terminal):
            acc.green_resume_pending_clearance = blocked
            acc.green_resume_started_s = None if blocked else time_s
            acc.green_resume_recorded = False
        elif signal != "green":
            acc.green_resume_pending_clearance = False
        if (signal == "green"
                and acc.green_resume_pending_clearance
                and not blocked and not terminal):
            # The three-second response clock begins when the crossing,
            # downstream space and other conflict hazards actually clear,
            # not merely when the lamp changes behind an occupied conflict.
            acc.green_resume_pending_clearance = False
            acc.green_resume_started_s = time_s
        if acc.green_resume_started_s is not None:
            elapsed = max(0.0, time_s - acc.green_resume_started_s)
            if speed > self.config.moving_speed_kmh:
                self._record_decision_episode(
                    acc, "green_resume", acc.green_resume_started_s,
                    time_s, elapsed <= self.config.green_resume_deadline_s,
                    "resumed_after_green" if elapsed
                    <= self.config.green_resume_deadline_s
                    else "green_resume_too_late", {
                        "response_time_s": round(elapsed, 3),
                        "deadline_s": self.config.green_resume_deadline_s,
                    })
                acc.green_resume_started_s = None
                acc.green_resume_recorded = True
                acc.green_resume_pending_clearance = False
            elif elapsed > self.config.green_resume_deadline_s:
                self._record_decision_episode(
                    acc, "green_resume", acc.green_resume_started_s,
                    time_s, False, "green_resume_too_late", {
                        "response_time_s": None,
                        "deadline_s": self.config.green_resume_deadline_s,
                    })
                acc.green_resume_started_s = None
                acc.green_resume_recorded = True
                acc.green_resume_pending_clearance = False
        acc.last_traffic_light = signal

    @staticmethod
    def _record_decision_episode(
        acc: _VehicleAccumulator, episode_type: str,
        start_time_s: float, end_time_s: float, reasonable: bool,
        reason: str, details: Optional[dict] = None,
    ) -> None:
        acc.decision_episodes.append({
            "episode_id": (
                f"{acc.vehicle_id}:{episode_type}:"
                f"{len(acc.decision_episodes) + 1}"),
            "type": episode_type,
            "start_time_s": round(float(start_time_s), 3),
            "end_time_s": round(float(end_time_s), 3),
            "reasonable": bool(reasonable),
            "reason": reason,
            "details": dict(details or {}),
        })

    def _observe_intersection_blocking(
        self, acc: _VehicleAccumulator, vehicle,
        awareness: dict, dt: float,
    ) -> None:
        if dt <= 0 or not vehicle.active_connector_id:
            return
        downstream_gap = _finite(awareness.get("downstream_gap_m"))
        required_gap = (
            float(vehicle.length_m)
            + float(self.traffic_manager.config.min_gap_m))
        if (downstream_gap is not None
                and downstream_gap < required_gap
                and vehicle.current_speed_kmh
                <= self.config.moving_speed_kmh):
            acc.intersection_blocking_s += dt

    def _observe_communication(
        self, acc: _VehicleAccumulator, vehicle, awareness: dict,
        time_s: float, dt: float,
    ) -> None:
        if dt > 0 and vehicle.signal_state.high_beam:
            weather = str(getattr(
                self.traffic_manager, "_current_weather", "")).lower()
            misuse = "fog" in weather
            if not misuse:
                model = self.traffic_manager.perception_model
                for other_id, other in self.traffic_manager.vehicles.items():
                    if other_id == vehicle.vehicle_id or other.arrived or other.route_failed:
                        continue
                    geometry = model.relative_geometry(
                        other_id, vehicle.vehicle_id)
                    if geometry is None:
                        continue
                    distance, bearing = geometry
                    yaw_delta = abs(model._angle_difference(
                        float(other.yaw_rad), float(vehicle.yaw_rad)))
                    if (distance <= 120.0 and abs(bearing) <= 55.0
                            and yaw_delta >= math.radians(100.0)):
                        misuse = True
                        break
            if misuse:
                acc.high_beam_misuse_s += dt

        for event in self.traffic_manager.horn_event_log:
            if (event.source_id != vehicle.vehicle_id
                    or event.event_id in acc.observed_horn_ids
                    or event.start_time_s > time_s + 1e-9):
                continue
            acc.observed_horn_ids.add(event.event_id)
            acc.horn_events += 1
            leader = awareness.get("leader") or {}
            leader_gap = _finite(leader.get("gap_m"))
            relevant = bool(
                (leader_gap is not None and leader_gap <= 30.0)
                or awareness.get("oncoming")
                or awareness.get("shared_corridor_hazard")
                or awareness.get("connector_conflicts")
                or awareness.get("pedestrian_hazards")
                or awareness.get("route_blocked"))
            if not relevant:
                acc.unnecessary_horn_events += 1
                acc.events.append({
                    "type": "unnecessary_horn",
                    "time_s": round(float(event.start_time_s), 3),
                    "intensity": event.intensity,
                })

    def _has_legitimate_stop_reason(
        self, vehicle, env, awareness: dict,
    ) -> bool:
        if vehicle.arrived or vehicle.route_failed or vehicle.is_crashed:
            return True
        if awareness.get("route_blocked") or vehicle.lane_route_blocked:
            return True

        distance_to_light = _finite(
            getattr(env, "dist_to_light_m", None))
        if (has_restrictive_approach_signal(env)
                and distance_to_light is not None
                and distance_to_light <= 30.0):
            return True

        leader = awareness.get("leader")
        if leader:
            leader_gap = _finite(leader.get("gap_m"))
            if leader_gap is not None and leader_gap <= 15.0:
                return True

        # The exit-lane gap starts at the connector exit.  Include the
        # remaining path before deciding whether ego is actually blocked.
        downstream_gap = _finite(awareness.get("downstream_gap_m"))
        connector_remaining = _finite(awareness.get(
            "distance_to_connector_end_m"))
        if (bool(vehicle.active_connector_id)
                and downstream_gap is not None
                and connector_remaining is not None
                and connector_remaining + downstream_gap <= 15.0):
            return True
        blocker = awareness.get("connector_path_blocker") or {}
        blocker_gap = _finite(blocker.get("gap_m"))
        if (bool(vehicle.active_connector_id)
                and blocker_gap is not None
                and blocker_gap <= 15.0):
            return True

        oncoming = awareness.get("oncoming")
        if oncoming:
            oncoming_gap = _finite(oncoming.get("gap_m"))
            if (oncoming_gap is not None
                    and oncoming_gap <= self.config.stop_oncoming_gap_m):
                return True
        if awareness.get("shared_corridor_hazard"):
            return True

        for conflict in awareness.get("connector_conflicts", []):
            ego_ttc = _finite(conflict.get("ego_ttc_s"))
            if conflict.get("other_in_conflict_zone"):
                return True
            if ego_ttc is not None and ego_ttc <= 5.0:
                return True

        for hazard in awareness.get("pedestrian_hazards", []):
            distance = _finite(hazard.get("distance_to_crosswalk_m"))
            if distance is not None and distance <= 25.0:
                return True
        return False

    def finalize(self) -> Dict[str, dict]:
        """Return one serialisable report per evaluated vehicle."""
        reports: Dict[str, dict] = {}
        for vehicle_id, acc in self._vehicles.items():
            vehicle = self.traffic_manager.get_state(vehicle_id)
            if vehicle is None:
                continue
            reports[vehicle_id] = self._build_report(acc, vehicle)
        return reports

    def _build_report(self, acc: _VehicleAccumulator, vehicle) -> dict:
        collision_events = self._collision_events_for(vehicle.vehicle_id)
        collision_count = len(collision_events)
        # SUMO records entity_a=collider and entity_b=victim. Exempt the
        # vehicle victim from scoring, while retaining physical involvement.
        # Other/legacy contacts have no reliable roles: keep their penalty.
        at_fault_collisions = [
            item for item in collision_events
            if not (
                getattr(item, "physics_source", "") == "sumo"
                and item.collision_type.startswith("sumo_")
                and item.entity_a_type == item.entity_b_type == "vehicle"
                and item.entity_b == vehicle.vehicle_id)
        ]
        pedestrian_collisions = sum(
            1 for item in collision_events
            if "pedestrian" in (
                item.entity_a_type, item.entity_b_type))

        red_light_violations = acc.red_light_entries

        route_progress = self._route_progress(acc, vehicle)
        active_time = max(
            acc.observed_time_s
            if not vehicle.arrived else acc.moving_time_s
            + acc.unexplained_idle_s,
            1e-6,
        )
        moving_time = max(acc.moving_time_s, 1e-6)

        if at_fault_collisions:
            safety_score = 0.0
        else:
            safety_score = max(
                0.0, 1.0 - 0.15 * acc.near_miss_episodes)

        overspeed_ratio = acc.overspeed_time_s / active_time
        compliance_score = max(
            0.0,
            1.0
            - min(0.6, 1.5 * overspeed_ratio)
            - min(0.25, 0.05 * acc.overspeed_episodes)
            - min(1.0, 0.5 * red_light_violations),
        )

        discomfort = (
            acc.hard_acceleration_s
            + acc.hard_braking_s
            + 0.5 * acc.high_jerk_s
            + 0.5 * acc.high_lateral_jerk_s
        ) / moving_time
        comfort_score = max(0.0, 1.0 - min(1.0, discomfort))

        interaction_score = max(
            0.0,
            1.0
            - min(0.6, 0.2 * acc.unsafe_lane_changes)
            - min(0.4, acc.intersection_blocking_s / 20.0),
        )

        idle_ratio = min(
            1.0, acc.unexplained_idle_s / active_time)
        efficiency_score = max(
            0.0, 0.7 * route_progress + 0.3 * (1.0 - idle_ratio))

        lane_signal_ratio = (
            acc.unsignaled_lane_changes / acc.lane_changes
            if acc.lane_changes else 0.0)
        communication_score = max(
            0.0,
            1.0
            - min(0.55, 0.55 * lane_signal_ratio)
            - min(0.3, acc.high_beam_misuse_s / 20.0)
            - min(0.3, 0.1 * acc.unnecessary_horn_events),
        )

        dimension_scores = {
            "safety": safety_score,
            "compliance": compliance_score,
            "comfort": comfort_score,
            "interaction": interaction_score,
            "communication": communication_score,
            "efficiency": efficiency_score,
        }
        trajectory_score = (
            0.35 * safety_score
            + 0.25 * compliance_score
            + 0.15 * comfort_score
            + 0.10 * interaction_score
            + 0.15 * efficiency_score
        )
        trajectory_quality_score = (
            0.9 * trajectory_score + 0.1 * communication_score)

        hard_violations: List[dict] = []
        for item in at_fault_collisions:
            hard_violations.append({
                "type": (
                    "vehicle_pedestrian_collision"
                    if "pedestrian" in (
                        item.entity_a_type, item.entity_b_type)
                    else "collision"),
                "time_s": round(float(item.time_s), 3),
                "other_entity": (
                    item.entity_b
                    if item.entity_a == vehicle.vehicle_id
                    else item.entity_a),
                "collision_type": item.collision_type,
                "location": item.location,
            })
        if red_light_violations:
            hard_violations.append({
                "type": "red_light_violation",
                "count": red_light_violations,
            })
        if acc.unsafe_lane_changes:
            hard_violations.append({
                "type": "unsafe_lane_change",
                "count": acc.unsafe_lane_changes,
            })

        # An at-fault crash is unreasonable even if a preceding sampled risk
        # never crossed one of the diagnostic TTC thresholds.
        collision_episode_signatures = {
            (item.get("type"), item.get("end_time_s"))
            for item in acc.decision_episodes
        }
        for item in at_fault_collisions:
            signature = ("collision_outcome", round(float(item.time_s), 3))
            if signature not in collision_episode_signatures:
                self._record_decision_episode(
                    acc, "collision_outcome", item.time_s, item.time_s,
                    False, "collision", {
                        "other_entity": (
                            item.entity_b if item.entity_a == vehicle.vehicle_id
                            else item.entity_a),
                        "collision_type": item.collision_type,
                    })

        rational_total = len(acc.decision_episodes)
        rational_passed = sum(
            1 for item in acc.decision_episodes
            if item.get("reasonable", False))
        rational_episode_rate = (
            rational_passed / rational_total if rational_total else None)
        hard_safety_passed = not hard_violations
        task_completed = bool(vehicle.arrived)
        reasonable_driving_pass = (
            bool(
                hard_safety_passed
                and task_completed
                and rational_episode_rate
                >= self.config.rational_episode_pass_rate)
            if rational_total > 0 else None)
        evidence_sufficient = bool(
            task_completed
            or collision_count
            or acc.observed_time_s
            >= self.config.minimum_trajectory_observation_s)

        report = {
            "evaluation_type": "physical_trajectory",
            "agency_attribution": (
                "physical_outcome_only; use command audit for LLM attribution"),
            "applicable": acc.sample_count > 0,
            "evidence_sufficient": evidence_sufficient,
            "score_status": (
                "final" if evidence_sufficient else "provisional"),
            "minimum_trajectory_observation_s": (
                self.config.minimum_trajectory_observation_s),
            "hard_safety_passed": hard_safety_passed,
            "task_completed": task_completed,
            "reasonable_driving_pass": reasonable_driving_pass,
            "rational_episode_rate": (
                round(rational_episode_rate, 4)
                if rational_episode_rate is not None else None),
            "rational_episode_threshold": (
                self.config.rational_episode_pass_rate),
            "trajectory_quality_score": round(
                trajectory_quality_score, 4),
            "dimension_scores": {
                key: round(value, 4)
                for key, value in dimension_scores.items()
            },
            "metrics": {
                "arrived": bool(vehicle.arrived),
                "route_failed": bool(vehicle.route_failed),
                "route_failure_reason": vehicle.route_failure_reason,
                "route_failure_time_s": vehicle.route_failure_time_s,
                "route_progress": (
                    round(route_progress, 4)
                    if vehicle.arrived or acc.route_distance_available else None),
                "best_route_progress": round(route_progress, 4),
                "distance_traveled_m": round(
                    float(vehicle.distance_traveled_m), 3),
                "observed_time_s": round(acc.observed_time_s, 3),
                "moving_time_s": round(acc.moving_time_s, 3),
                "unexplained_idle_s": round(
                    acc.unexplained_idle_s, 3),
                "collision_count": collision_count,
                "at_fault_collision_count": len(at_fault_collisions),
                "pedestrian_collision_count": pedestrian_collisions,
                "near_miss_episodes": acc.near_miss_episodes,
                "red_light_violations": red_light_violations,
                "overspeed_episodes": acc.overspeed_episodes,
                "overspeed_time_s": round(acc.overspeed_time_s, 3),
                "max_overspeed_kmh": round(
                    acc.max_overspeed_kmh, 3),
                "lane_changes": vehicle.lane_change_completions,
                "lane_change_attempts": vehicle.lane_change_attempts,
                "lane_change_uncompleted": vehicle.lane_change_uncompleted,
                "lane_change_pending": max(
                    0, vehicle.lane_change_attempts
                    - vehicle.lane_change_completions
                    - vehicle.lane_change_uncompleted),
                "unsafe_lane_change_attempts": acc.unsafe_lane_changes,
                "signaled_lane_change_attempts": acc.signaled_lane_changes,
                "unsignaled_lane_change_attempts": acc.unsignaled_lane_changes,
                "horn_events": acc.horn_events,
                "unnecessary_horn_events": (
                    acc.unnecessary_horn_events),
                "high_beam_misuse_s": round(
                    acc.high_beam_misuse_s, 3),
                "intersection_blocking_s": round(
                    acc.intersection_blocking_s, 3),
                "hard_acceleration_s": round(
                    acc.hard_acceleration_s, 3),
                "hard_braking_s": round(acc.hard_braking_s, 3),
                "high_jerk_s": round(acc.high_jerk_s, 3),
                "high_lateral_jerk_s": round(
                    acc.high_lateral_jerk_s, 3),
                "max_speed_kmh": round(acc.max_speed_kmh, 3),
                "max_acceleration_mps2": round(
                    acc.max_acceleration_mps2, 3),
                "max_braking_mps2": round(
                    acc.max_braking_mps2, 3),
                "max_jerk_mps3": round(acc.max_jerk_mps3, 3),
                "min_ttc_s": _round_or_none(acc.min_ttc_s),
                "min_vehicle_gap_m": _round_or_none(
                    acc.min_vehicle_gap_m),
                "min_pedestrian_distance_m": _round_or_none(
                    acc.min_pedestrian_distance_m),
                "rational_episode_count": rational_total,
                "rational_episode_passed": rational_passed,
            },
            "hard_violations": hard_violations,
            "decision_episodes": [*acc.decision_episodes],
            "events": [*acc.events],
        }
        report["single_vehicle_layer_score_100"] = (
            single_vehicle_layer_score_100(report))
        return report

    def _collision_events_for(self, vehicle_id: str) -> List[Any]:
        seen = set()
        events = []
        for item in self.traffic_manager._collision_log:
            if vehicle_id not in (item.entity_a, item.entity_b):
                continue
            signature = (
                min(item.entity_a, item.entity_b),
                max(item.entity_a, item.entity_b),
                round(float(item.time_s), 4),
                item.collision_type,
            )
            if signature in seen:
                continue
            seen.add(signature)
            events.append(item)
        return events

    @staticmethod
    def _observe_navigation(acc: _VehicleAccumulator, vehicle, navigation: dict):
        destination_node = str(getattr(vehicle, "destination_node", "") or "")
        if acc.route_destination_node != destination_node:
            # Only a destination change starts a new evaluation journey.
            acc.route_destination_node = destination_node
            acc.initial_remaining_distance_m = None
            acc.max_route_progress = 0.0
        remaining = _finite(navigation.get("remaining_distance_m"))
        acc.route_distance_available = bool(
            navigation.get("status") in {"active", "arrived"}
            and remaining is not None and remaining >= 0.0)
        # Clear stale distance on failure; never infer destination progress
        # from physical steering actions or from reaching a dead-end lane.
        acc.last_remaining_distance_m = (
            remaining if acc.route_distance_available else None)
        if acc.route_distance_available:
            if acc.initial_remaining_distance_m is None:
                acc.initial_remaining_distance_m = remaining
            acc.max_route_progress = max(
                acc.max_route_progress,
                DrivingEvaluator._instantaneous_route_progress(acc, vehicle))

    @staticmethod
    def _instantaneous_route_progress(
        acc: _VehicleAccumulator, vehicle,
    ) -> float:
        initial_remaining = acc.initial_remaining_distance_m
        current_remaining = acc.last_remaining_distance_m
        if (initial_remaining is not None and current_remaining is not None
                and initial_remaining > 1e-6):
            return max(
                0.0, min(
                    0.999,
                    (initial_remaining - current_remaining)
                    / initial_remaining,
                ),
            )
        return 0.0

    @staticmethod
    def _route_progress(acc: _VehicleAccumulator, vehicle) -> float:
        if vehicle.arrived:
            return 1.0
        instantaneous = DrivingEvaluator._instantaneous_route_progress(
            acc, vehicle)
        return max(
            0.0,
            min(0.999, max(acc.max_route_progress, instantaneous)),
        )
