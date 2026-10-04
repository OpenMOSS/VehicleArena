"""Static and executable validation for the two-study scene catalog."""

from __future__ import annotations

import copy
import ctypes
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from evaluation.experiments.scene_catalog import (
    CATALOG_SCHEMA, Topology, iter_catalog_scenes,
)
from evaluation.experiments.environment_design import (
    DAYNIGHT_EDGES,
    WEATHER_CONDITIONS,
    WEATHER_TRANSITION_EDGES,
    weather_condition_allowed,
    weather_condition_month_allowed,
)
from simulation.lane_level_runtime import LaneGeometryRuntime
from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from evaluation.experiments.task_compatibility import crosswalk_signal_relationship, REVISION


def _point_segment_distance(point, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(0.0, min(1.0, (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_sq))
    projection = start[0] + dx * ratio, start[1] + dy * ratio
    return math.hypot(point[0] - projection[0], point[1] - projection[1])


def _paths_near(first: Sequence[Sequence[float]],
                second: Sequence[Sequence[float]], threshold: float) -> bool:
    return any(
        _point_segment_distance(point, start, end) <= threshold
        for point in first
        for start, end in zip(second, second[1:])
    ) or any(
        _point_segment_distance(point, start, end) <= threshold
        for point in second
        for start, end in zip(first, first[1:])
    )


class SceneValidationError(ValueError):
    pass


class SceneValidator:
    def __init__(self, network_dir: Path):
        self.network_dir = Path(network_dir)
        self._runtime: Dict[str, LaneGeometryRuntime] = {}

    def runtime(self, network: str) -> LaneGeometryRuntime:
        if network not in self._runtime:
            # A full 12-map validation does not need all large topology
            # indexes resident simultaneously. Catalog callers group/revisit
            # networks deterministically; one active runtime is sufficient.
            self._runtime.clear()
            gc.collect()
            try:
                ctypes.CDLL(None).malloc_trim(0)
            except (AttributeError, OSError):
                pass
            path = self.network_dir / f"{network}_lane_level.json"
            self._runtime[network] = LaneGeometryRuntime(
                json.loads(path.read_text(encoding="utf-8")))
        return self._runtime[network]

    def validate(self, scenario: Mapping[str, Any],
                 expected: Mapping[str, Any]) -> dict:
        scene_id = str(scenario.get("scenario_id", ""))
        errors = []

        def require(value: bool, message: str) -> None:
            if not value:
                errors.append(message)

        require(bool(scene_id), "missing scenario_id")
        require(expected.get("schema") == CATALOG_SCHEMA,
                "unexpected expected.json schema")
        require(expected.get("scene_id") == scene_id,
                "expected.json scene_id mismatch")
        try:
            MultiScenario.from_dict(dict(scenario))
        except Exception as exc:
            errors.append(f"MultiScenario parse failed: {exc}")

        network = str(scenario.get("road_network_id", ""))
        scene_metadata = scenario.get("experiment_scene") or {}
        if scene_metadata.get("runtime_compatibility", {}).get("task_config_revision") == REVISION:
            for vehicle in scenario.get("vehicles", []):
                require("native_lane_change_enabled" not in vehicle.get("initial_physical_state", {}),
                        "native_lane_change_enabled is unsupported by the old engine")
            for pedestrian in scenario.get("pedestrians", []):
                require("crossing_trigger" not in pedestrian.get("initial_physical_state", {}),
                        "crossing_trigger is unsupported by the old engine")
            require(not scenario.get("experiment_world_events"),
                    "SUMO-native tasks cannot contain runtime world events")
        weather_profile = scene_metadata.get("weather_profile") or {}
        weather_keyframes = list(scenario.get("weather_keyframes") or [])
        if weather_keyframes:
            weather_month = weather_profile.get("climate_month")
            require(weather_month is not None,
                    "weather profile is missing climate_month")
            for keyframe in weather_keyframes:
                condition = str(keyframe.get("condition", ""))
                require(condition in WEATHER_CONDITIONS,
                        f"unsupported weather condition {condition!r}")
                if condition in WEATHER_CONDITIONS:
                    require(weather_condition_allowed(network, condition),
                            "weather_geography_incompatible: "
                            f"{network} cannot use {condition}")
                    if weather_month is not None:
                        require(weather_condition_month_allowed(
                            network, condition, int(weather_month)),
                            "weather_geography_incompatible: "
                            f"{network} cannot use {condition} in month "
                            f"{weather_month}")
            weather_conditions = [
                str(item.get("condition", ""))
                for item in weather_keyframes]
            require(all(edge in WEATHER_TRANSITION_EDGES for edge in zip(
                weather_conditions, weather_conditions[1:])),
                "weather_transition_not_adjacent")
            weather_times = [
                60.0 * float(item.get("t", -1.0))
                for item in weather_keyframes]
            require(weather_times == sorted(weather_times)
                    and len(weather_times) == len(set(weather_times)),
                    "weather keyframes are not strictly increasing")
            require(all(time_s < float(scenario["total_time_s"])
                        for time_s in weather_times),
                    "weather_keyframe_at_or_after_time_limit")
            temperatures = [
                float(item.get("temperature", 0.0))
                for item in weather_keyframes]
            require(all(abs(second - first) <= 4.0 + 1e-9
                        for first, second in zip(
                            temperatures, temperatures[1:])),
                    "weather_parameter_inconsistent: temperature jump")

        daynight_keyframes = list(
            scenario.get("daynight_keyframes") or [])
        if daynight_keyframes:
            periods = [str(item.get("period", ""))
                       for item in daynight_keyframes]
            require(all(edge in DAYNIGHT_EDGES for edge in zip(
                periods, periods[1:])),
                "daynight_transition_not_forward")
            daynight_times = [
                60.0 * float(item.get("t", -1.0))
                for item in daynight_keyframes]
            require(daynight_times == sorted(daynight_times)
                    and len(daynight_times) == len(set(daynight_times)),
                    "day/night keyframes are not strictly increasing")
            require(all(time_s < float(scenario["total_time_s"])
                        for time_s in daynight_times),
                    "daynight_keyframe_at_or_after_time_limit")

        # Weather and day/night timelines are independent controls. Events
        # may occur close together or at the same timestamp; only each
        # timeline's ordering and its strict placement before the task
        # horizon are validated above.
        try:
            runtime = self.runtime(network)
        except Exception as exc:
            errors.append(f"HD map load failed: {exc}")
            runtime = None

        vehicles = {
            str(item["vehicle_id"]): item
            for item in scenario.get("vehicles", [])}
        pedestrians = {
            str(item["ped_id"]): item
            for item in scenario.get("pedestrians", [])}
        require(len(vehicles) == len(scenario.get("vehicles", [])),
                "duplicate vehicle_id")
        require(len(pedestrians) == len(scenario.get("pedestrians", [])),
                "duplicate ped_id")
        require(not (set(vehicles) & set(pedestrians)),
                "vehicle and pedestrian IDs overlap")

        if runtime:
            for entity_id, vehicle in vehicles.items():
                state = vehicle.get("initial_physical_state", {})
                if vehicle.get("agent_config", {}).get("type") == "sumo":
                    require(not {"target_speed_kmh", "desired_speed_kmh"}.intersection(state),
                            f"{entity_id}: native SUMO NPC accepts initial speed_kmh, "
                            "not persistent target_speed_kmh/desired_speed_kmh hints")
                lane_id = state.get("lane_id")
                connector_id = state.get("connector_id")
                require(bool(lane_id) ^ bool(connector_id),
                        f"{entity_id}: needs exactly one lane_id/connector_id")
                require(lane_id is None or lane_id in runtime._lane_by_id,
                        f"{entity_id}: unknown lane_id {lane_id}")
                require(connector_id is None
                        or connector_id in runtime._connector_by_id,
                        f"{entity_id}: unknown connector_id {connector_id}")
                route_actions = state.get("lane_route_actions", [])
                require(isinstance(route_actions, list),
                        f"{entity_id}: lane_route_actions is not a list")
                for action in route_actions:
                    action_connector = action.get("connector_id")
                    require(action.get("type") in ("connector", "lane_change"),
                            f"{entity_id}: unknown route action type")
                    require(action_connector is None
                            or action_connector in runtime._connector_by_id,
                            f"{entity_id}: unknown route connector "
                            f"{action_connector}")
            for entity_id, pedestrian in pedestrians.items():
                state = pedestrian.get("initial_physical_state", {})
                crosswalk_id = state.get("crosswalk_id")
                require(bool(crosswalk_id),
                        f"{entity_id}: needs one mapped crosswalk_id")
                require(crosswalk_id in runtime._crosswalk_by_id,
                        f"{entity_id}: unknown crosswalk_id {crosswalk_id}")

            for assertion in expected.get("setup_assertions", []):
                try:
                    self._assert_setup(
                        assertion, vehicles, pedestrians, runtime, require,
                        scenario)
                except Exception as exc:
                    errors.append(
                        f"assertion {assertion.get('kind')} failed: {exc}")

        if errors:
            raise SceneValidationError(
                f"{scene_id}: " + "; ".join(errors))
        return {
            "scene_id": scene_id,
            "network": network,
            "vehicles": len(vehicles),
            "pedestrians": len(pedestrians),
            "assertions": len(expected.get("setup_assertions", [])),
        }

    def _assert_setup(self, assertion: Mapping[str, Any], vehicles: dict,
                      pedestrians: dict, runtime: LaneGeometryRuntime,
                      require, scenario: Mapping[str, Any]) -> None:
        kind = assertion["kind"]

        def state(entity_id: str) -> dict:
            return vehicles[entity_id]["initial_physical_state"]

        def connector(entity_id: str) -> dict:
            entity_state = state(entity_id)
            connector_id = entity_state.get("connector_id")
            if connector_id is None:
                connector_id = next(
                    action["connector_id"]
                    for action in entity_state.get("lane_route_actions", [])
                    if action.get("type") == "connector")
            return runtime._connector_by_id[connector_id]

        if kind == "entity_count":
            require(len(vehicles) == assertion["vehicles"],
                    "vehicle count mismatch")
            require(len(pedestrians) == assertion["pedestrians"],
                    "pedestrian count mismatch")
        elif kind == "related_background_traffic":
            focal_id = str(assertion["entity"])
            cohort_ids = list(assertion[
                "longitudinal_cohort_vehicle_ids"])
            route_ids = list(assertion["route_aligned_vehicle_ids"])
            crossing_ids = list(assertion["crossing_vehicle_ids"])
            related_ids = route_ids + crossing_ids
            require(
                len(related_ids) >= int(assertion["minimum_count"])
                and len(set(related_ids)) == len(related_ids),
                "related background traffic is below its distinct-vehicle floor")
            require(
                int(assertion["longitudinal_cohort_minimum_count"])
                <= len(cohort_ids)
                <= int(assertion["longitudinal_cohort_maximum_count"])
                and set(cohort_ids).issubset(route_ids),
                "longitudinal cohort is outside its 3--8 vehicle contract")
            route_lanes = [
                runtime._lane_by_id[lane_id]
                for lane_id in assertion["focal_route_lane_ids"]]
            route_segment_ids = {
                lane["segment_id"] for lane in route_lanes}
            focal_connectors = set(assertion["focal_connector_ids"])
            require(focal_id in vehicles and route_segment_ids,
                    "related traffic has no focal route")
            for entity_id in related_ids:
                vehicle = vehicles.get(entity_id)
                require(vehicle is not None,
                        f"related background vehicle {entity_id} is missing")
                require(
                    not vehicle.get("is_evaluated")
                    and vehicle.get("agent_config", {}).get("type") == "sumo"
                    and vehicle.get("initial_node")
                    != vehicle.get("destination_node"),
                    f"{entity_id} is not a non-focal SUMO trip vehicle")
            for entity_id in route_ids:
                lane = runtime._lane_by_id[state(entity_id)["lane_id"]]
                require(
                    lane["segment_id"] in route_segment_ids,
                    f"{entity_id} is not on the focal route or an adjacent lane")
            anchor_lane = runtime._lane_by_id[
                assertion["longitudinal_cohort_anchor_lane_id"]]
            focal_state = state(focal_id)
            if focal_state.get("lane_id"):
                focal_progress = float(focal_state.get("progress", 0.0))
            else:
                focal_progress = 0.04
            maximum_offset_m = float(assertion[
                "longitudinal_cohort_maximum_offset_m"])
            recorded_offsets = assertion[
                "longitudinal_cohort_initial_route_offsets_m"]
            recorded_destinations = assertion[
                "longitudinal_cohort_destination_nodes"]
            require(
                set(recorded_offsets) == set(cohort_ids)
                and set(recorded_destinations) == set(cohort_ids),
                "longitudinal cohort evidence is incomplete")
            for entity_id in cohort_ids:
                vehicle = vehicles[entity_id]
                lane = runtime._lane_by_id[state(entity_id)["lane_id"]]
                recorded_offset_m = float(recorded_offsets[entity_id])
                is_anchor_sibling = (
                    lane["segment_id"] == anchor_lane["segment_id"]
                    and lane["direction"] == anchor_lane["direction"])
                downstream_lanes = route_lanes[1:2]
                is_immediate_downstream = bool(
                    downstream_lanes
                    and lane["segment_id"]
                    == downstream_lanes[0]["segment_id"]
                    and lane["direction"]
                    == downstream_lanes[0]["direction"])
                require(
                    is_anchor_sibling or is_immediate_downstream,
                    f"{entity_id} is not on the focal start corridor")
                require(
                    abs(recorded_offset_m) <= maximum_offset_m + 1e-6,
                    f"{entity_id} starts outside the focal cohort radius")
                if is_anchor_sibling:
                    measured_offset_m = (
                        float(state(entity_id)["progress"])
                        * float(lane["length_m"])
                        - focal_progress * float(lane["length_m"]))
                    require(
                        abs(measured_offset_m - recorded_offset_m) <= 1e-3,
                        f"{entity_id} cohort offset evidence is inconsistent")
                require(
                    vehicle["destination_node"]
                    == recorded_destinations[entity_id],
                    f"{entity_id} destination evidence is inconsistent")
                require(
                    vehicle["destination_node"]
                    == vehicles[focal_id]["destination_node"]
                    or (bool(scenario.get("pedestrians"))
                        and vehicle["destination_node"]
                        == lane["end_node"]),
                    f"{entity_id} does not share the focal destination "
                    "or pedestrian approach")
            for entity_id in crossing_ids:
                selected = connector(entity_id)
                require(any(
                    selected["id"] in runtime._connector_conflicts.get(
                        focal_connector_id, set())
                    for focal_connector_id in focal_connectors),
                    f"{entity_id} does not cross the focal connector in path topology")
            metadata = scenario.get("experiment_scene", {}).get(
                "related_background_traffic", {})
            require(
                metadata.get("route_aligned_vehicle_ids") == route_ids
                and metadata.get("crossing_vehicle_ids") == crossing_ids,
                "related background metadata differs from its assertion")
            require(
                metadata.get("longitudinal_cohort_vehicle_ids")
                == cohort_ids
                and metadata.get(
                    "longitudinal_cohort_initial_route_offsets_m")
                == recorded_offsets
                and metadata.get("longitudinal_cohort_destination_nodes")
                == recorded_destinations,
                "longitudinal cohort metadata differs from its assertion")
        elif kind == "same_lane":
            lanes = [state(entity)["lane_id"]
                     for entity in assertion["entities"]]
            require(len(set(lanes)) == 1, "entities are not on the same lane")
        elif kind == "initial_longitudinal_gap_m":
            rear = state(assertion["rear"])
            front = state(assertion["front"])
            lane = runtime._lane_by_id[rear["lane_id"]]
            gap = (front["progress"] - rear["progress"]) * lane["length_m"]
            require(0.0 < gap <= assertion["maximum"],
                    f"longitudinal gap {gap:.3f} m outside expected range")
        elif kind == "connector_conflict":
            first, second = [connector(entity)
                             for entity in assertion["entities"]]
            require(second["id"] in runtime._connector_conflicts.get(
                first["id"], set()), "connectors do not conflict")
        elif kind == "signal_control":
            entities = assertion.get("entities") or [
                entity for entity in vehicles
                if ("connector_id" in state(entity)
                    or any(action.get("type") == "connector"
                           for action in state(entity).get(
                               "lane_route_actions", [])))]
            values = [bool(connector(entity).get("signal_controlled"))
                      for entity in entities]
            require(bool(values), "signal assertion has no connector entities")
            require(any(values) == assertion["value"],
                    "signal-control classification mismatch")
        elif kind == "shared_opposing_corridor":
            first, second = [
                runtime._lane_by_id[state(entity)["lane_id"]]
                for entity in assertion["entities"]]
            require(first["segment_id"] == second["segment_id"]
                    and first["direction"] != second["direction"]
                    and first.get("shared_bidirectional")
                    and second.get("shared_bidirectional"),
                    "lanes are not one shared opposing corridor")
        elif kind == "narrow_shared_corridor_sequence":
            ego_id = assertion["entity"]
            peer_id = assertion["peer"]
            ego_state = state(ego_id)
            peer_state = state(peer_id)
            ego_lane = runtime._lane_by_id[ego_state["lane_id"]]
            require(ego_lane["id"] == assertion["ego_approach_lane_id"],
                    "ego narrow-road approach lane changed")
            require(not ego_lane.get("shared_bidirectional"),
                    "ego must start outside the shared corridor")
            approach_lanes = [
                lane for lane in runtime._lane_by_id.values()
                if lane["segment_id"] == ego_lane["segment_id"]]
            require(len(approach_lanes) == 2
                    and {lane["direction"] for lane in approach_lanes}
                    == {"forward", "backward"}
                    and not any(lane.get("shared_bidirectional")
                                for lane in approach_lanes),
                    "ego approach must be an ordinary two-way two-lane road")
            actual_setback = (
                (1.0 - float(ego_state["progress"]))
                * float(ego_lane["length_m"]))
            require(abs(actual_setback - assertion["ego_setback_m"]) < .01,
                    "ego narrow-road setback changed")

            def validate_route(entity_id: str, connector_ids: list[str],
                               initial_lane_id: str) -> list[dict]:
                actual = state(entity_id).get("lane_route_actions", [])
                require(actual == [
                    {"type": "connector", "connector_id": connector_id}
                    for connector_id in connector_ids],
                    f"{entity_id} narrow-road route changed")
                cursor = initial_lane_id
                route_lanes = [runtime._lane_by_id[cursor]]
                for connector_id in connector_ids:
                    selected = runtime._connector_by_id[connector_id]
                    require(selected["from_lane"] == cursor,
                            f"{entity_id} narrow-road route is discontinuous")
                    require(not selected.get("signal_controlled"),
                            "narrow-road sequence must be unsignalized")
                    cursor = selected["to_lane"]
                    route_lanes.append(runtime._lane_by_id[cursor])
                return route_lanes

            ego_route = validate_route(
                ego_id, list(assertion["ego_route_connector_ids"]),
                ego_lane["id"])
            corridor_segments = list(assertion["corridor_segment_ids"])
            ego_shared = [
                lane for lane in ego_route
                if lane.get("shared_bidirectional")]
            require([lane["segment_id"] for lane in ego_shared]
                    == corridor_segments,
                    "ego route does not traverse the complete shared corridor")
            require(all(lane["direction"] == assertion["ego_direction"]
                        for lane in ego_shared),
                    "ego shared-corridor direction changed")
            require(all(abs(float(lane["width_m"])
                            - float(assertion["physical_width_m"])) < .01
                        for lane in ego_shared),
                    "shared corridor is not one authored vehicle width")

            require(peer_state["lane_id"]
                    == assertion["peer_initial_lane_id"],
                    "oncoming initial shared lane changed")
            require(abs(float(peer_state["progress"])
                        - float(assertion["peer_initial_progress"])) < 1e-6,
                    "oncoming initial corridor progress changed")
            peer_lane = runtime._lane_by_id[peer_state["lane_id"]]
            require(peer_lane.get("shared_bidirectional")
                    and peer_lane["segment_id"] in corridor_segments
                    and peer_lane["direction"] != assertion["ego_direction"],
                    "oncoming vehicle is not committed in the opposing corridor")
            paired_ego_lane = next(
                lane for lane in ego_shared
                if lane["segment_id"] == peer_lane["segment_id"])
            require(paired_ego_lane["centerline_xy"]
                    == list(reversed(peer_lane["centerline_xy"])),
                    "opposing logical lanes do not share one physical line")
            peer_entry = runtime._connector_by_id[
                assertion["peer_entry_connector_id"]]
            require(not peer_entry.get("signal_controlled")
                    and runtime._lane_by_id[
                        peer_entry["to_lane"]].get("shared_bidirectional"),
                    "oncoming entry into the corridor is not unsignalized")
            peer_route = validate_route(
                peer_id, list(assertion["peer_route_connector_ids"]),
                peer_lane["id"])
            require(not peer_route[-1].get("shared_bidirectional"),
                    "oncoming route must leave the shared corridor")
            require(bool(vehicles[ego_id].get("is_evaluated"))
                    and bool(vehicles[peer_id].get("is_evaluated"))
                    == bool(assertion.get("peer_evaluated", False)),
                    "narrow-road evaluated-vehicle assignment changed")
        elif kind == "narrow_oncoming_platoon":
            follower_id = assertion["follower"]
            follower = state(follower_id)
            lane = runtime._lane_by_id[follower["lane_id"]]
            require(lane["id"] == assertion["follower_approach_lane_id"]
                    and not lane.get("shared_bidirectional"),
                    "narrow-road follower must start outside the corridor")
            actual_setback = ((1.0 - float(follower["progress"]))
                              * float(lane["length_m"]))
            require(abs(actual_setback
                        - float(assertion["follower_setback_m"])) < .01,
                    "narrow-road follower setback changed")
            connector_ids = list(assertion["follower_route_connector_ids"])
            require(follower.get("lane_route_actions") == [
                {"type": "connector", "connector_id": connector_id}
                for connector_id in connector_ids],
                "narrow-road follower route changed")
            cursor = lane["id"]
            entered_shared = False
            for connector_id in connector_ids:
                selected = runtime._connector_by_id[connector_id]
                require(selected["from_lane"] == cursor
                        and not selected.get("signal_controlled"),
                        "narrow-road follower route is discontinuous or signaled")
                cursor = selected["to_lane"]
                entered_shared = entered_shared or bool(
                    runtime._lane_by_id[cursor].get("shared_bidirectional"))
            require(entered_shared
                    and not runtime._lane_by_id[cursor].get(
                        "shared_bidirectional"),
                    "narrow-road follower must traverse and leave the corridor")
            require(bool(vehicles[follower_id].get("is_evaluated"))
                    == bool(assertion.get("peer_evaluated", False)),
                    "narrow-road follower evaluation assignment changed")
        elif kind == "initial_signal":
            signal = runtime.signal_state(assertion["connector_id"], 0.0)
            require(signal is not None and signal.signal == assertion["value"],
                    "initial signal state mismatch")
        elif kind == "bounded_red_light_approach":
            entity_id = assertion["entity"]
            placement = state(entity_id)
            connector_id = assertion["connector_id"]
            selected = runtime._connector_by_id[connector_id]
            lane = runtime._lane_by_id[placement["lane_id"]]
            target = runtime._lane_by_id[selected["to_lane"]]
            signal = runtime.signal_state(connector_id, 0.0)
            require(selected["from_lane"] == lane["id"],
                    "red-light approach does not feed the selected connector")
            require(selected["turn"] == assertion["turn"],
                    "red-light movement turn changed")
            require(placement.get("lane_route_actions") == [{
                "type": "connector", "connector_id": connector_id,
            }], "red-light route must contain exactly the selected movement")
            require(signal is not None and signal.signal == "red",
                    "bounded red-light approach is not initially red")
            if signal is not None:
                require(
                    float(assertion["minimum_red_remaining_s"])
                    <= float(signal.remaining_seconds)
                    <= float(assertion["maximum_red_remaining_s"]),
                    "initial residual red time is outside the bounded window")
            approach_distance = (
                runtime.stop_progress(lane["id"])
                - float(placement["progress"])) * float(lane["length_m"])
            require(
                float(assertion["minimum_approach_distance_m"])
                <= approach_distance
                <= float(assertion["maximum_approach_distance_m"]),
                "red-light approach distance is outside the bounded window")
            speed = float(placement.get("speed_kmh", -1.0))
            require(
                float(assertion["minimum_initial_speed_kmh"])
                <= speed <= float(assertion["maximum_initial_speed_kmh"]),
                "red-light initial speed is outside the bounded window")
            require(
                float(target["length_m"])
                <= float(assertion["maximum_exit_lane_length_m"]),
                "red-light exit lane is too long for an isolated task")
            require(vehicles[entity_id]["destination_node"]
                    == target["end_node"],
                    "red-light destination must end after the selected movement")
        elif kind == "same_target_lane":
            targets = [connector(entity)["to_lane"]
                       for entity in assertion["entities"]]
            require(len(set(targets)) == 1,
                    "merge connectors have different targets")
        elif kind == "merge_approach_arrival_window":
            entities = list(assertion["entities"])
            setbacks = list(assertion["setbacks_m"])
            require(len(entities) == len(setbacks) == 2,
                    "merge arrival assertion needs two vehicles")
            arrival_times = []
            for entity_id, authored_setback in zip(entities, setbacks):
                placement = state(entity_id)
                lane = runtime._lane_by_id[placement["lane_id"]]
                actual_setback = (
                    (1.0 - float(placement["progress"]))
                    * float(lane["length_m"]))
                require(abs(actual_setback - float(authored_setback)) < .01,
                        "merge approach setback changed")
                speed = float(placement["speed_kmh"]) / 3.6
                require(speed > 0.0, "merge approach must be moving")
                arrival_times.append(actual_setback / max(speed, 1e-6))
            require(max(arrival_times) - min(arrival_times)
                    <= float(assertion["maximum_approach_ttc_gap_s"]),
                    "merge approaches no longer reach together")
        elif kind in {"ordinary_oncoming_stream", "native_oncoming_stream"}:
            ego_id = assertion["entity"]
            peer_ids = list(assertion["peer_ids"])
            ego_lane = runtime._lane_by_id[assertion["ego_lane_id"]]
            opposing_lane = runtime._lane_by_id[
                assertion["opposing_lane_id"]]
            require(state(ego_id).get("lane_id") == ego_lane["id"],
                    "ego oncoming-stream lane changed")
            require(ego_lane["segment_id"] == opposing_lane["segment_id"]
                    and ego_lane["direction"] != opposing_lane["direction"]
                    and not ego_lane.get("shared_bidirectional")
                    and not opposing_lane.get("shared_bidirectional"),
                    "oncoming stream is not on an ordinary two-way road")
            require(min(float(ego_lane["length_m"]),
                        float(opposing_lane["length_m"]))
                    >= float(assertion["minimum_road_length_m"]),
                    "oncoming-stream road is too short")
            progresses = [float(value)
                          for value in assertion["peer_progresses"]]
            require(len(peer_ids) == len(progresses) == 3,
                    "oncoming stream needs exactly three peers")
            for peer_id, progress in zip(peer_ids, progresses):
                placement = state(peer_id)
                require(placement.get("lane_id") == opposing_lane["id"]
                        and abs(float(placement["progress"])
                                - progress) < 1e-6,
                        "oncoming peer placement changed")
                require(float(placement.get("speed_kmh", 0.0)) > 0.0,
                        "oncoming peer must initially move")
                if kind == "ordinary_oncoming_stream":
                    require(placement.get("native_lane_change_enabled") is False,
                            "legacy authored peer must be pinned")
                require(not vehicles[peer_id].get("is_evaluated"),
                        "Basic oncoming peers must remain SUMO vehicles")
        elif kind == "unprotected_left_turn_across_path":
            left_id = assertion["left_turn_entity"]
            through_ids = list(assertion["through_entities"])
            left = connector(left_id)
            through_connectors = [connector(entity_id)
                                  for entity_id in through_ids]
            through = through_connectors[0]
            require(left["id"] == assertion["left_connector_id"]
                    and all(item["id"]
                            == assertion["through_connector_id"]
                            for item in through_connectors),
                    "left-turn connector assignment changed")
            require(left["turn"] == "left"
                    and all(item["turn"] == "straight"
                            for item in through_connectors),
                    "left-turn-across-path movement types changed")
            require(not left.get("signal_controlled")
                    and all(not item.get("signal_controlled")
                            for item in through_connectors)
                    and runtime.signal_state(left["id"], 0.0) is None
                    and runtime.signal_state(through["id"], 0.0) is None,
                    "left-turn-across-path task must be unsignalized")
            require(through["id"] in runtime._connector_conflicts.get(
                left["id"], set()), "left and through paths do not conflict")
            left_lane = runtime._lane_by_id[left["from_lane"]]
            through_lane = runtime._lane_by_id[through["from_lane"]]
            heading_difference = Topology.heading_difference_deg(
                Topology.approach_heading(left_lane),
                Topology.approach_heading(through_lane))
            require(heading_difference
                    >= float(assertion["minimum_opposing_heading_deg"]),
                    "through vehicle is not opposing the left turn")
            ttc = {}
            for entity_id, selected, other in [
                    (left_id, left, through),
                    *[(entity_id, selected, left)
                      for entity_id, selected in zip(
                          through_ids, through_connectors)]]:
                placement = state(entity_id)
                lane = runtime._lane_by_id[selected["from_lane"]]
                setback = ((1.0 - float(placement["progress"]))
                           * float(lane["length_m"]))
                conflict_s = next(
                    item["self_distance_s_m"]
                    for item in runtime.connector_conflict_points(
                        selected["id"])
                    if item["other_connector_id"] == other["id"])
                ttc[entity_id] = (
                    setback + float(conflict_s)) / (
                        float(placement["speed_kmh"]) / 3.6)
            require(abs(ttc[left_id]
                        - float(assertion["left_initial_ttc_s"])) < .02
                    and all(abs(ttc[entity_id] - float(authored)) < .02
                            for entity_id, authored in zip(
                                through_ids,
                                assertion["through_initial_ttc_s"])),
                    "authored left-turn TTC changed")
            require(0.0 < ttc[left_id] - ttc[through_ids[0]]
                    <= float(assertion["maximum_ttc_gap_s"]),
                    "through vehicle no longer has the authored priority window")
            if len(through_ids) > 1:
                require(max(ttc[entity_id] for entity_id in through_ids)
                        - min(ttc[entity_id] for entity_id in through_ids)
                        <= float(assertion["maximum_through_headway_s"]),
                        "opposing through stream headway is too large")
            study = scenario.get("experiment_scene", {}).get("study")
            evaluated = {
                entity_id for entity_id in (left_id, *through_ids)
                if vehicles[entity_id].get("is_evaluated")}
            require(
                evaluated == ({left_id, *through_ids}
                              if study == "multi_llm" else {"ego"}),
                "left-turn evaluated-vehicle assignment changed")
        elif kind == "platoon_pressure_setup":
            ego_id = assertion["entity"]
            lead_id, rear_id = assertion["lead"], assertion["rear"]
            ego_state = state(ego_id)
            lane = runtime._lane_by_id[ego_state["lane_id"]]
            lead_state, rear_state = state(lead_id), state(rear_id)
            require(lead_state["lane_id"] == ego_state["lane_id"]
                    == rear_state["lane_id"],
                    "platoon vehicles must share one lane")
            lead_gap = ((lead_state["progress"] - ego_state["progress"])
                        * lane["length_m"])
            rear_gap = ((ego_state["progress"] - rear_state["progress"])
                        * lane["length_m"])
            require(abs(lead_gap - float(assertion["lead_gap_m"])) < .01
                    and abs(rear_gap - float(assertion["rear_gap_m"])) < .01,
                    "platoon longitudinal gaps changed")
            target_lane = runtime._lane_by_id[
                assertion["adjacent_lane_id"]]
            require(target_lane["segment_id"] == lane["segment_id"]
                    and target_lane["direction"] == lane["direction"]
                    and abs(target_lane["index"] - lane["index"]) == 1,
                    "platoon side lane is not adjacent")
            require(all(state(entity).get("lane_id") == target_lane["id"]
                        for entity in assertion["side_entities"]),
                    "platoon side-lane occupants moved")
        elif kind == "obstacle_gap_change_setup":
            ego_id = assertion["entity"]
            ego_state = state(ego_id)
            source_lane = runtime._lane_by_id[ego_state["lane_id"]]
            target_lane = runtime._lane_by_id[assertion["target_lane_id"]]
            require(target_lane["segment_id"] == source_lane["segment_id"]
                    and target_lane["direction"] == source_lane["direction"]
                    and abs(target_lane["index"] - source_lane["index"]) == 1,
                    "obstacle bypass target lane is not adjacent")
            obstacle = state(assertion["obstacle"])
            front = state(assertion["front"])
            rear = state(assertion["rear"])
            require(obstacle["lane_id"] == source_lane["id"]
                    and front["lane_id"] == rear["lane_id"]
                    == target_lane["id"],
                    "obstacle-gap lane assignment changed")
            offsets = (
                (obstacle["progress"] - ego_state["progress"])
                * source_lane["length_m"],
                (front["progress"] - ego_state["progress"])
                * source_lane["length_m"],
                (ego_state["progress"] - rear["progress"])
                * source_lane["length_m"],
            )
            authored = (
                assertion["obstacle_gap_m"], assertion["front_offset_m"],
                assertion["rear_offset_m"])
            require(all(abs(float(actual) - float(expected)) < .01
                        for actual, expected in zip(offsets, authored)),
                    "obstacle-gap longitudinal placement changed")
            require(bool(obstacle.get("crashed")),
                    "mandatory obstacle is no longer crashed")
        elif kind == "localized_crossing_stream":
            ego_id = assertion["entity"]
            peers = list(assertion["cross_entities"])
            ego = runtime._connector_by_id[assertion["ego_connector_id"]]
            cross = runtime._connector_by_id[
                assertion["cross_connector_id"]]
            require(not ego.get("signal_controlled")
                    and not cross.get("signal_controlled")
                    and cross["id"] in runtime._connector_conflicts.get(
                        ego["id"], set()),
                    "localized crossing stream is not one unsignalized conflict")
            topology = object.__new__(Topology)
            topology.runtime = runtime
            require(topology.connector_conflict_angle_deg(ego, cross)
                    >= float(assertion["minimum_crossing_angle_deg"]),
                    "localized crossing angle is too small")
            placements = [(ego_id, ego, assertion["ego_setback_m"])]
            placements.extend(zip(
                peers, [cross] * len(peers),
                assertion["cross_setbacks_m"]))
            for entity_id, selected, authored_setback in placements:
                placement = state(entity_id)
                lane = runtime._lane_by_id[selected["from_lane"]]
                actual = ((1.0 - float(placement["progress"]))
                          * float(lane["length_m"]))
                require(placement["lane_id"] == lane["id"]
                        and abs(actual - float(authored_setback)) < .01,
                        "localized crossing-stream placement changed")
                require(placement.get("lane_route_actions") == [{
                    "type": "connector", "connector_id": selected["id"]}],
                    "localized crossing route changed")
            require(len(peers) == 3 and len(set(peers)) == 3,
                    "localized crossing stream needs three peers")
        elif kind == "localized_merge_stream":
            ego_id = assertion["entity"]
            peers = list(assertion["stream_entities"])
            ego = runtime._connector_by_id[assertion["ego_connector_id"]]
            stream = runtime._connector_by_id[
                assertion["stream_connector_id"]]
            require(ego["to_lane"] == stream["to_lane"]
                    and ego["from_lane"] != stream["from_lane"]
                    and not ego.get("signal_controlled")
                    and not stream.get("signal_controlled"),
                    "localized merge does not converge into one lane")
            placements = [(ego_id, ego, assertion["ego_setback_m"])]
            placements.extend(zip(
                peers, [stream] * len(peers),
                assertion["stream_setbacks_m"]))
            for entity_id, selected, authored_setback in placements:
                placement = state(entity_id)
                lane = runtime._lane_by_id[selected["from_lane"]]
                actual = ((1.0 - float(placement["progress"]))
                          * float(lane["length_m"]))
                require(placement["lane_id"] == lane["id"]
                        and abs(actual - float(authored_setback)) < .01,
                        "localized merge-stream placement changed")
                require(placement.get("lane_route_actions") == [{
                    "type": "connector", "connector_id": selected["id"]}],
                    "localized merge route changed")
            expected_count = int(assertion.get(
                "expected_stream_count", 3))
            require(len(peers) == expected_count
                    and len(set(peers)) == expected_count,
                    "localized merge stream peer count changed")
            if assertion.get("all_entities_evaluated"):
                require(all(vehicles[entity_id].get("is_evaluated")
                            for entity_id in [ego_id, *peers]),
                        "all multi-agent merge vehicles must be evaluated")
        elif kind == "same_junction_minimum_conflicts":
            connectors = [connector(entity)
                          for entity in assertion["entities"]]
            require(len({item["node_id"] for item in connectors}) == 1,
                    "four-way entities do not share one junction")
            conflicts = sum(
                second["id"] in runtime._connector_conflicts.get(
                    first["id"], set())
                for index, first in enumerate(connectors)
                for second in connectors[index + 1:])
            require(conflicts >= assertion["minimum_conflicting_pairs"],
                    f"only {conflicts} conflicting connector pairs")
        elif kind == "synchronized_unsignalized_arrivals":
            entities = list(assertion["entities"])
            setbacks = [float(value) for value in assertion["setbacks_m"]]
            speed_kmh = float(assertion["speed_kmh"])
            require(len(entities) == len(setbacks) == 4,
                    "synchronized four-way task needs four approaches")
            actual_times = []
            for entity_id, authored_setback in zip(entities, setbacks):
                placement = state(entity_id)
                selected = connector(entity_id)
                lane = runtime._lane_by_id[selected["from_lane"]]
                actual_setback = ((1.0 - float(placement["progress"]))
                                  * float(lane["length_m"]))
                require(abs(actual_setback - authored_setback) < .01,
                        "synchronized four-way setback changed")
                require(not selected.get("signal_controlled")
                        and abs(float(placement["speed_kmh"])
                                - speed_kmh) < 1e-6,
                        "synchronized four-way approach changed")
                actual_times.append(
                    actual_setback / (speed_kmh / 3.6))
            require(max(actual_times) - min(actual_times)
                    <= float(assertion["maximum_arrival_spread_s"]),
                    "four-way arrival spread is too large")
            require(all(vehicles[entity].get("is_evaluated")
                        for entity in entities),
                    "all synchronized four-way vehicles must be evaluated")
        elif kind == "connector_crosswalk_conflict":
            vehicle_connector = connector(assertion["vehicle"])
            ped_state = pedestrians[assertion["pedestrian"]][
                "initial_physical_state"]
            crosswalk_id = ped_state["crosswalk_id"]
            require(any(
                item["crosswalk_id"] == crosswalk_id
                for item in runtime.connector_crosswalk_points(
                    vehicle_connector["id"])),
                "connector does not conflict with pedestrian crosswalk")
        elif kind == "scheduled_crosswalk_entry":
            selected = connector(assertion["vehicle"])
            starts = assertion["start_times_s"]
            require(len(starts) >= assertion.get("minimum_pedestrian_count", 1),
                    "scheduled crossing group is too small")
            require(len(set(starts.values())) == len(starts), "pedestrians must enter in a staggered stream")
            require(max(starts.values()) - min(starts.values())
                    >= assertion.get("minimum_release_span_s", 0) - 1e-6,
                    "scheduled crowd entry window is too short")
            for ped_id, start in starts.items():
                ped = pedestrians[ped_id]
                placement = ped["initial_physical_state"]
                require(ped.get("agent_config", {}).get("type") == "sumo"
                        and not ped.get("is_evaluated"), "scheduled pedestrian must be native SUMO")
                require(abs(float(ped.get("start_time", 0)) * 60.0 - start) < 1e-6 and start >= .1,
                        "scheduled pedestrian start changed")
                require("crossing_trigger" not in placement and "waiting" not in placement
                        and placement["progress"] == 0, "unsupported pedestrian staging configuration")
                require(crosswalk_signal_relationship(runtime, placement["crosswalk_id"],
                        selected["id"]) == "unsignalized", "crossing must stay unsignalized")
                require(any(hit["crosswalk_id"] == placement["crosswalk_id"]
                            for hit in runtime.connector_crosswalk_points(selected["id"])),
                        "timed crossing no longer intersects the vehicle route")
        elif kind == "separated_two_way_meeting":
            lanes = []
            for entity, placement in assertion["placements"].items():
                actual = state(entity)
                require(actual["lane_id"] == placement["lane_id"]
                        and actual["progress"] == placement["progress"], "meeting placement changed")
                lane = runtime._lane_by_id[actual["lane_id"]]
                lanes.append(lane)
                require(not lane.get("shared_bidirectional") and lane["length_m"] >= 120,
                        "meeting requires a separated native road")
                require(vehicles[entity]["initial_node"] == lane["start_node"]
                        and vehicles[entity]["destination_node"] == lane["end_node"],
                        "meeting endpoints must match the selected directed lane")
            require({lane["segment_id"] for lane in lanes} == {assertion["segment_id"]}
                    and len({lane["direction"] for lane in lanes}) == 2,
                    "meeting needs opposite directions on the same road")
            from simulation.lane_level_runtime import pose_at
            opposite = next(lane for lane in lanes if lane["direction"] != lanes[0]["direction"])
            require(math.dist(pose_at(lanes[0]["centerline_xy"], lanes[0]["length_m"] / 2)[:2],
                              pose_at(opposite["centerline_xy"], opposite["length_m"] / 2)[:2]) >= 2.5,
                    "opposing centerlines overlap")
        elif kind == "unsignalized_pedestrian_ttc":
            vehicle_connector = connector(assertion["vehicle"])
            ped_state = pedestrians[assertion["pedestrian"]]["initial_physical_state"]
            require(not vehicle_connector.get("signal_controlled") and
                    crosswalk_signal_relationship(runtime, ped_state["crosswalk_id"],
                        vehicle_connector["id"]) == "unsignalized",
                    "pedestrian yield site must have no traffic signals")
            require(runtime._crosswalk_by_id[ped_state["crosswalk_id"]]["road_segment_id"]
                    == runtime._lane_by_id[vehicle_connector["from_lane"]]["segment_id"],
                    "pedestrian yield baseline must cross the ego approach road")
            require(ped_state.get("progress") == 0 and ped_state.get("waiting") is True,
                    "pedestrian must start waiting at crossing endpoint")
            require(ped_state.get("crossing_trigger") == {
                "vehicle_id": assertion["vehicle"], "ttc_s": assertion["ttc_s"]},
                "pedestrian must use ego TTC trigger")
        elif kind == "dense_unsignalized_pedestrian_group":
            vehicle_id = assertion["vehicle"]
            pedestrian_ids = list(assertion["pedestrian_ids"])
            selected = connector(vehicle_id)
            crosswalk_id = assertion["crosswalk_id"]
            require(selected["id"] == assertion["connector_id"]
                    and not selected.get("signal_controlled")
                    and runtime.signal_state(selected["id"], 0.0) is None
                    and crosswalk_signal_relationship(runtime,
                        crosswalk_id, selected["id"]) == "unsignalized",
                    "crowded crossing must be genuinely unsignalized")
            require(len(pedestrian_ids)
                    >= int(assertion["minimum_pedestrian_count"])
                    and len(set(pedestrian_ids)) == len(pedestrian_ids),
                    "crowded crossing does not have enough distinct pedestrians")
            thresholds = []
            for pedestrian_id in pedestrian_ids:
                placement = pedestrians[pedestrian_id][
                    "initial_physical_state"]
                trigger = placement.get("crossing_trigger", {})
                require(placement.get("crosswalk_id") == crosswalk_id
                        and float(placement.get("progress", -1.0)) == 0.0
                        and placement.get("waiting") is True,
                        "crowd pedestrian must wait at the authored endpoint")
                require(trigger.get("vehicle_id") == vehicle_id
                        and float(trigger.get("ttc_s", 0.0)) > 0.0,
                        "crowd pedestrian has no valid ego TTC trigger")
                thresholds.append(float(trigger.get("ttc_s", 0.0)))
                require(not pedestrians[pedestrian_id].get("is_evaluated"),
                        "crowd pedestrians must remain SUMO controlled")
            require(len(set(thresholds)) == len(thresholds)
                    and max(thresholds) - min(thresholds)
                    >= float(assertion["minimum_release_span_s"]),
                    "crowd releases are not staggered across the required span")
        elif kind == "middle_signal_queue":
            ego = state(assertion["entity"])
            lane = runtime._lane_by_id[ego["lane_id"]]
            front, rear = state(assertion["front"]), state(assertion["rear"])
            require(front["lane_id"] == ego["lane_id"] == rear["lane_id"], "queue must share a lane")
            require(front["progress"] > ego["progress"] > rear["progress"], "ego must be physically in the middle")
            for a, b in ((front, ego), (ego, rear)):
                require(abs((a["progress"] - b["progress"]) * lane["length_m"] - assertion["spacing_m"]) < 0.01,
                        "queue spacing differs from the authored gap")
            require(sum(l["segment_id"] == lane["segment_id"] and l["direction"] == lane["direction"]
                        for l in runtime._lane_by_id.values()) == 1 and not lane.get("shared_bidirectional"),
                    "queue approach must have one non-shared lane in this direction")
            for entity in (assertion["front"], assertion["entity"], assertion["rear"]):
                placement = state(entity)
                require(placement["speed_kmh"] == 0, "queue must start stopped")
                require(placement.get("lane_route_actions") == [
                    {"type": "connector", "connector_id": assertion["connector_id"]}],
                    "all queue vehicles must take the selected connector")
            require(vehicles[assertion["entity"]].get("is_evaluated") is True
                    and not vehicles[assertion["front"]].get("is_evaluated")
                    and not vehicles[assertion["rear"]].get("is_evaluated"),
                    "only the middle vehicle is evaluated")
        elif kind == "unsignalized_crossing_stream":
            ego_id = assertion["entity"]
            peers = assertion["cross_entities"]
            ego = runtime._connector_by_id[assertion["ego_connector_id"]]
            cross = runtime._connector_by_id[assertion["cross_connector_id"]]
            lane = runtime._lane_by_id[ego["from_lane"]]
            siblings = [l for l in runtime._lane_by_id.values()
                        if (l["segment_id"], l["direction"]) == (lane["segment_id"], lane["direction"])]
            require(len(siblings) == 1 and "shared" not in lane["id"],
                    "ego approach must have one non-shared same-direction lane")
            # Use the same local conflict-tangent geometry as site selection,
            # without loading a second copy of the full map runtime.
            topology = object.__new__(Topology)
            topology.runtime = runtime
            angle = topology.connector_conflict_angle_deg(ego, cross)
            require(angle >= assertion["minimum_crossing_angle_deg"],
                    "crossing stream must have a material crossing angle")
            require(len(peers) == 3 and len(set(peers)) == 3,
                    "challenge needs three distinct crossing vehicles")
            for vid, selected, setback in [
                (ego_id, ego, assertion["ego_setback_m"]),
                *[(vid, cross, distance) for vid, distance in
                  zip(peers, assertion["cross_setbacks_m"])]]:
                placement = state(vid)
                source = runtime._lane_by_id[selected["from_lane"]]
                actual = (1 - placement["progress"]) * source["length_m"]
                require(placement["lane_id"] == selected["from_lane"]
                        and abs(actual - setback) < .01,
                        "crossing stream placement/setback changed")
                require(placement.get("lane_route_actions") == [
                    {"type": "connector", "connector_id": selected["id"]}],
                    "crossing stream must retain the selected route")
                require(vehicles[vid]["destination_node"] ==
                        runtime._lane_by_id[selected["to_lane"]]["end_node"],
                        "crossing destination must be beyond the junction")
                require(bool(vehicles[vid].get("is_evaluated")) == (vid == ego_id),
                        "only ego is evaluated in crossing stream")
        elif kind == "required_lane_change_turn":
            initial = runtime._lane_by_id[state(assertion["entity"])["lane_id"]]
            target = runtime._lane_by_id[assertion["target_lane_id"]]
            turn = runtime._connector_by_id[assertion["connector_id"]]
            outgoing = runtime._connectors_from
            require({c["turn"] for c in outgoing.get(initial["id"], [])} == {"straight"},
                    "initial lane must be straight-only")
            require(initial["segment_id"] == target["segment_id"]
                    and initial["direction"] == target["direction"]
                    and abs(initial["index"] - target["index"]) == 1,
                    "turn lane must be adjacent and same-direction")
            require(turn["from_lane"] == target["id"] and turn["turn"] in {"left", "right"},
                    "target lane must provide the selected turn")
            destination = vehicles[assertion["entity"]]["destination_node"]
            require(runtime._lane_by_id[turn["to_lane"]]["end_node"] == destination,
                    "destination must follow the selected turn")
            actions = state(assertion["entity"]).get("lane_route_actions", [])
            require(len(actions) == 2 and actions[0].get("type") == "lane_change"
                    and actions[0].get("to_lane_id") == target["id"]
                    and actions[1].get("connector_id") == turn["id"],
                    "route must change lanes then take the selected connector")
            reachable, pending = {initial["id"]}, [initial["id"]]
            while pending:
                for edge in outgoing.get(pending.pop(), []):
                    if edge["to_lane"] not in reachable:
                        reachable.add(edge["to_lane"])
                        pending.append(edge["to_lane"])
            require(destination not in {runtime._lane_by_id[l]["end_node"] for l in reachable},
                    "destination is reachable without any lane change")
        elif kind == "adjacent_lane":
            first, second = [
                runtime._lane_by_id[state(entity)["lane_id"]]
                for entity in assertion["entities"]]
            require(first["segment_id"] == second["segment_id"]
                    and first["direction"] == second["direction"]
                    and abs(first["index"] - second["index"]) == 1,
                    "entities are not in adjacent same-direction lanes")
        elif kind == "active_lane_change_overlap":
            changing = state(assertion["changing"])
            occupant = state(assertion["target_occupant"])
            require(changing.get("lane_change_target_lane_id")
                    == occupant.get("lane_id"),
                    "lane-change target is not the occupied lane")
            require(abs(changing["progress"] - occupant["progress"]) < 1e-6,
                    "changing vehicle and target occupant are not aligned")
            require(0.0 < changing.get("lane_change_progress", 0.0) < 1.0,
                    "vehicle is not initialized inside a lane change")
        elif kind == "crashed_lane_obstacle":
            require(bool(state(assertion["entity"]).get("crashed")),
                    "entity is not initialized as crashed")
        elif kind == "minimum_parallel_lanes":
            lane = runtime._lane_by_id[assertion["lane_id"]]
            count = sum(
                item["segment_id"] == lane["segment_id"]
                and item["direction"] == lane["direction"]
                for item in runtime._lane_by_id.values())
            require(count >= assertion["count"],
                    f"only {count} usable parallel lanes")
        elif kind == "minimum_distinct_segments":
            segments = set()
            for entity in vehicles:
                entity_state = state(entity)
                if "lane_id" in entity_state:
                    segments.add(runtime._lane_by_id[
                        entity_state["lane_id"]]["segment_id"])
            require(len(segments) >= assertion["count"],
                    f"only {len(segments)} distinct initial segments")
        elif kind in {
                "nearby_oncoming_stream", "scheduled_oncoming_stream",
                "native_full_network_oncoming_stream"}:
            ego_id = assertion["entity"]
            peer_ids = list(assertion["peer_ids"])
            require(len(peer_ids) >= assertion["minimum_runtime_encounters"]
                    and len(set(peer_ids)) == len(peer_ids),
                    "oncoming stream needs distinct peer vehicles")
            lane = runtime._lane_by_id[assertion["lane_id"]]
            ego_lane = runtime._lane_by_id[state(ego_id)["lane_id"]]
            require(_paths_near(
                ego_lane["centerline_xy"], lane["centerline_xy"],
                float(assertion["maximum_lateral_separation_m"])),
                "oncoming lane is not near the ego route")
            first_dx = ego_lane["centerline_xy"][-1][0] - ego_lane["centerline_xy"][0][0]
            first_dy = ego_lane["centerline_xy"][-1][1] - ego_lane["centerline_xy"][0][1]
            second_dx = lane["centerline_xy"][-1][0] - lane["centerline_xy"][0][0]
            second_dy = lane["centerline_xy"][-1][1] - lane["centerline_xy"][0][1]
            denominator = math.hypot(first_dx, first_dy) * math.hypot(second_dx, second_dy)
            cosine = max(-1.0, min(1.0,
                (first_dx * second_dx + first_dy * second_dy) / denominator))
            require(math.degrees(math.acos(cosine)) >= assertion["minimum_heading_difference_deg"],
                    "authored traffic lane is not opposite to ego")
            progresses = [float(value) for value in assertion["progresses"]]
            require(len(progresses) == len(peer_ids),
                    "oncoming positions do not match peer count")
            for first in range(len(progresses)):
                for second in range(first + 1, len(progresses)):
                    require(abs(progresses[first] - progresses[second]) * lane["length_m"]
                            >= assertion["minimum_initial_spacing_m"],
                            "oncoming vehicles are initially too close")
            events = scenario.get("experiment_world_events", [])
            for peer_id, progress in zip(peer_ids, progresses):
                placement = state(peer_id)
                require(placement.get("lane_id") == lane["id"]
                        and abs(float(placement.get("progress", -1)) - progress) < 1e-6,
                        "oncoming placement changed")
                require(not vehicles[peer_id].get("is_evaluated")
                        and vehicles[peer_id].get("agent_config", {}).get("type") == "sumo",
                        "oncoming vehicle must remain a SUMO NPC")
                matching = [e for e in events
                            if e.get("entity_id") == peer_id]
                if kind == "native_full_network_oncoming_stream":
                    require(abs(float(placement.get("speed_kmh", -1))
                                - float(assertion["release_speed_kmh"])) < 1e-6,
                            "native oncoming speed changed")
                    require(not matching,
                            "native SUMO oncoming vehicle cannot have world events")
                    continue
                require(float(placement.get("speed_kmh", -1)) == 0.0,
                        "staged oncoming vehicle must initially wait")
                if kind == "scheduled_oncoming_stream":
                    require(matching == [
                        {"at_s": 0.0, "entity_id": peer_id,
                         "action": "set_vehicle_speed", "speed_kmh": 0.0},
                        {"at_s": assertion["release_times_s"][peer_id], "entity_id": peer_id,
                         "action": "clear_vehicle_speed_override"}],
                        "scheduled oncoming release event changed")
                    continue
                require(placement.get("native_lane_change_enabled") is False,
                        "authored oncoming vehicle must retain its mapped lane")
                matching = [event for event in events
                            if event.get("entity_id") == peer_id
                            and event.get("action") == "release_vehicle_on_proximity"
                            and event.get("reference_vehicle_id") == ego_id]
                require(len(matching) == 1
                        and float(matching[0].get("at_s", -1)) == 0.0
                        and float(matching[0].get("trigger_distance_m", -1))
                        == float(assertion["trigger_distance_m"])
                        and float(matching[0].get("release_speed_kmh", -1))
                        == float(assertion["release_speed_kmh"]),
                        "oncoming proximity-release event changed")
        elif kind == "compact_turn_route":
            vehicle = vehicles[assertion["entity"]]
            placement = state(assertion["entity"])
            ids = assertion["connector_ids"]
            actions = placement.get("lane_route_actions", [])
            require(actions == [{"type": "connector", "connector_id": cid} for cid in ids],
                    "compact route connector order changed")
            require(len(ids) == 3 and len({cid.rsplit("::", 1)[0] for cid in ids}) == 3,
                    "compact route needs three separate turning junctions")
            connectors = [runtime._connector_by_id[cid] for cid in ids]
            require({c["turn"] for c in connectors} == {"left", "right"},
                    "compact route needs three turns and both directions")
            initial = runtime._lane_by_id[placement["lane_id"]]
            require(abs((1-placement["progress"])*initial["length_m"] - assertion["initial_setback_m"]) < .01,
                    "compact route must start 60 m before first turn")
            cursor = initial["id"]
            for i, connector in enumerate(connectors):
                require(connector["from_lane"] == cursor, "compact route must be lane-continuous")
                cursor = connector["to_lane"]
                lane = runtime._lane_by_id[cursor]
                if i < 2:
                    require(assertion["minimum_gap_m"] <= lane["length_m"] <= assertion["maximum_gap_m"],
                            "compact turn spacing outside bounds")
            for lane_id in [initial["id"], *[c["to_lane"] for c in connectors]]:
                lane = runtime._lane_by_id[lane_id]
                siblings = [l for l in runtime._lane_by_id.values() if
                            (l["segment_id"], l["direction"]) == (lane["segment_id"], lane["direction"])]
                require(len(siblings) == 1 and "shared" not in lane_id,
                        "compact route must use single non-shared directional lanes")
            require(runtime._lane_by_id[cursor]["end_node"] == vehicle["destination_node"],
                    "compact route destination changed")
            plan = runtime.plan_lane_route(vehicle["initial_node"], vehicle["destination_node"],
                                           current_lane_id=initial["id"])
            require(plan is not None and [a.get("connector_id") for a in plan["actions"]] == ids,
                    "replanning must preserve compact turn sequence")
        elif kind == "minimum_route_connectors":
            actions = state(assertion["entity"]).get(
                "lane_route_actions", [])
            connectors = [
                runtime._connector_by_id[action["connector_id"]]
                for action in actions if action.get("type") == "connector"]
            require(len(connectors) >= assertion["count"],
                    f"route has only {len(connectors)} connectors")
            turns = sum(item["turn"] != "straight" for item in connectors)
            require(turns >= assertion.get("minimum_turns", 0),
                    f"route has only {turns} turns")
        elif kind == "scheduled_world_event":
            matching = [
                event for event in scenario.get(
                    "experiment_world_events", [])
                if event.get("entity_id") == assertion["entity"]
                and event.get("action") == assertion["action"]
                and abs(float(event.get("at_s", -1.0))
                        - float(assertion["at_s"])) < 1e-9
            ]
            require(bool(matching), "scheduled world event is missing")
        elif kind == "weather_transition":
            keyframes = list(scenario.get("weather_keyframes", []))
            require(len(keyframes) >= 2,
                    "weather transition needs at least two keyframes")
            initial = keyframes[0] if keyframes else {}
            require(abs(float(initial.get("t", -1.0))) < 1e-9,
                    "weather timeline does not start at zero")
            require(initial.get("condition")
                    == assertion["initial_condition"],
                    "initial weather condition mismatch")
            matching = [
                keyframe for keyframe in keyframes
                if keyframe.get("condition")
                == assertion["target_condition"]
                and abs(60.0 * float(keyframe.get("t", -1.0))
                        - float(assertion["at_s"])) < 1e-9
            ]
            require(bool(matching), "weather transition is missing")
            second_condition = assertion.get("second_condition")
            second_at_s = assertion.get("second_at_s")
            if second_condition is not None:
                second = [
                    keyframe for keyframe in keyframes
                    if keyframe.get("condition") == second_condition
                    and abs(60.0 * float(keyframe.get("t", -1.0))
                            - float(second_at_s)) < 1e-9
                ]
                require(bool(second),
                        "second weather transition is missing")
        elif kind == "daynight_transition":
            keyframes = list(scenario.get("daynight_keyframes", []))
            require(len(keyframes) >= 2,
                    "day/night transition needs at least two keyframes")
            initial = keyframes[0] if keyframes else {}
            require(abs(float(initial.get("t", -1.0))) < 1e-9,
                    "day/night timeline does not start at zero")
            require(initial.get("period") == assertion["initial_period"],
                    "initial day/night period mismatch")
            matching = [
                keyframe for keyframe in keyframes
                if keyframe.get("period") == assertion["target_period"]
                and abs(60.0 * float(keyframe.get("t", -1.0))
                        - float(assertion["at_s"])) < 1e-9
            ]
            require(bool(matching), "day/night transition is missing")
            second_period = assertion.get("second_period")
            second_at_s = assertion.get("second_at_s")
            if second_period is not None:
                second = [
                    keyframe for keyframe in keyframes
                    if keyframe.get("period") == second_period
                    and abs(60.0 * float(keyframe.get("t", -1.0))
                            - float(second_at_s)) < 1e-9
                ]
                require(bool(second),
                        "second day/night transition is missing")
        else:
            raise ValueError(f"unknown setup assertion {kind!r}")


def validate_scene_catalog(catalog_dir: Path, *, source_root: Path | None = None,
                           boot: bool = False,
                           smoke_duration_s: float | None = None) -> dict:
    """Validate every catalog scene; optionally execute SUMO smoke runs."""
    catalog_dir = Path(catalog_dir)
    source_root = source_root or Path(__file__).resolve().parents[2]
    validator = SceneValidator(source_root / "simulation" / "road_networks")
    reports = []
    failures = []
    for entry, scenario_path, expected_path in iter_catalog_scenes(catalog_dir):
        engine = None
        try:
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            report = validator.validate(scenario, expected)
            if boot or smoke_duration_s is not None:
                executable = copy.deepcopy(scenario)
                executable["total_time_s"] = (
                    float(smoke_duration_s)
                    if smoke_duration_s is not None else 0.1)
                executable["tick_interval_s"] = max(
                    0.1, executable["total_time_s"])
                for vehicle in executable.get("vehicles", []):
                    vehicle["agent_config"] = {"type": "sumo"}
                for pedestrian in executable.get("pedestrians", []):
                    pedestrian["agent_config"] = {"type": "sumo"}
                engine = MultiSimEngine(MultiScenario.from_dict(executable))
                result = engine.run({})
                report["executed_s"] = executable["total_time_s"]
                report["collisions"] = len(engine.traffic_mgr.collision_log)
                allow_initial_collision = bool(
                    expected.get("smoke", {}).get(
                        "allow_initial_collision", False))
                initial_collisions = [
                    event for event in engine.traffic_mgr.collision_log
                    if event.time_s <= executable.get("physics_step_s", 0.1)
                    + 1e-9]
                if initial_collisions and not allow_initial_collision:
                    raise SceneValidationError(
                        "unexpected collision during initialization smoke run")
                minimum_collisions = int(expected.get(
                    "smoke", {}).get("minimum_initial_collisions", 0))
                if len(initial_collisions) < minimum_collisions:
                    raise SceneValidationError(
                        f"expected at least {minimum_collisions} initial "
                        f"collisions, observed {len(initial_collisions)}")
                report["completed"] = result is not None
            reports.append(report)
        except Exception as exc:
            failures.append({
                "scene_id": entry.get("scene_id"),
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            # MultiSimEngine owns a complete mutable lane runtime.  Large
            # multi-map catalogs otherwise retain cyclic references until the
            # interpreter decides to collect them and can start swapping.
            engine = None
            gc.collect()
    summary = {
        "schema": CATALOG_SCHEMA,
        "scene_count": len(reports) + len(failures),
        "passed": len(reports),
        "failed": len(failures),
        "failures": failures,
    }
    if failures:
        raise SceneValidationError(json.dumps(summary, ensure_ascii=False))
    return summary
