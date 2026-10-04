"""Offline task redesign for the current engine, never a runtime controller.

SUMO NPCs receive an initial pose, route and speed, then remain under native
SUMO authority. Single-width opposing tasks are explicitly replaced, not
claimed equivalent.
"""
from __future__ import annotations

import copy
import hashlib
import math
from collections import defaultdict

from simulation.lane_level_runtime import pose_at

REVISION = "task-config-native-sumo-v2"

# These five mapped crossings need an explicit fixed-time fallback because the
# compatibility runtime cannot release pedestrians from a live TTC trigger.
# The single-pedestrian values leave one 0.1 s step of margin before the latest
# passing release.  The crowd keeps its original 0.6 s cadence and 4.2 s span,
# shifted to the first collision-free release window.  Keep this exception in
# the authoring layer so partial/full scene regeneration remains stable.
PEDESTRIAN_RELEASE_OVERRIDES_S = {
    "basic_097_crosswalk__guangzhou_tianhe": {
        "pedestrian": 0.4,
    },
    "basic_146_crowded_unsignalized_crosswalk__guangzhou_tianhe": {
        f"pedestrian_{index + 1}": round(4.0 + 0.6 * index, 1)
        for index in range(8)
    },
    "basic_161_crosswalk__expansion__beijing_guomao__site_01": {
        "pedestrian": 0.3,
    },
    "basic_162_crosswalk__expansion__hongkong_central__site_02": {
        "pedestrian": 0.1,
    },
    "basic_168_crosswalk__expansion__wuhan_hankou__site_08": {
        "pedestrian": 0.7,
    },
}


def crosswalk_signal_relationship(runtime, crosswalk_id, connector_id):
    """Task-authoring query; keep it outside the physical runtime."""
    crosswalk = runtime._crosswalk_by_id[crosswalk_id]
    node_id = crosswalk["node_id"]
    plan = runtime._signal_plans.get(runtime._node_to_junction.get(node_id, node_id))
    if plan is None:
        return "unsignalized"
    if runtime._signal_plan_by_connector.get(connector_id) != plan:
        return "unknown"
    phases = [p for p in plan["phases"] if p["green_s"] > 0]
    if not any(p.get("pedestrian_green") for p in phases):
        return "unknown"
    if not any(connector_id in p["connector_ids"] for p in phases):
        return "unknown"
    if any(p.get("pedestrian_green") and connector_id in p["connector_ids"]
           for p in phases):
        return "concurrent"
    return "alternating"


def _ceil_step(seconds):
    return round(max(0.1, math.ceil((seconds - 1e-9) * 10) / 10), 1)


def _pedestrian_start(vehicle, crosswalk_id, threshold, runtime):
    st = vehicle["initial_physical_state"]
    speed = float(st.get("speed_kmh", 0)) / 3.6
    if speed <= 0:
        raise ValueError("Cannot design fixed pedestrian release from a stationary reference")
    lane = runtime._lane_by_id[st["lane_id"]]
    distance = (1 - st["progress"]) * lane["length_m"]
    for action in st.get("lane_route_actions", []):
        if action.get("type") != "connector":
            continue
        cid = action["connector_id"]
        for hit in runtime.connector_crosswalk_points(cid):
            if hit["crosswalk_id"] == crosswalk_id:
                return _ceil_step((distance + hit["connector_distance_s_m"] - 2.3)
                                  / speed - threshold)
        c = runtime._connector_by_id[cid]
        distance += c["length_m"] + runtime._lane_by_id[c["to_lane"]]["length_m"]
    raise ValueError("Crosswalk is not on the authored reference route")


def _separated_meeting(scene, expected, runtime):
    groups = defaultdict(list)
    for lane in runtime._lane_by_id.values():
        if not lane.get("shared_bidirectional") and lane["length_m"] >= 120:
            groups[lane["segment_id"]].append(lane)
    candidates = []
    for segment, lanes in sorted(groups.items()):
        if len(lanes) != 2 or len({lane["direction"] for lane in lanes}) != 2:
            continue  # One real lane per direction; no pinning switch needed.
        first, second = sorted(lanes, key=lambda lane: lane["id"])
        # Reject geometrically overlapping centerlines despite distinct IDs.
        gap = math.dist(pose_at(first["centerline_xy"], first["length_m"] / 2)[:2],
                        pose_at(second["centerline_xy"], second["length_m"] / 2)[:2])
        if gap >= 2.5:
            candidates.append((segment, first, second))
    if not candidates:
        raise ValueError("No native two-way replacement road on this task's map")
    index = int(hashlib.sha256(scene["scenario_id"].encode()).hexdigest()[:8], 16)
    segment, first, second = candidates[index % len(candidates)]
    if index % 2:
        first, second = second, first
    placements = {}
    for i, v in enumerate(scene["vehicles"]):
        lane = first if i == 0 else second
        # Meet away from SUMO's endpoint trimming; keep the opposing platoon separated.
        position = 20.0 if i == 0 else 25.0 + 20.0 * (len(scene["vehicles"]) - 1 - i)
        progress = round(position / lane["length_m"], 6)
        speed = float(v["initial_physical_state"].get("speed_kmh", 0)) or 20.0
        v.update(initial_node=lane["start_node"], destination_node=lane["end_node"],
                 destination_name=f"Task destination {lane['end_node']}", initial_lane=lane["index"])
        v["initial_physical_state"] = {"lane_id": lane["id"], "progress": progress,
                                       "speed_kmh": speed, "lane_route_actions": []}
        if v["agent_config"]["type"] == "llm":
            v["initial_physical_state"].update(target_speed_kmh=speed, desired_speed_kmh=speed)
        placements[v["vehicle_id"]] = {"lane_id": lane["id"], "progress": progress}
    expected["setup_assertions"] = [a for a in expected["setup_assertions"]
                                    if not a["kind"].startswith("narrow_")]
    expected["setup_assertions"].append({"kind": "separated_two_way_meeting",
                                         "segment_id": segment, "placements": placements})
    scene["name"] = "Native SUMO two-way meeting (replaces single-width bottleneck)"
    scene["experiment_scene"]["purpose"] = (
        "Pass opposing traffic on separate mapped lanes. This no longer tests "
        "single-width priority negotiation and is not comparable to that task.")
    scene["experiment_scene"]["runtime_acceptance"] = [
        {"kind": "collision_free", "entities": list(placements)},
        {"kind": "route_arrival", "entities": [v["vehicle_id"] for v in scene["vehicles"]
                                                    if v.get("is_evaluated")]},
    ]


def _preserve_pedestrian_stagger(scene, expected):
    """Shift a crowd together rather than clamp all early releases to t=0."""
    changes = scene.get("experiment_scene", {}).get("task_migration", {}).get("changes", [])
    pedestrians = {p["ped_id"]: p for p in scene.get("pedestrians", [])}
    groups = defaultdict(list)
    for change in changes:
        if change["kind"] == "fixed_pedestrian_spawn":
            ped = pedestrians[change["entity"]]
            groups[(change["original_trigger"]["vehicle_id"],
                    ped["initial_physical_state"]["crosswalk_id"])].append(change)
    for group in groups.values():
        base = min(change["start_time_s"] for change in group)
        largest = max(change["original_trigger"]["ttc_s"] for change in group)
        for change in group:
            start = round(base + largest - change["original_trigger"]["ttc_s"], 1)
            change["start_time_s"] = start
            # MultiScenario pedestrian start_time is in minutes, unlike events.
            pedestrians[change["entity"]]["start_time"] = start / 60.0
    for assertion in expected["setup_assertions"]:
        if assertion["kind"] == "scheduled_crosswalk_entry":
            assertion["start_times_s"] = {pid: round(pedestrians[pid]["start_time"] * 60, 6)
                                           for pid in assertion["start_times_s"]}


def _apply_pedestrian_release_overrides(scene, expected, changes=None):
    """Apply reference-validated fixed releases and update their provenance."""
    overrides = PEDESTRIAN_RELEASE_OVERRIDES_S.get(scene["scenario_id"], {})
    if not overrides:
        return {}
    pedestrians = {p["ped_id"]: p for p in scene.get("pedestrians", [])}
    missing = sorted(set(overrides) - set(pedestrians))
    if missing:
        raise ValueError(
            f"pedestrian release override references missing actors: {missing}")
    change_entries = (
        changes if changes is not None else
        scene.get("experiment_scene", {}).get(
            "task_migration", {}).get("changes", []))
    changes_by_entity = {
        change["entity"]: change for change in change_entries
        if change.get("kind") == "fixed_pedestrian_spawn"
    }
    for ped_id, start_s in overrides.items():
        pedestrians[ped_id]["start_time"] = float(start_s) / 60.0
        change = changes_by_entity.get(ped_id)
        if change is not None:
            change["start_time_s"] = float(start_s)
    for assertion in expected.get("setup_assertions", []):
        if assertion.get("kind") != "scheduled_crosswalk_entry":
            continue
        start_times = assertion.get("start_times_s", {})
        for ped_id, start_s in overrides.items():
            if ped_id in start_times:
                start_times[ped_id] = float(start_s)
    return dict(overrides)


def adapt_scene(scene, expected, runtime):
    """Mutate task JSON only. Idempotent; never used inside a simulation tick."""
    meta = scene.setdefault("experiment_scene", {})
    meta.setdefault("runtime_compatibility", {})["task_config_revision"] = REVISION
    if meta.get("task_migration", {}).get("revision") == REVISION:
        _preserve_pedestrian_stagger(scene, expected)
        _apply_pedestrian_release_overrides(scene, expected)
        if "separated_two_way_meeting" in meta.get("interaction_tags", []):
            meta["interaction_tags"] = [tag for tag in meta["interaction_tags"]
                                        if tag != "mutual_exclusion"]
        return meta["task_migration"]["changes"]
    originals = copy.deepcopy(expected.get("setup_assertions", []))
    changes = []
    vehicles = {v["vehicle_id"]: v for v in scene["vehicles"]}
    fixed_peds = {}
    last_start = {}
    for ped in scene.get("pedestrians", []):
        state = ped["initial_physical_state"]
        trigger = state.pop("crossing_trigger", None)
        if not trigger:
            continue
        if trigger.get("mode", "ttc") != "ttc":
            raise ValueError("Signal-triggered tasks need an explicit static redesign")
        start = _pedestrian_start(vehicles[trigger["vehicle_id"]], state["crosswalk_id"],
                                  float(trigger["ttc_s"]), runtime)
        key = (trigger["vehicle_id"], state["crosswalk_id"])
        start = round(max(start, last_start.get(key, -0.1) + 0.2), 1)
        start = max(float(ped.get("start_time", 0)) * 60.0, start)
        last_start[key] = start
        ped["start_time"] = start / 60.0
        state.pop("waiting", None)
        state.pop("spawned", None)
        fixed_peds[ped["ped_id"]] = start
        changes.append({"kind": "fixed_pedestrian_spawn", "entity": ped["ped_id"],
                        "start_time_s": start, "original_trigger": trigger})
    native_releases = {}
    events = []
    for event in scene.get("experiment_world_events", []):
        entity = str(event.get("entity_id", ""))
        if event["action"] != "release_vehicle_on_proximity":
            if ((vehicles.get(entity, {}).get("agent_config") or {}).get(
                    "type", "sumo") == "sumo"):
                changes.append({
                    "kind": "removed_npc_world_event",
                    "entity": entity,
                    "removed_event": copy.deepcopy(event),
                })
                continue
            events.append(event)
            continue
        speed_kmh = float(event["release_speed_kmh"])
        state = vehicles[entity]["initial_physical_state"]
        state["speed_kmh"] = speed_kmh
        state.pop("target_speed_kmh", None)
        state.pop("desired_speed_kmh", None)
        native_releases[entity] = speed_kmh
        changes.append({
            "kind": "native_sumo_motion",
            "entity": entity,
            "initial_speed_kmh": speed_kmh,
            "removed_event": copy.deepcopy(event),
        })
    if events:
        scene["experiment_world_events"] = events
    else:
        scene.pop("experiment_world_events", None)
    removed_locks = []
    for vehicle in scene["vehicles"]:
        if "native_lane_change_enabled" in vehicle["initial_physical_state"]:
            vehicle["initial_physical_state"].pop("native_lane_change_enabled")
            removed_locks.append(vehicle["vehicle_id"])
    if removed_locks:
        changes.append({"kind": "native_lane_choice", "entities": removed_locks})
    if any(a["kind"].startswith("narrow_") for a in originals):
        _separated_meeting(scene, expected, runtime)
        changes.append({"kind": "replace_single_width_task", "comparison_equivalent": False})
    for assertion in expected["setup_assertions"]:
        kind = assertion["kind"]
        if kind in {"unsignalized_pedestrian_ttc", "dense_unsignalized_pedestrian_group"}:
            ids = ([assertion["pedestrian"]] if "pedestrian" in assertion
                   else assertion["pedestrian_ids"])
            assertion["kind"] = "scheduled_crosswalk_entry"
            assertion["start_times_s"] = {pid: fixed_peds[pid] for pid in ids}
        elif kind == "nearby_oncoming_stream":
            assertion["kind"] = "native_full_network_oncoming_stream"
        elif kind == "ordinary_oncoming_stream":
            assertion["kind"] = "native_oncoming_stream"
    fixed_peds.update(_apply_pedestrian_release_overrides(
        scene, expected, changes=changes))
    if changes:
        meta["task_migration"] = {"revision": REVISION, "changes": changes,
                                  "original_setup_assertions": originals,
                                  "not_equivalent_to_dynamic_tasks": True}
        tags = meta.get("interaction_tags", [])
        tags = [t for t in tags if t not in {"ttc_triggered", "staggered_ttc_release",
                "proximity_released", "shared_bidirectional", "sequence_entry"}]
        tags.append("task_config_adapted")
        if fixed_peds: tags.append("fixed_time_pedestrian_entry")
        if native_releases: tags.append("sumo_native_oncoming")
        if any(c["kind"] == "replace_single_width_task" for c in changes):
            tags = [t for t in tags if "narrow" not in t and "single_width" not in t
                    and t != "mutual_exclusion"]
            tags.append("separated_two_way_meeting")
        meta["interaction_tags"] = tags
        if fixed_peds:
            for check in expected.get("runtime_acceptance", []):
                if check["kind"] == "minimum_pedestrian_releases":
                    check["kind"] = "minimum_pedestrian_spawns"
        if any(c["kind"] == "replace_single_width_task" for c in changes):
            expected["runtime_acceptance"] = copy.deepcopy(meta["runtime_acceptance"])
        if fixed_peds:
            meta["purpose"] = "Fixed-time pedestrian entry; vehicle NPCs remain SUMO-native."
        scene["total_time_s"] = max(scene["total_time_s"],
                                     max([0, *fixed_peds.values()]) + 10)
    _preserve_pedestrian_stagger(scene, expected)
    return changes
