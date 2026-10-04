"""Offline paired interaction scope. Reads trajectories; never drives actors.

The three sets are treatment direct contacts, reference direct contacts, and
causal following-chain exposure in either world. The command-line report
focuses on NPC traffic impact and separate LTER completion/queue metrics.
"""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
from statistics import mean

from evaluation.experiments.reference_protocol import (
    reference_compatibility,
)

# v4 makes the signed paired delta the canonical LTER metric.  The positive
# and benefit decompositions remain in the payload for auditability.
LTER_REVISION = "lter_v4_signed_net_vehicle_seconds"
PAIRED_TRAFFIC_IMPACT_REVISION = "paired_traffic_impact_v3_protocol_hash"
LTER_AFFECTED_DELAY_S = 5.0
LTER_AFFECTED_DELAY_RATIO = 0.10
LTER_AFFECTED_QUEUE_S = 5.0
SAME_LANE_DISTANCE_M = 60.0
ADJACENT_LANE_DISTANCE_M = 30.0
CONNECTOR_APPROACH_DISTANCE_M = 60.0
CONNECTOR_ETA_DIFFERENCE_S = 4.0
FOLLOWING_HEADWAY_S = 3.0
FOLLOWING_CHAIN_S = 2.0
STOPPED_GAP_M = 10.0
MAXIMUM_SAMPLE_GAP_S = 0.5


def active(row):
    return (row.get("present_in_physics_world", True)
            and not row.get("arrived") and not row.get("route_failed"))


def collect_interactions(payload, topology):
    """Collect direct evidence and time-local nearest-leader edges.

    Adjacent lanes follow the map planner's same-segment, same-direction,
    consecutive-index rule. No Euclidean-only relationship is accepted.
    Unobserved route branches are not guessed: at most a frozen explicit
    connector action is used when no later connector was observed.
    """
    lanes = {x["id"]: x for x in topology["lanes"]}
    connectors = {x["id"]: x for x in topology["connectors"]}
    conflicts = {frozenset((x["connector_a"], x["connector_b"]))
                 for x in topology.get("connector_conflicts", [])}
    scenario = payload["variant"]["scenario"]
    vehicles = {v["vehicle_id"]: v for v in scenario["vehicles"]}
    focal = payload["system_evaluation"]["focal_vehicle_id"]
    eligible = set(payload["system_evaluation"]["map_trip_vehicle_ids"])
    frames = defaultdict(dict)
    grouped = defaultdict(list)
    for row in payload["trajectories"]["vehicles"]:
        frames[float(row["time_s"])][row["vehicle_id"]] = row
        grouped[row["vehicle_id"]].append(row)
    upcoming = {}
    for vid, rows in grouped.items():
        next_connector = None
        for row in sorted(rows, key=lambda r: r["time_s"], reverse=True):
            cid = row.get("active_connector_id")
            if cid in connectors:
                next_connector = cid
            lane_id = row.get("current_lane_id")
            chosen = next_connector
            if not chosen or connectors[chosen]["from_lane"] != lane_id:
                actions = vehicles[vid].get("initial_physical_state", {}).get("lane_route_actions", [])
                chosen = next((a.get("connector_id") for a in actions[
                    int(row.get("lane_route_action_index", 0)):]
                    if a.get("connector_id") in connectors
                    and connectors[a["connector_id"]]["from_lane"] == lane_id), None)
            upcoming[(vid, float(row["time_s"]))] = cid if cid in connectors else chosen

    def location(row):
        cid = row.get("active_connector_id")
        if cid in connectors:
            return "connector:" + cid, row["edge_progress"] * connectors[cid].get("length_m", 0)
        lane = lanes.get(row.get("current_lane_id"))
        return (lane["id"], row["edge_progress"] * lane["length_m"]) if lane else (None, 0)

    def approach(row):
        cid = upcoming.get((row["vehicle_id"], float(row["time_s"])))
        if not cid:
            return None
        if row.get("active_connector_id") == cid:
            return cid, 0.0, 0.0
        lane = lanes.get(row.get("current_lane_id"))
        if not lane:
            return None
        distance = max(0, (1-row["edge_progress"]) * lane["length_m"])
        speed = row["speed_kmh"] / 3.6
        # A stopped approach qualifies only near the entrance; no infinite ETA.
        eta = distance / speed if speed >= 0.5 else (0.0 if distance <= 10 else float("inf"))
        return cid, distance, eta

    evidence = {}
    direct = set()
    edges = []
    times = sorted(frames)
    for t, end in zip(times, times[1:]):
        dt = end-t
        if dt <= 0 or dt > MAXIMUM_SAMPLE_GAP_S + 1e-6:
            continue
        world = {v:r for v,r in frames[t].items() if active(r)}
        ego = world.get(focal)
        if ego:
            ek, es = location(ego)
            el = lanes.get(ego.get("current_lane_id"))
            ea = approach(ego)
            for vid in eligible & world.keys():
                row = world[vid]
                key, station = location(row)
                distance = abs(station-es)
                kind = None
                if (key is not None and key == ek
                        and distance <= SAME_LANE_DISTANCE_M):
                    kind = "same_lane"
                lane = lanes.get(row.get("current_lane_id"))
                if (kind is None and el and lane
                        and not ego.get("active_connector_id") and not row.get("active_connector_id")
                        and lane["segment_id"] == el["segment_id"]
                        and lane["direction"] == el["direction"]
                        and lane.get("z_level", 0) == el.get("z_level", 0)
                        and abs(lane["index"]-el["index"]) == 1
                        and distance <= ADJACENT_LANE_DISTANCE_M):
                    kind = "adjacent_lane"
                other = approach(row)
                if kind is None and ea and other:
                    a, da, ta = ea
                    b, db, tb = other
                    conflict = frozenset((a,b)) in conflicts or connectors[a]["to_lane"] == connectors[b]["to_lane"]
                    if (a != b and conflict
                            and max(da, db) <= CONNECTOR_APPROACH_DISTANCE_M
                            and abs(ta - tb) <= CONNECTOR_ETA_DIFFERENCE_S):
                        kind, distance = "connector_conflict_or_merge", max(da,db)
                if kind:
                    item = evidence.setdefault((vid,kind), dict(vehicle_id=vid, relation=kind,
                        first_time_s=t, last_time_s=end, duration_s=0.0, minimum_distance_m=distance))
                    item["last_time_s"] = end
                    item["duration_s"] += dt
                    item["minimum_distance_m"] = min(item["minimum_distance_m"], distance)
                    if item["duration_s"] + 1e-6 >= 1:
                        direct.add(vid)
                        item.setdefault("qualified_at_s", end)
        by_lane = defaultdict(list)
        for vid, row in world.items():
            key, station = location(row)
            if key:
                by_lane[key].append((station, vid, row))
        for occupants in by_lane.values():
            occupants.sort(key=lambda x:(x[0],x[1]))
            for rear, front in zip(occupants, occupants[1:]):
                gap = max(0,front[0]-rear[0]-(rear[2].get("length_m",4.6)+front[2].get("length_m",4.6))/2)
                speed = rear[2]["speed_kmh"]/3.6
                if gap <= (FOLLOWING_HEADWAY_S * speed
                           if speed >= 0.5 else STOPPED_GAP_M):
                    edges.append(dict(start_s=t,end_s=end,leader=front[1],follower=rear[1],gap_m=gap))
    for event in payload.get("collision_log", []):
        pair = {event.get("entity_a"), event.get("entity_b")}
        if focal in pair:
            for vid in (pair-{focal}) & eligible:
                direct.add(vid)
                evidence[(vid,"collision")] = dict(vehicle_id=vid,relation="collision",
                    qualified_at_s=float(event.get("time_s",0)),first_time_s=float(event.get("time_s",0)))
    records = [v for v in evidence.values() if "qualified_at_s" in v]
    return dict(direct_vehicle_ids=sorted(direct), evidence=records, following_edges=edges)


def propagate(world, focal, eligible):
    """Forward-time exposure: a later contact cannot activate an earlier queue."""
    activated = {focal: 0.0}
    for item in world["evidence"]:
        vid = item["vehicle_id"]
        activated[vid] = min(activated.get(vid,float("inf")),item["qualified_at_s"])
    runs = {}
    evidence = []
    for edge in sorted(world["following_edges"],key=lambda e:e["start_s"]):
        leader, follower = edge["leader"],edge["follower"]
        if follower not in eligible or activated.get(leader,float("inf")) > edge["start_s"]+1e-6:
            continue
        key = leader,follower
        start,last = runs.get(key,(edge["start_s"],edge["start_s"]))
        if abs(last-edge["start_s"]) > 1e-6:
            start = edge["start_s"]
        runs[key] = start,edge["end_s"]
        if (edge["end_s"] - start + 1e-6 >= FOLLOWING_CHAIN_S
                and follower not in activated):
            activated[follower] = edge["end_s"]
            evidence.append(dict(vehicle_id=follower,leader=leader,relation="following_chain",
                first_time_s=start,qualified_at_s=edge["end_s"],gap_m=edge["gap_m"]))
    return evidence


def _percentile(values, quantile):
    """Linear percentile without requiring a large sample or scipy."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def paired_lter(treatment, reference, selected_vehicle_ids):
    """Compute LLM traffic externality on treatment-arrived NPCs.

    The paired scope is the direct/propagated ego-exposure union assembled by
    :func:`paired_traffic_impact`.  A treatment NPC that does not arrive is
    reported separately by the arrival-failure metric and is excluded from
    completion-time deltas.  Queue exposure remains available for every
    selected NPC because a vehicle can spend time in a queue before timing out.
    Completion and queue changes are reported directly in vehicle-seconds as
    signed treatment-minus-reference net deltas. Positive and benefit-only
    decompositions are retained for audit; they are not the primary metric.
    No ego-route normalization or weighted composite is used.
    """
    tm = treatment.get("system_evaluation", {})
    rm = reference.get("system_evaluation", {})
    selected = sorted(set(map(str, selected_vehicle_ids)))
    if not selected:
        return {
            "metric_revision": LTER_REVISION,
            "applicable": False,
            "reason": "no_interaction_related_trip_vehicles",
        }
    treatment_metrics = tm.get("trip_vehicle_metrics")
    reference_metrics = rm.get("trip_vehicle_metrics")
    if not isinstance(treatment_metrics, dict):
        return {"metric_revision": LTER_REVISION, "applicable": False,
                "reason": "treatment_missing_trip_vehicle_metrics"}
    if not isinstance(reference_metrics, dict):
        return {"metric_revision": LTER_REVISION, "applicable": False,
                "reason": "reference_missing_trip_vehicle_metrics"}
    records = {}
    for vehicle_id in selected:
        treatment_record = treatment_metrics.get(vehicle_id)
        reference_record = reference_metrics.get(vehicle_id)
        if not isinstance(treatment_record, dict) or not isinstance(reference_record, dict):
            return {"metric_revision": LTER_REVISION, "applicable": False,
                    "reason": "missing_paired_trip_vehicle_record",
                    "vehicle_id": vehicle_id}
        try:
            values = {
                "treatment_completion": float(treatment_record["completion_time_s"]),
                "reference_completion": float(reference_record["completion_time_s"]),
                "treatment_queue": float(treatment_record["queue_wait_s"]),
                "reference_queue": float(reference_record["queue_wait_s"]),
            }
        except (KeyError, TypeError, ValueError):
            return {"metric_revision": LTER_REVISION, "applicable": False,
                    "reason": "invalid_paired_trip_vehicle_metric",
                    "vehicle_id": vehicle_id}
        if any(not math.isfinite(value) for value in values.values()):
            return {"metric_revision": LTER_REVISION, "applicable": False,
                    "reason": "nonfinite_paired_trip_vehicle_metric",
                    "vehicle_id": vehicle_id}
        records[vehicle_id] = {
            **values,
            "treatment_arrived": bool(treatment_record.get("arrived")),
            "reference_arrived": bool(reference_record.get("arrived")),
        }

    completion_vehicle_ids = [
        vehicle_id for vehicle_id in selected
        if records[vehicle_id]["treatment_arrived"]
    ]
    excluded_unarrived_vehicle_ids = [
        vehicle_id for vehicle_id in selected
        if not records[vehicle_id]["treatment_arrived"]
    ]
    per_vehicle = {}
    net_delay = 0.0
    net_queue = 0.0
    positive_delay = 0.0
    positive_queue_total = 0.0
    benefit_delay = 0.0
    benefit_queue = 0.0
    affected_delays = []
    for vehicle_id in selected:
        record = records[vehicle_id]
        treatment_completion = record["treatment_completion"]
        reference_completion = record["reference_completion"]
        treatment_queue = record["treatment_queue"]
        reference_queue = record["reference_queue"]
        delta_queue = treatment_queue - reference_queue
        positive_queue = max(0.0, delta_queue)
        negative_queue = min(0.0, delta_queue)
        queue_benefit = max(0.0, -delta_queue)
        net_queue += delta_queue
        positive_queue_total += positive_queue
        benefit_queue += queue_benefit
        completion_scored = record["treatment_arrived"]
        if completion_scored:
            delta_completion = treatment_completion - reference_completion
            positive_completion = max(0.0, delta_completion)
            negative_completion = min(0.0, delta_completion)
            completion_benefit = max(0.0, -delta_completion)
            relative_delay = positive_completion / max(reference_completion, 30.0)
            affected = (
                positive_completion >= LTER_AFFECTED_DELAY_S
                or relative_delay >= LTER_AFFECTED_DELAY_RATIO
                or positive_queue >= LTER_AFFECTED_QUEUE_S
            )
            if affected:
                affected_delays.append(positive_completion)
            net_delay += delta_completion
            positive_delay += positive_completion
            benefit_delay += completion_benefit
        else:
            delta_completion = None
            positive_completion = None
            negative_completion = None
            completion_benefit = None
            relative_delay = None
            affected = False
        per_vehicle[vehicle_id] = {
            "treatment_arrived": record["treatment_arrived"],
            "reference_arrived": record["reference_arrived"],
            "completion_time_scored": completion_scored,
            "treatment_completion_time_s": round(treatment_completion, 6),
            "reference_completion_time_s": round(reference_completion, 6),
            "completion_delta_s": (
                round(delta_completion, 6)
                if delta_completion is not None else None),
            "positive_completion_delta_s": (
                round(positive_completion, 6)
                if positive_completion is not None else None),
            # Signed negative deltas and positive benefit magnitudes are kept
            # separately so a net improvement is visible rather than hidden.
            "negative_completion_delta_s": (
                round(negative_completion, 6)
                if negative_completion is not None else None),
            "benefit_completion_delta_s": (
                round(completion_benefit, 6)
                if completion_benefit is not None else None),
            "treatment_queue_wait_s": round(treatment_queue, 6),
            "reference_queue_wait_s": round(reference_queue, 6),
            "queue_delta_s": round(delta_queue, 6),
            "positive_queue_delta_s": round(positive_queue, 6),
            "negative_queue_delta_s": round(negative_queue, 6),
            "benefit_queue_delta_s": round(queue_benefit, 6),
            "relative_delay": (
                round(relative_delay, 6)
                if relative_delay is not None else None),
            "affected": affected,
        }
    affected_count = len(affected_delays)
    return {
        "metric_revision": LTER_REVISION,
        "applicable": True,
        "scope": "interaction_exposure_union",
        "scope_vehicle_ids": selected,
        "vehicle_count": len(selected),
        "completion_vehicle_ids": completion_vehicle_ids,
        "completion_vehicle_count": len(completion_vehicle_ids),
        "excluded_unarrived_vehicle_ids": excluded_unarrived_vehicle_ids,
        "excluded_unarrived_vehicle_count": len(
            excluded_unarrived_vehicle_ids),
        "affected_vehicle_count": affected_count,
        "affected_vehicle_rate": (
            round(affected_count / len(completion_vehicle_ids), 6)
            if completion_vehicle_ids else None),
        # Canonical metrics: signed treatment-reference deltas.  Positive
        # values mean extra delay/queue; negative values mean improvement.
        "net_completion_delta_vehicle_s": round(net_delay, 6),
        "mean_net_completion_delta_s": (
            round(net_delay / len(completion_vehicle_ids), 6)
            if completion_vehicle_ids else None),
        "net_queue_delta_vehicle_s": round(net_queue, 6),
        "mean_net_queue_delta_s": (
            round(net_queue / len(selected), 6) if selected else None),
        # Decompositions are secondary audit fields, retained for diagnosing
        # whether a net result is driven by harms or improvements.
        "positive_completion_delta_vehicle_s": round(positive_delay, 6),
        "benefit_completion_delta_vehicle_s": round(benefit_delay, 6),
        "positive_queue_delta_vehicle_s": round(positive_queue_total, 6),
        "benefit_queue_delta_vehicle_s": round(benefit_queue, 6),
        "affected_excess_delay_p90_s": round(
            _percentile(affected_delays, 0.90), 6)
            if affected_delays else None,
        "affected_excess_delay_max_s": round(max(affected_delays), 6)
            if affected_delays else None,
        "thresholds": {
            "affected_delay_s": LTER_AFFECTED_DELAY_S,
            "affected_delay_ratio": LTER_AFFECTED_DELAY_RATIO,
            "affected_queue_s": LTER_AFFECTED_QUEUE_S,
        },
        "per_vehicle": per_vehicle,
    }


def paired_traffic_impact(treatment, reference, topology):
    """Return traffic-impact metrics without constructing a composite score.

    NPC collisions and non-arrivals use the complete non-focal trip-vehicle
    set.  LTER and its queue decomposition use the smaller direct/propagated
    interaction scope, so unrelated NPCs cannot dilute the result.
    """
    compatibility = reference_compatibility(treatment, reference)
    if not compatibility["applicable"]:
        return {
            "metric_revision": PAIRED_TRAFFIC_IMPACT_REVISION,
            **compatibility,
        }
    tm = dict(treatment["system_evaluation"])
    rm = dict(reference["system_evaluation"])
    treatment_npc_ids = set(map(str, tm.get("map_trip_vehicle_ids", [])))
    reference_npc_ids = set(map(str, rm.get("map_trip_vehicle_ids", [])))
    if treatment_npc_ids != reference_npc_ids:
        return {
            "metric_revision": PAIRED_TRAFFIC_IMPACT_REVISION,
            "applicable": False,
            "reason": "trip_vehicle_set_mismatch",
            "reference_compatibility": compatibility,
        }
    npc_ids = sorted(treatment_npc_ids)
    treatment_trip_metrics = tm.get("trip_vehicle_metrics") or {}
    reference_trip_metrics = rm.get("trip_vehicle_metrics") or {}
    if (not isinstance(treatment_trip_metrics, dict)
            or not isinstance(reference_trip_metrics, dict)
            or any(vehicle_id not in treatment_trip_metrics
                   or vehicle_id not in reference_trip_metrics
                   for vehicle_id in npc_ids)):
        return {
            "metric_revision": PAIRED_TRAFFIC_IMPACT_REVISION,
            "applicable": False,
            "reason": "missing_paired_trip_vehicle_record",
            "reference_compatibility": compatibility,
        }

    treatment_collision_log = treatment.get("collision_log", []) or []
    npc_collision_events = [
        event for event in treatment_collision_log
        if (event.get("entity_a") in treatment_npc_ids
            or event.get("entity_b") in treatment_npc_ids)
    ]
    npc_not_arrived_ids = [
        vehicle_id for vehicle_id in npc_ids
        if not bool(treatment_trip_metrics[vehicle_id].get("arrived"))
    ]

    worlds = [collect_interactions(payload, topology)
              for payload in (treatment, reference)]
    eligible = set(tm["map_trip_vehicle_ids"])
    indirect = [
        propagate(world, tm["focal_vehicle_id"], eligible)
        for world in worlds
    ]
    selected = set().union(
        *(set(world["direct_vehicle_ids"]) for world in worlds),
        *({item["vehicle_id"] for item in items} for items in indirect),
    ) & eligible
    selected = sorted(selected)
    lter = paired_lter(treatment, reference, selected)
    collision_vehicle_ids = sorted({
        entity_id
        for event in npc_collision_events
        for entity_id in (event.get("entity_a"), event.get("entity_b"))
        if entity_id in treatment_npc_ids
    })
    return {
        "metric_revision": PAIRED_TRAFFIC_IMPACT_REVISION,
        "applicable": True,
        "reference_compatibility": compatibility,
        "protocol_hash": compatibility.get("protocol_hash"),
        "npc_vehicle_ids": npc_ids,
        "npc_vehicle_count": len(npc_ids),
        "npc_collision_count": len(npc_collision_events),
        "npc_collision_vehicle_ids": collision_vehicle_ids,
        "npc_not_arrived_count": len(npc_not_arrived_ids),
        "npc_not_arrived_vehicle_ids": npc_not_arrived_ids,
        "selected_vehicle_ids": selected,
        "lter": lter,
        "interaction_evidence": {
            label: dict(direct=world["evidence"], indirect=items)
            for label, world, items in zip(
                ("treatment", "reference"), worlds, indirect)
        },
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--treatment",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    rows = []
    maps = Path(__file__).resolve().parents[2]/"simulation/road_networks"
    for path in sorted(args.treatment.glob("*.json.gz")):
        with gzip.open(path,"rt") as stream:
            t = json.load(stream)
        with gzip.open(args.reference/path.name,"rt") as stream:
            r = json.load(stream)
        for payload in (t,r):
            if payload["run"]["status"] != "completed" or payload.get("agent_infrastructure_errors"):
                raise ValueError(f"Invalid run: {path.name}")
        network = t["variant"]["scenario"]["road_network_id"]
        topology = json.loads((maps/f"{network}_lane_level.json").read_text())
        impact = paired_traffic_impact(t, r, topology)
        rows.append(dict(task=path.name.removesuffix(".json.gz"),
                          impact=impact))
        lter = impact.get("lter", {})
        print(
            rows[-1]["task"],
            f"NPC_collision={impact.get('npc_collision_count')}",
            f"NPC_not_arrived={impact.get('npc_not_arrived_count')}",
            f"net_completion="
            f"{lter.get('net_completion_delta_vehicle_s')}",
            f"net_queue="
            f"{lter.get('net_queue_delta_vehicle_s')}",
            f"scope={len(impact.get('selected_vehicle_ids', []))}",
            flush=True)
    applicable = [
        row["impact"] for row in rows if row["impact"].get("applicable")
    ]
    lter_values = [
        impact.get("lter", {})
        for impact in applicable
        if impact.get("lter", {}).get("applicable")
    ]

    def mean_metric(items, key):
        values = [item.get(key) for item in items
                  if item.get(key) is not None]
        return round(mean(values), 6) if values else None

    result = dict(
        revision=PAIRED_TRAFFIC_IMPACT_REVISION,
        task_count=len(rows),
        applicable_count=len(applicable),
        mean_npc_collision_count=mean_metric(
            applicable, "npc_collision_count"),
        mean_npc_not_arrived_count=mean_metric(
            applicable, "npc_not_arrived_count"),
        lter_revision=LTER_REVISION,
        lter_applicable_count=len(lter_values),
        mean_net_completion_delta_vehicle_s=mean_metric(
            lter_values, "net_completion_delta_vehicle_s"),
        mean_net_queue_delta_vehicle_s=mean_metric(
            lter_values, "net_queue_delta_vehicle_s"),
        rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    selected_count = sum(len(r["impact"].get("selected_vehicle_ids", []))
                         for r in rows)
    lines = ["# Traffic Impact + LTER：交互车辆并集统计", "",
        f"有效配对 {len(applicable)}/{len(rows)}。",
        f"NPC 碰撞次数均值 {result['mean_npc_collision_count']}；"
        f"NPC 未到达数均值 {result['mean_npc_not_arrived_count']}。",
        f"完成时间净增量均值 {result['mean_net_completion_delta_vehicle_s']} 车辆秒；"
        f"排队时间净增量均值 {result['mean_net_queue_delta_vehicle_s']} 车辆秒。",
        f"交互范围共保留 {selected_count} 个车辆行程。", "",
        "集合 = treatment 直接交互 ∪ reference 直接交互 ∪ 两侧按时间传播的跟车链。",
        "未到达 NPC 单独统计；完成时间净增量只在 treatment 已到达的 NPC 上计算，"
        "其分母同步使用这批车辆。排队净增量单独报告，不与完成时间相加；"
        "正值表示变差，负值表示相对 reference 改善。正向/受益分解仅用于审计。",
        "空交互集合的 LTER 记 N/A。", "",
        "| 任务 | NPC碰撞 | NPC未到达 | 完成时间净增量 (vehicle-s) | 排队净增量 (vehicle-s) | 完成时间车辆数 | 交互车辆数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        impact = row["impact"]
        lter = impact.get("lter", {})
        lines.append(
            f"| {row['task']} | {impact.get('npc_collision_count')} | "
            f"{impact.get('npc_not_arrived_count')} | "
            f"{lter.get('net_completion_delta_vehicle_s', 'N/A')} | "
            f"{lter.get('net_queue_delta_vehicle_s', 'N/A')} | "
            f"{lter.get('completion_vehicle_count', 'N/A')} | "
            f"{len(impact.get('selected_vehicle_ids', []))} |")
    args.output.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print({k:v for k,v in result.items() if k!="rows"})


if __name__ == "__main__":
    main()
