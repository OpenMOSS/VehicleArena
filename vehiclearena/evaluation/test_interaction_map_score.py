from evaluation.experiments.interaction_map_score import (
    collect_interactions, paired_lter, paired_traffic_impact, propagate)
from evaluation.experiments.reference_protocol import (
    REFERENCE_PROTOCOL_REVISION, REFERENCE_PROTOCOL_SCHEMA, protocol_hash,
)


def _system_metrics(*, completion=10.0, queue=0.0, arrived=True):
    return {
        "focal_vehicle_id": "ego",
        "required_trip_vehicle_ids": ["ego", "npc"],
        "map_trip_vehicle_ids": ["npc"],
        "trip_vehicle_metrics": {
            "ego": {
                "arrived": True,
                "completion_time_s": 8.0,
                "queue_wait_s": 0.0,
            },
            "npc": {
                "arrived": arrived,
                "completion_time_s": completion,
                "queue_wait_s": queue,
            },
        },
        "collision_count": 0,
        "secondary_collision_count": 0,
        "confirmed_deadlock_count": 0,
        "all_trip_vehicles_arrived": bool(arrived),
    }


def topology():
    return {"lanes": [dict(id=k,segment_id=seg,direction=direction,index=i,
                            length_m=200,z_level=0)
                      for k,seg,direction,i in [("a","road","forward",0),
                        ("b","road","forward",1),("opposite","road","backward",2),
                        ("remote","other","forward",0)]],
            "connectors": [], "connector_conflicts": []}


def payload(lane="a", offset=10):
    rows=[]
    for tick in range(31):
        for vid,lid,pos in [("ego","a",100),("npc",lane,100-offset)]:
            rows.append(dict(vehicle_id=vid,time_s=tick/10,current_lane_id=lid,
                             edge_progress=pos/200,speed_kmh=20,length_m=4.6))
    metrics=_system_metrics()
    return dict(system_evaluation=metrics,source_hash="same-runtime",manifest_hash="same-manifest",
                run=dict(reference_policy="all_sumo"),collision_log=[],
                trajectories=dict(vehicles=rows),variant=dict(scenario_hash="same-scene",
                  scenario=dict(vehicles=[dict(vehicle_id="ego"),dict(vehicle_id="npc")])) )


def add_protocol(value, marker="same-protocol"):
    record = {
        "schema": REFERENCE_PROTOCOL_SCHEMA,
        "revision": REFERENCE_PROTOCOL_REVISION,
        "marker": marker,
    }
    value["reference_protocol"] = record
    value["protocol_hash"] = protocol_hash(record)
    return value


def test_topology_excludes_nearby_unrelated_and_opposing_roads():
    for lane in ["remote","opposite"]:
        assert collect_interactions(payload(lane),topology())["direct_vehicle_ids"] == []
    assert collect_interactions(payload("b"),topology())["direct_vehicle_ids"] == ["npc"]


def test_terminal_actors_are_excluded_from_interaction_scope():
    t=payload()
    for row in t["trajectories"]["vehicles"]:
        if row["vehicle_id"] == "npc":
            row["arrived"]=True
    assert collect_interactions(t,topology())["direct_vehicle_ids"] == []


def test_following_propagates_forward_in_time_only():
    world=dict(evidence=[dict(vehicle_id="leader",qualified_at_s=3)],
               following_edges=[dict(start_s=t/10,end_s=(t+1)/10,
                    leader="leader",follower="tail",gap_m=5) for t in range(50)])
    result=propagate(world,"ego",{"leader","tail"})
    assert len(result)==1 and result[0]["qualified_at_s"] == 5
    world["following_edges"]=world["following_edges"][:20]
    assert propagate(world,"ego",{"leader","tail"}) == []


def test_sampling_gap_cannot_manufacture_contact_duration():
    t=payload()
    t["trajectories"]["vehicles"]=[r for r in t["trajectories"]["vehicles"] if r["time_s"] in [0,3]]
    assert collect_interactions(t,topology())["direct_vehicle_ids"] == []


def test_connector_conflict_requires_overlapping_arrival_windows():
    m=topology()
    m['connectors']=[dict(id='c1',from_lane='a',to_lane='b'),
                     dict(id='c2',from_lane='remote',to_lane='opposite')]
    m['connector_conflicts']=[dict(connector_a='c1',connector_b='c2')]
    p=payload('remote')
    for v,cid in zip(p['variant']['scenario']['vehicles'],['c1','c2']):
        v['initial_physical_state']=dict(lane_route_actions=[dict(type='connector',connector_id=cid)])
    for row in p['trajectories']['vehicles']:
        row['edge_progress']=0.9
    assert collect_interactions(p,m)['direct_vehicle_ids']==['npc']
    for row in p['trajectories']['vehicles']:
        if row['vehicle_id']=='npc':
            row['speed_kmh']=1
    assert collect_interactions(p,m)['direct_vehicle_ids']==[]


def test_only_nearest_leader_forms_propagation_edge():
    p=payload(offset=80)
    p['variant']['scenario']['vehicles'].append(dict(vehicle_id='middle'))
    for tick in range(31):
        p['trajectories']['vehicles'].append(dict(vehicle_id='middle',time_s=tick/10,
            current_lane_id='a',edge_progress=0.1,speed_kmh=20,length_m=4.6))
    world=collect_interactions(p,topology())
    assert not any(e['leader']=='ego' and e['follower']=='middle' for e in world['following_edges'])


def _lter_payload(completion, queue, *, extra=False):
    result = payload()
    result["variant"]["scenario"]["vehicles"] = [
        {"vehicle_id": "ego", "initial_physical_state": {
            "lane_id": "a", "progress": 0.0, "lane_route_actions": []}},
        {"vehicle_id": "npc", "initial_physical_state": {
            "lane_id": "a", "progress": 0.5, "lane_route_actions": []}},
    ]
    metrics = result["system_evaluation"]
    metrics["trip_vehicle_metrics"]["npc"] = {
        "arrived": True, "completion_time_s": completion,
        "queue_wait_s": queue,
    }
    if extra:
        result["variant"]["scenario"]["vehicles"].append({
            "vehicle_id": "npc_extra", "initial_physical_state": {
                "lane_id": "a", "progress": 0.7, "lane_route_actions": []}})
        metrics["trip_vehicle_metrics"]["npc_extra"] = {
            "arrived": True, "completion_time_s": 10.0,
            "queue_wait_s": 0.0,
        }
    return result


def test_lter_reports_signed_net_vehicle_seconds_for_selected_scope():
    treatment = _lter_payload(20.0, 5.0)
    reference = _lter_payload(10.0, 0.0)
    one = paired_lter(treatment, reference, ["npc"])
    assert one["net_completion_delta_vehicle_s"] == 10.0
    assert one["net_queue_delta_vehicle_s"] == 5.0
    assert one["positive_completion_delta_vehicle_s"] == 10.0
    assert one["benefit_completion_delta_vehicle_s"] == 0.0
    assert "excess_delay_vehicle_s" not in one
    assert "excess_queue_vehicle_s" not in one

    treatment_extra = _lter_payload(20.0, 5.0, extra=True)
    reference_extra = _lter_payload(10.0, 0.0, extra=True)
    two = paired_lter(
        treatment_extra, reference_extra, ["npc", "npc_extra"])
    assert two["net_completion_delta_vehicle_s"] == one["net_completion_delta_vehicle_s"]
    assert two["net_queue_delta_vehicle_s"] == one["net_queue_delta_vehicle_s"]
    assert two["vehicle_count"] == 2
    assert two["affected_vehicle_count"] == 1


def test_lter_excludes_treatment_unarrived_npc_from_completion_time():
    treatment = _lter_payload(20.0, 5.0, extra=True)
    reference = _lter_payload(10.0, 0.0, extra=True)
    treatment["system_evaluation"]["trip_vehicle_metrics"]["npc"][
        "arrived"] = False
    treatment["system_evaluation"]["trip_vehicle_metrics"]["npc_extra"].update(
        completion_time_s=30.0)

    result = paired_lter(
        treatment, reference, ["npc", "npc_extra"])

    assert result["completion_vehicle_ids"] == ["npc_extra"]
    assert result["completion_vehicle_count"] == 1
    assert result["excluded_unarrived_vehicle_ids"] == ["npc"]
    assert result["net_completion_delta_vehicle_s"] == 20.0
    assert result["mean_net_completion_delta_s"] == 20.0
    assert result["affected_vehicle_count"] == 1
    assert result["affected_vehicle_rate"] == 1.0
    assert result["per_vehicle"]["npc"]["completion_time_scored"] is False
    assert result["per_vehicle"]["npc"]["positive_completion_delta_s"] is None


def test_lter_net_delta_preserves_improvements_instead_of_clipping_them():
    treatment = _lter_payload(5.0, 2.0)
    reference = _lter_payload(10.0, 5.0)

    result = paired_lter(treatment, reference, ["npc"])

    assert result["net_completion_delta_vehicle_s"] == -5.0
    assert result["net_queue_delta_vehicle_s"] == -3.0
    assert result["positive_completion_delta_vehicle_s"] == 0.0
    assert result["positive_queue_delta_vehicle_s"] == 0.0
    assert result["benefit_completion_delta_vehicle_s"] == 5.0
    assert result["benefit_queue_delta_vehicle_s"] == 3.0
    assert result["per_vehicle"]["npc"]["completion_delta_s"] == -5.0
    assert result["per_vehicle"]["npc"]["negative_completion_delta_s"] == -5.0
    assert result["per_vehicle"]["npc"]["queue_delta_s"] == -3.0
    assert result["per_vehicle"]["npc"]["negative_queue_delta_s"] == -3.0


def test_traffic_impact_reports_npc_failures_without_composite_score():
    treatment = _lter_payload(20.0, 5.0)
    reference = _lter_payload(10.0, 0.0)
    treatment["system_evaluation"]["trip_vehicle_metrics"]["npc"][
        "arrived"] = False
    treatment["collision_log"] = [
        {"entity_a": "ego", "entity_b": "npc", "time_s": 1.0}
    ]

    result = paired_traffic_impact(treatment, reference, topology())

    assert result["applicable"] is True
    assert result["npc_collision_count"] == 1
    assert result["npc_not_arrived_count"] == 1
    assert result["npc_not_arrived_vehicle_ids"] == ["npc"]
    assert result["lter"]["completion_vehicle_count"] == 0
    assert result["lter"]["net_completion_delta_vehicle_s"] == 0.0


def test_protocol_hash_allows_different_source_and_manifest_provenance():
    treatment = add_protocol(_lter_payload(20.0, 5.0))
    reference = add_protocol(_lter_payload(10.0, 0.0))
    treatment.update(source_hash="source-a", manifest_hash="manifest-a")
    reference.update(source_hash="source-b", manifest_hash="manifest-b")

    result = paired_traffic_impact(treatment, reference, topology())

    assert result["applicable"] is True
    compatibility = result["reference_compatibility"]
    assert compatibility["compatibility_mode"] == "protocol_hash"
    assert compatibility["provenance_mismatches"] == [
        "source_hash", "manifest_hash"]


def test_protocol_hash_mismatch_rejects_reference():
    treatment = add_protocol(_lter_payload(20.0, 5.0), "protocol-a")
    reference = add_protocol(_lter_payload(10.0, 0.0), "protocol-b")

    result = paired_traffic_impact(treatment, reference, topology())

    assert result["applicable"] is False
    assert result["reason"] == "paired_protocol_hash_mismatch"


def test_tampered_protocol_record_is_rejected():
    treatment = add_protocol(_lter_payload(20.0, 5.0))
    reference = add_protocol(_lter_payload(10.0, 0.0))
    reference["reference_protocol"]["marker"] = "tampered"

    result = paired_traffic_impact(treatment, reference, topology())

    assert result["applicable"] is False
    assert result["reason"] == "paired_reference_protocol_invalid"


def test_legacy_artifacts_keep_exact_provenance_fallback():
    treatment = _lter_payload(20.0, 5.0)
    reference = _lter_payload(10.0, 0.0)
    reference["source_hash"] = "different-runtime"

    result = paired_traffic_impact(treatment, reference, topology())

    assert result["applicable"] is False
    assert result["compatibility_mode"] == "legacy_exact_hashes"
    assert result["reason"] == "paired_source_hash_mismatch"
