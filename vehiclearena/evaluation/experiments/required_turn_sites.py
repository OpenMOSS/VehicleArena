#!/usr/bin/env python3
"""Verify eight lane-change-before-turn sites against the current lane graph.

Run from any directory; emits JSON to stdout and never changes frozen tasks.
This is a topology check, not a SUMO/LLM rollout or surveyed lane-marking audit.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SITES = (
    ("beijing_guomao", "n11113007817_n1859897266::lane_1", "connector::n11113007817::1"),
    ("hongkong_central", "n1615603888_n1615604003::lane_0", "connector::n1615604003::2"),
    ("shanghai_lujiazui", "n1785697732_n84466496::lane_1", "connector::n84466496::7"),
    ("tokyo_shinjuku", "n295423004_n295423016::lane_2", "connector::intersection::n295422511::4"),
    ("guangzhou_tianhe", "n2048660645_n9446307866::lane_1", "connector::n2048660645::1"),
    ("wuhan_hankou", "n1880367373_n1880396010::lane_2", "connector::intersection::n1880367367::11"),
    ("newyork_manhattan_mid", "n42446941_n42449932::lane_1", "connector::n42449932::8"),
    ("chengdu_jinjiang", "n1159517085_n1296102309::lane_1", "connector::n1296102309::1"),
)


def verify_sites():
    reports = []
    for number, (network, initial_id, connector_id) in enumerate(SITES, 1):
        directory = ROOT / "vehiclearena/simulation/road_networks"
        data = json.loads((directory / f"{network}_lane_level.json").read_text())
        base = json.loads((directory / f"{network}.json").read_text())
        segments = {s["id"]: s for s in base["segments"]}
        lanes = {lane["id"]: lane for lane in data["lanes"]}
        connectors = {c["id"]: c for c in data["connectors"]}
        outgoing = defaultdict(list)
        for connector in connectors.values():
            outgoing[connector["from_lane"]].append(connector)
        initial = lanes[initial_id]
        turn = connectors[connector_id]
        target = lanes[turn["from_lane"]]
        exit_lane = lanes[turn["to_lane"]]
        assert {c["turn"] for c in outgoing[initial_id]} == {"straight"}
        assert initial["segment_id"] == target["segment_id"]
        assert initial["direction"] == target["direction"]
        assert abs(initial["index"] - target["index"]) == 1
        assert initial["end_junction"] == target["end_junction"] == turn["node_id"]
        assert not initial.get("shared_bidirectional")
        assert not target.get("shared_bidirectional")
        assert initial["length_m"] >= 120 and exit_lane["length_m"] >= 60
        assert turn["turn"] in {"left", "right"}
        assert initial["z_level"] == target["z_level"] == turn["z_level"]

        # All legal connectors are allowed, including U-turns and arbitrarily
        # long detours. Only lateral lane-change edges are excluded.
        reachable = {initial_id}
        queue = [initial_id]
        while queue:
            for connector in outgoing[queue.pop()]:
                following = connector["to_lane"]
                if following not in reachable:
                    reachable.add(following)
                    queue.append(following)
        destination = exit_lane["end_node"]
        assert destination not in {lanes[l]["end_node"] for l in reachable}
        assert destination != initial["start_node"]
        reports.append({
            "site_id": f"required_turn_{number:02d}", "network": network,
            "approach_road": segments[initial["segment_id"]].get("name", ""),
            "exit_road": segments[exit_lane["segment_id"]].get("name", ""),
            "initial_lane_id": initial_id, "target_lane_id": target["id"],
            "connector_id": connector_id, "junction_id": turn["node_id"],
            "turn": turn["turn"], "exit_lane_id": exit_lane["id"],
            "destination_node": destination,
            "approach_length_m": initial["length_m"],
            "exit_length_m": exit_lane["length_m"],
            "proposed_initial_progress": round(1 - 100 / initial["length_m"], 6),
            "proposed_initial_speed_kmh": 30,
            "target_lane_allowed_turns": sorted({c["turn"] for c in outgoing[target["id"]]}),
            "signal_controlled": turn["signal_controlled"],
            "reachable_lane_count_without_lane_change": len(reachable),
            "destination_reachable_without_lane_change": False,
            "validation": "lane_graph_only; no SUMO or LLM rollout",
        })
    assert len({r["network"] for r in reports}) == 8
    assert sum(r["turn"] == "left" for r in reports) == 4
    return reports


if __name__ == "__main__":
    print(json.dumps(verify_sites(), ensure_ascii=False, indent=2))
