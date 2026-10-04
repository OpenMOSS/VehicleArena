"""Contracts for the 80-scene Basic expansion and its 100/80 split."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG_ROOT = (
    ROOT / "vehiclearena/evaluation/experiments/scenarios")


def _catalog() -> dict:
    return json.loads((CATALOG_ROOT / "catalog.json").read_text())


def _category(entry: dict) -> str:
    return entry["source_template_id"].split("__", 1)[0]


def test_basic_expansion_has_frozen_ids_and_requested_distribution() -> None:
    entries = [
        entry for entry in _catalog()["entries"]
        if entry["experiment_id"] == "Basic"
        and int(entry["scene_id"].split("_")[1]) >= 161
    ]
    assert len(entries) == 80
    assert {int(entry["scene_id"].split("_")[1]) for entry in entries} == set(
        range(161, 241))
    assert Counter(map(_category, entries)) == {
        "baseline_crosswalk": 10,
        "baseline_signalized_intersection": 1,
        "baseline_straight_following": 5,
        "baseline_unsignalized_intersection": 10,
        "chassis_continuous_turns": 1,
        "chassis_full_network_route": 5,
        "chassis_lead_vehicle_braking": 1,
        "chassis_narrow_road": 1,
        "chassis_red_light_stop": 1,
        "baseline_required_turn": 10,
        "traffic_oncoming_stream": 5,
        "traffic_unprotected_left_turn": 5,
        "traffic_obstacle_gap_change": 5,
        "traffic_crowded_unsignalized_crosswalk": 5,
        "traffic_platoon_pressure": 5,
        "traffic_crossing_stream": 5,
        "traffic_merge_stream": 5,
    }


def test_expansion_uses_new_maps_except_for_compiled_pedestrian_sites() -> None:
    catalog = _catalog()
    old_basic = [
        entry for entry in catalog["entries"]
        if entry["experiment_id"] == "Basic"
        and int(entry["scene_id"].split("_")[1]) <= 160
    ]
    expansion = [
        entry for entry in catalog["entries"]
        if entry["experiment_id"] == "Basic"
        and int(entry["scene_id"].split("_")[1]) >= 161
    ]
    old_maps = {entry["network"] for entry in old_basic}
    pedestrian = [
        entry for entry in expansion
        if _category(entry) in {
            "baseline_crosswalk",
            "traffic_crowded_unsignalized_crosswalk",
        }
    ]
    new_map_entries = [entry for entry in expansion if entry not in pedestrian]
    assert len(pedestrian) == 15
    assert len(new_map_entries) == 65
    assert not ({entry["network"] for entry in new_map_entries} & old_maps)
    assert len({entry["network"] for entry in new_map_entries}) == 65


def test_pedestrian_expansion_uses_distinct_compiled_routes() -> None:
    entries = [
        entry for entry in _catalog()["entries"]
        if entry["experiment_id"] == "Basic"
        and int(entry["scene_id"].split("_")[1]) in range(161, 241)
        and _category(entry) in {
            "baseline_crosswalk",
            "traffic_crowded_unsignalized_crosswalk",
        }
    ]
    connector_ids = []
    for entry in entries:
        scenario = json.loads((CATALOG_ROOT / entry["scenario"]).read_text())
        connector_ids.append(
            scenario["vehicles"][0]["initial_physical_state"]
            ["lane_route_actions"][0]["connector_id"])
    assert len(connector_ids) == len(set(connector_ids)) == 15


def test_recommended_split_is_exactly_capability_stratified() -> None:
    catalog = _catalog()
    split = json.loads(
        (CATALOG_ROOT / catalog["basic_split"]["path"]).read_text())
    assert split["train_count"] == len(split["train_scene_ids"]) == 100
    assert split["test_count"] == len(split["test_scene_ids"]) == 80
    assert not (set(split["train_scene_ids"]) & set(split["test_scene_ids"]))
    assert set(split["train_scene_ids"]) | set(split["test_scene_ids"]) == {
        entry["scene_id"] for entry in catalog["entries"]
        if entry["experiment_id"] == "Basic"
    }
    for category, counts in split["capability_distribution"].items():
        if category in split["weighted_capabilities"]:
            assert counts == {"total": 18, "train": 10, "test": 8}
        else:
            assert counts == {"total": 9, "train": 5, "test": 4}
        # Both subsets therefore have the exact same normalized category
        # weight: weighted categories are 2x every ordinary category.
        assert counts["train"] / 100 == counts["test"] / 80
    assert split["environment_distribution"] == {
        group: {"total": 45, "train": 25, "test": 20}
        for group in (
            "control", "weather_only", "daynight_only", "combined")
    }
    by_id = {
        entry["scene_id"]: entry for entry in catalog["entries"]
        if entry["experiment_id"] == "Basic"}
    for group in split["environment_distribution"]:
        assert sum(
            by_id[scene_id]["environment_group"] == group
            for scene_id in split["train_scene_ids"]) == 25
        assert sum(
            by_id[scene_id]["environment_group"] == group
            for scene_id in split["test_scene_ids"]) == 20


def test_required_turn_expansion_balances_left_and_right() -> None:
    entries = [
        entry for entry in _catalog()["entries"]
        if entry["experiment_id"] == "Basic"
        and int(entry["scene_id"].split("_")[1]) in range(196, 206)
    ]
    turns = []
    for entry in entries:
        expected = json.loads((CATALOG_ROOT / entry["expected"]).read_text())
        assertion = next(
            item for item in expected["setup_assertions"]
            if item["kind"] == "required_lane_change_turn")
        scenario = json.loads((CATALOG_ROOT / entry["scenario"]).read_text())
        action = scenario["vehicles"][0]["initial_physical_state"][
            "lane_route_actions"][1]
        assert action["connector_id"] == assertion["connector_id"]
        turns.append(action["turn"])
    assert Counter(turns) == {"left": 5, "right": 5}
