"""Regression contracts for the rebuilt Basic and MultiLLM traffic studies."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluation.experiments.scene_catalog import (
    RETIRED_BASIC_STUDY_TEMPLATES,
    RETIRED_MULTI_STUDY_TEMPLATES,
)


CATALOG_ROOT = Path(__file__).parent / "experiments" / "scenarios"


def _catalog() -> dict:
    return json.loads((CATALOG_ROOT / "catalog.json").read_text())


def _payload(entry: dict, name: str) -> dict:
    return json.loads((CATALOG_ROOT / entry[name]).read_text())


def test_rebuild_replaces_every_retired_formal_template():
    catalog = _catalog()
    for entry in catalog["entries"]:
        retired = (
            RETIRED_BASIC_STUDY_TEMPLATES
            if entry["experiment_id"] == "Basic"
            else RETIRED_MULTI_STUDY_TEMPLATES)
        assert entry["source_template_id"].split("__", 1)[0] not in retired

    basic = [
        entry for entry in catalog["entries"]
        if entry["experiment_id"] == "Basic"
        and 133 <= int(entry["scene_id"].split("_")[1]) <= 160
    ]
    multi = [
        entry for entry in catalog["entries"]
        if entry["experiment_id"] == "MultiLLM"
    ]
    assert len(basic) == 28
    assert {int(e["scene_id"].split("_")[1]) for e in basic} == set(
        range(133, 161))
    assert len(multi) == 40
    assert {int(e["scene_id"].split("_")[1]) for e in multi} == set(
        range(21, 61))
    for entry in basic + multi:
        assert _payload(entry, "expected")["runtime_acceptance"]


def test_multillm_study_has_eighty_llm_slots_and_four_agents_each():
    entries = [
        entry for entry in _catalog()["entries"]
        if entry["experiment_id"] == "MultiLLM"
    ]
    distribution = Counter()
    total_slots = 0
    for entry in entries:
        scenario = _payload(entry, "scenario")
        llm_ids = {
            vehicle["vehicle_id"] for vehicle in scenario["vehicles"]
            if vehicle["agent_config"]["type"] == "llm"
        }
        scene = scenario["experiment_scene"]
        assert scene["focal_vehicle_id"] in llm_ids
        assert set(scene["fixed_peer_vehicle_ids"]) == (
            llm_ids - {scene["focal_vehicle_id"]})
        assert all(
            vehicle["is_evaluated"]
            for vehicle in scenario["vehicles"]
            if vehicle["vehicle_id"] in llm_ids)
        distribution[len(llm_ids)] += 1
        total_slots += len(llm_ids)

    assert distribution == {4: 40}
    assert total_slots == 160
    assert total_slots / len(entries) == 4.0


def test_crowded_unsignalized_crossings_are_staggered_eight_person_streams():
    entries = [
        entry for entry in _catalog()["entries"]
        if "crowded_unsignalized_crosswalk" in entry["scene_id"]
    ]
    assert len(entries) == 9
    assert len({entry["network"] for entry in entries}) == 5
    for entry in entries:
        scenario = _payload(entry, "scenario")
        expected = _payload(entry, "expected")
        cohort_ids = scenario["experiment_scene"][
            "related_background_traffic"]["longitudinal_cohort_vehicle_ids"]
        assert 3 <= len(cohort_ids) <= 5
        assert len(scenario["vehicles"]) == 6 + len(cohort_ids)
        assert len(scenario["pedestrians"]) == 8
        assert all(
            pedestrian["agent_config"]["type"] == "sumo"
            and pedestrian["start_time"] * 60.0 >= 0.1
            for pedestrian in scenario["pedestrians"])
        states = [
            pedestrian["initial_physical_state"]
            for pedestrian in scenario["pedestrians"]]
        assert len({state["crosswalk_id"] for state in states}) == 1
        starts = [p["start_time"] * 60.0 for p in scenario["pedestrians"]]
        assert len(set(starts)) == 8 and max(starts) - min(starts) >= 4.0
        assert all("crossing_trigger" not in state for state in states)
        simultaneous = next(
            item for item in expected["runtime_acceptance"]
            if item["kind"] == "minimum_simultaneous_crosswalk_occupancy")
        assert simultaneous["minimum_count"] == 4
