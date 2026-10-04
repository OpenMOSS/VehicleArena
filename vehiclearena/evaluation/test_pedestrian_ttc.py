"""Fixed task-side pedestrian entry works with the unchanged native engine."""
import json
from pathlib import Path

import pytest

from evaluation.experiments.telemetry import ExperimentTrackedEngine
from evaluation.experiments.task_compatibility import (
    PEDESTRIAN_RELEASE_OVERRIDES_S,
)
from simulation.multi_sim_engine import MultiScenario
from simulation.sumo_traffic_manager import SumoTrafficManager

ROOT = Path(__file__).parent / "experiments/scenarios/Basic"


def test_reference_validated_pedestrian_release_overrides_are_frozen():
    for scene_id, expected_starts in PEDESTRIAN_RELEASE_OVERRIDES_S.items():
        raw = json.loads((ROOT / scene_id / "scenario.json").read_text())
        actual_starts = {
            ped["ped_id"]: round(float(ped["start_time"]) * 60.0, 6)
            for ped in raw["pedestrians"]
            if ped["ped_id"] in expected_starts
        }
        assert actual_starts == expected_starts
        scheduled = next(
            assertion for assertion in json.loads(
                (ROOT / scene_id / "expected.json").read_text())[
                    "setup_assertions"]
            if assertion["kind"] == "scheduled_crosswalk_entry")
        assert scheduled["start_times_s"] == expected_starts


@pytest.mark.parametrize("name", sorted(
    path.parent.name for path in ROOT.glob("*/scenario.json")
    if any(change["kind"] == "fixed_pedestrian_spawn"
           for change in json.loads(path.read_text())["experiment_scene"]
           .get("task_migration", {}).get("changes", []))
))
def test_pedestrian_enters_at_fixed_scenario_time(name):
    raw = json.loads((ROOT / name / "scenario.json").read_text())
    last_start = max(p["start_time"] * 60.0 for p in raw["pedestrians"])
    raw.update(physics_only_mode=True, total_time_s=last_start + 2,
               stop_when_all_vehicles_terminal=False, enable_driving_evaluation=False)
    for v in raw["vehicles"]:
        v["agent_config"] = {"type": "sumo"}
    result = ExperimentTrackedEngine(MultiScenario.from_dict(raw)).run({})
    for ped in raw["pedestrians"]:
        start = ped["start_time"] * 60.0
        assert start >= 0.1 and "crossing_trigger" not in ped["initial_physical_state"]
        rows = [row for row in result._pedestrian_trajectory if row["ped_id"] == ped["ped_id"]]
        before = [row for row in rows if row["time_s"] < start - 1e-6]
        assert before and all(not row["spawned"] for row in before)
        after = [row for row in rows if row["time_s"] >= start + .1]
        assert after and any(row["spawned"] for row in after), (name, ped["ped_id"], start)
    assert not any(event["type"] == "pedestrian_crossing_triggered"
                   for event in result._physics_events)
    assert not hasattr(SumoTrafficManager, "release_triggered_pedestrians")


def test_crowd_retains_stagger_without_a_runtime_trigger():
    name = "basic_224_crowded_unsignalized_crosswalk__expansion__guangzhou_tianhe__site_04"
    raw = json.loads((ROOT / name / "scenario.json").read_text())
    starts = [p["start_time"] * 60.0 for p in raw["pedestrians"]]
    assert len(starts) == 8 and len(set(starts)) == 8
    assert max(starts) - min(starts) == pytest.approx(4.2)
    assert all("crossing_trigger" not in p["initial_physical_state"] for p in raw["pedestrians"])
