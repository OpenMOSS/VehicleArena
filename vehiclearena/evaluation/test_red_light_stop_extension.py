"""Contracts for the four bounded red-light braking extensions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.experiments.scene_catalog import RED_LIGHT_STOP_EXTRA_SITES
from evaluation.experiments.scene_validator import SceneValidator
from simulation.lane_level_runtime import LaneGeometryRuntime


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = ROOT / "evaluation" / "experiments" / "scenarios"
NETWORK_ROOT = ROOT / "simulation" / "road_networks"


def task_id(index: int, network: str) -> str:
    return f"basic_{129 + index:03d}_red_light_stop__extension__{network}"


@pytest.mark.parametrize(
    ("index", "network", "connector_id"),
    [
        (index, network, connector_id)
        for index, (network, connector_id)
        in enumerate(RED_LIGHT_STOP_EXTRA_SITES)
    ],
)
def test_bounded_red_light_stop_task_contract(
    index: int, network: str, connector_id: str,
) -> None:
    scene_id = task_id(index, network)
    directory = SCENARIO_ROOT / "Basic" / scene_id
    scenario = json.loads(
        (directory / "scenario.json").read_text(encoding="utf-8"))
    expected = json.loads(
        (directory / "expected.json").read_text(encoding="utf-8"))

    SceneValidator(NETWORK_ROOT).validate(scenario, expected)
    assertion = next(
        item for item in expected["setup_assertions"]
        if item["kind"] == "bounded_red_light_approach")
    runtime = LaneGeometryRuntime.load(
        NETWORK_ROOT / f"{network}_lane_level.json")
    signal = runtime.signal_state(connector_id, 0.0)
    placement = scenario["vehicles"][0]["initial_physical_state"]
    lane = runtime._lane_by_id[placement["lane_id"]]
    distance = (
        runtime.stop_progress(lane["id"]) - placement["progress"]
    ) * lane["length_m"]

    assert signal is not None and signal.signal == "red"
    assert 8.0 <= signal.remaining_seconds <= 15.0
    assert distance == pytest.approx(38.0, abs=0.001)
    assert placement["speed_kmh"] == 32.0
    assert assertion["connector_id"] == connector_id


def test_bounded_red_light_stop_environment_factorial() -> None:
    groups = []
    turns = set()
    for index, (network, connector_id) in enumerate(
            RED_LIGHT_STOP_EXTRA_SITES):
        scene_id = task_id(index, network)
        scenario = json.loads((
            SCENARIO_ROOT / "Basic" / scene_id / "scenario.json"
        ).read_text(encoding="utf-8"))
        groups.append(scenario["experiment_scene"]["environment_group"])
        runtime = LaneGeometryRuntime.load(
            NETWORK_ROOT / f"{network}_lane_level.json")
        turns.add(runtime._connector_by_id[connector_id]["turn"])

    assert groups == [
        "weather_only", "daynight_only", "combined", "control"]
    assert turns == {"left", "straight", "right"}
