"""Regression tests for the public open-source extension surface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYTHONPATH = str(REPOSITORY_ROOT / "vehiclearena")


def _run_python(*arguments: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = PYTHONPATH
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_documented_extension_and_scenario_validate():
    process = _run_python(
        "scripts/validate_extension.py",
        "--extension", "docs.examples.fleet_extension",
        "--scenario", "docs/examples/extended_scenario.json",
        "--rules", "docs/examples/rules",
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    extension = report["extensions"][0]
    assert extension["vehicle_modules"] == ["energyMeter"]
    assert extension["runtime_entities"] == ["cargo_bike"]
    assert extension["scenario_layers"] == ["school_zone"]
    assert report["scenario"]["vehicles"] == 1
    assert report["rules"]["user_intent_rules"] == 1


def test_registered_external_module_and_pending_hook_are_installed():
    source = r'''
from module.base_module import BaseModule
from registry import register_constraint, register_module
from vehiclearena import VehicleWorld

installed = []

@register_constraint("roadProbe")
def install(engine):
    installed.append(engine)

@register_module(
    "roadProbe", description="road probe", category="sensor",
    is_external=True,
)
class RoadProbe(BaseModule):
    pass

world = VehicleWorld(equipment_profile="full")
assert world.externalWorld.roadProbe is not None
assert world.roadProbe is world.externalWorld.roadProbe
assert world._get_module("roadProbe") is world.externalWorld.roadProbe
assert len(installed) == 1
'''
    process = _run_python("-c", source)
    assert process.returncode == 0, process.stderr


def test_removed_personality_field_fails_at_scenario_parse():
    source = r'''
from simulation.multi_sim_engine import MultiScenario

MultiScenario.from_dict({
    "scenario_id": "invalid_params",
    "road_network_id": "beijing_guomao",
    "vehicles": [{
        "vehicle_id": "ego",
        "initial_node": "n33399858",
        "driver_plugin": "normal",
        "agent_config": {"type": "sumo"},
    }],
})
'''
    process = _run_python("-c", source)
    assert process.returncode != 0
    assert "driver_plugin" in process.stderr


def test_scenario_layer_uses_explicit_order_and_validates_intensity():
    from simulation.scenario_generator import (
        ScenarioLayer, register_scenario_layer,
    )

    class LocalLayer(ScenarioLayer):
        def apply(self, ctx):
            pass

    with pytest.raises(ValueError, match="Invalid intensity"):
        LocalLayer("maximum")

    source = r'''
from simulation.scenario_generator import ScenarioLayer, register_scenario_layer

@register_scenario_layer("bad_layer", order=10)
class BadLayer(ScenarioLayer):
    def apply(self, ctx):
        pass

try:
    @register_scenario_layer("bad_layer", order=20)
    class DuplicateLayer(ScenarioLayer):
        def apply(self, ctx):
            pass
except ValueError as error:
    assert "already registered" in str(error)
else:
    raise AssertionError("duplicate layer registration must fail")
'''
    process = _run_python("-c", source)
    assert process.returncode == 0, process.stderr
