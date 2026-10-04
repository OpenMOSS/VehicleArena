"""Compact turns are distinct, nearby, continuous, and retained by replanning."""
import copy
import json
from pathlib import Path

import pytest

from evaluation.experiments.scene_catalog import (
    BASIC_COMPACT_TURN_EXPANSION_SITES,
    COMPACT_TURN_SITES,
    Topology,
    _compact_turn_route,
    normalize_native_npc_inputs,
)
from evaluation.experiments.scene_validator import SceneValidator, SceneValidationError

ROOT = Path(__file__).resolve().parents[2]
MAPS = ROOT / 'vehiclearena/simulation/road_networks'


@pytest.fixture(scope='module', params=COMPACT_TURN_SITES, ids=lambda s: s[0])
def site(request):
    network, lane, connectors, basis = request.param
    topology = Topology(network, MAPS)
    asset = _compact_turn_route(topology, lane, connectors, basis)
    normalize_native_npc_inputs(asset.scenario)
    validator = SceneValidator(MAPS)
    validator._runtime[network] = topology.runtime
    return topology, asset, validator


def test_compact_geometry_and_replanning(site):
    _, asset, validator = site
    validator.validate(asset.scenario, asset.expected)


def test_missing_turn_rejected(site):
    _, asset, validator = site
    scenario = copy.deepcopy(asset.scenario)
    scenario['vehicles'][0]['initial_physical_state']['lane_route_actions'].pop()
    with pytest.raises(SceneValidationError):
        validator.validate(scenario, asset.expected)


def test_wrong_start_rejected(site):
    _, asset, validator = site
    scenario = copy.deepcopy(asset.scenario)
    scenario['vehicles'][0]['initial_physical_state']['progress'] = 0.01
    with pytest.raises(SceneValidationError, match='60 m'):
        validator.validate(scenario, asset.expected)


def test_spacing_contract_enforced(site):
    _, asset, validator = site
    expected = copy.deepcopy(asset.expected)
    assertion = next(a for a in expected['setup_assertions'] if a['kind'] == 'compact_turn_route')
    assertion['maximum_gap_m'] = 40
    with pytest.raises(SceneValidationError, match='spacing'):
        validator.validate(asset.scenario, expected)


def test_frozen_catalog_has_nine_continuous_turn_tasks():
    root = ROOT / 'vehiclearena/evaluation/experiments/scenarios'
    catalog = json.loads((root / 'catalog.json').read_text())
    turns = [e for e in catalog['entries'] if e['source_template_id'].split('__')[0] == 'chassis_continuous_turns']
    # Four original routes, four compact-turn extensions, and one additional
    # distribution-matched Basic expansion route are frozen in the catalog.
    assert len(turns) == 9
    compact = [e for e in turns if 'compact_turns' in e['tags']]
    assert {int(e['scene_id'].split('_')[1]) for e in compact} == {
        117, 118, 119, 120, 187}
    assert {e['network'] for e in compact} == {
        *(s[0] for s in COMPACT_TURN_SITES),
        *(s[0] for s in BASIC_COMPACT_TURN_EXPANSION_SITES),
    }
    for e in compact:
        scenario = json.loads((root / e['scenario']).read_text())
        expected = json.loads((root / e['expected']).read_text())
        cohort_ids = scenario['experiment_scene'][
            'related_background_traffic']['longitudinal_cohort_vehicle_ids']
        assert 3 <= len(cohort_ids) <= 8
        assert len(scenario['vehicles']) == 6 + len(cohort_ids)
        assert scenario['vehicles'][0]['agent_config']['type'] == 'llm'
        assert any(a['kind'] == 'compact_turn_route' for a in expected['setup_assertions'])
