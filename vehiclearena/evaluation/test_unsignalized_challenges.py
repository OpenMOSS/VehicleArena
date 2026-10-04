"""New crossing sites retain geometry, priority, and stream-placement contracts."""
import copy
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from evaluation.experiments.scene_catalog import (
    Topology, UNSIGNALIZED_CHALLENGE_SITES, _unsignalized_challenge,
    normalize_native_npc_inputs,
)
from evaluation.experiments.scene_validator import SceneValidator, SceneValidationError
from simulation.sumo_map import SumoMapConverter

ROOT = Path(__file__).resolve().parents[2]
MAPS = ROOT / 'vehiclearena/simulation/road_networks'


@pytest.fixture(scope='module', params=UNSIGNALIZED_CHALLENGE_SITES, ids=lambda s: s[0])
def site(request):
    network, ego_id, cross_id, basis = request.param
    topology = Topology(network, MAPS)
    asset = _unsignalized_challenge(topology, ego_id, cross_id, basis)
    normalize_native_npc_inputs(asset.scenario)
    validator = SceneValidator(MAPS)
    validator._runtime[network] = topology.runtime
    return topology, asset, validator, ego_id, cross_id


def test_geometry_and_placement(site):
    topology, asset, validator, ego_id, cross_id = site
    validator.validate(asset.scenario, asset.expected)
    assert topology.connector_conflict_angle_deg(topology.connector[ego_id], topology.connector[cross_id]) > 85
    for path in (ROOT / 'vehiclearena/evaluation/experiments/scenarios/Basic').glob('*/scenario.json'):
        scenario = json.loads(path.read_text())
        source_family = (scenario.get('experiment_scene', {})
                         .get('source_template_id', '').split('__', 1)[0])
        if (scenario['road_network_id'] != topology.network
                or '__challenge__' in scenario['scenario_id']
                or source_family != 'baseline_unsignalized_intersection'):
            continue
        for vehicle in scenario['vehicles']:
            for action in vehicle.get('initial_physical_state', {}).get('lane_route_actions', []):
                if action.get('type') == 'connector':
                    assert action['connector_id'].rsplit('::', 1)[0] != ego_id.rsplit('::', 1)[0]


def test_native_priority_requires_ego_to_yield(site):
    if not shutil.which('netconvert'):
        pytest.skip('SUMO netconvert unavailable')
    topology, _, _, ego_id, cross_id = site
    bundle = SumoMapConverter().convert(topology.path)
    net = ET.parse(bundle.net_file).getroot()
    ego_via, cross_via = bundle.connector_via_lane[ego_id], bundle.connector_via_lane[cross_id]
    junction = next(j for j in net.findall('junction') if ego_via in j.get('intLanes', '').split())
    lanes = junction.get('intLanes').split()
    ego_index, cross_index = lanes.index(ego_via), lanes.index(cross_via)
    requests = {int(r.get('index')): r for r in junction.findall('request')}
    assert junction.get('type') == 'priority'
    assert requests[ego_index].get('response')[::-1][cross_index] == '1'
    assert requests[ego_index].get('foes')[::-1][cross_index] == '1'
    assert requests[cross_index].get('response')[::-1][ego_index] == '0'


def test_stream_misplacement_rejected(site):
    _, asset, validator, _, _ = site
    scenario = copy.deepcopy(asset.scenario)
    scenario['vehicles'][2]['initial_physical_state']['progress'] = scenario['vehicles'][1]['initial_physical_state']['progress']
    with pytest.raises(SceneValidationError, match='placement/setback'):
        validator.validate(scenario, asset.expected)


def test_route_bypass_rejected(site):
    _, asset, validator, _, _ = site
    scenario = copy.deepcopy(asset.scenario)
    scenario['vehicles'][0]['initial_physical_state']['lane_route_actions'] = []
    with pytest.raises(SceneValidationError):
        validator.validate(scenario, asset.expected)
