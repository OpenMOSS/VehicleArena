"""Full-network Basic tasks contain a reproducible nearby oncoming stream."""
import copy
import json
from pathlib import Path
import pytest

from evaluation.experiments.scene_catalog import (
    FULL_NETWORK_ONCOMING_SITES, Topology, _multi_vehicle, normalize_native_npc_inputs,
)
from evaluation.experiments.scene_validator import SceneValidator, SceneValidationError
from simulation.multi_sim_engine import MultiScenario
from evaluation.experiments.telemetry import ExperimentTrackedEngine
from evaluation.experiments.task_compatibility import adapt_scene

ROOT = Path(__file__).resolve().parents[2]
MAPS = ROOT / 'vehiclearena/simulation/road_networks'


def source_id(network):
    return 'chassis_full_network_route' + (
        '' if network == 'shanghai_lujiazui' else '__' + network)


@pytest.fixture(scope='module', params=FULL_NETWORK_ONCOMING_SITES, ids=str)
def site(request):
    network = request.param
    topology = Topology(network, MAPS)
    asset = _multi_vehicle(
        topology, 'chassis', source_id(network), title='probe',
        oncoming_stream=FULL_NETWORK_ONCOMING_SITES[network])
    # Raw templates precede study projection; apply the same native-input
    # cleanup used when publishing runnable study scenarios.
    normalize_native_npc_inputs(asset.scenario)
    adapt_scene(asset.scenario, asset.expected, topology.runtime)
    validator = SceneValidator(MAPS)
    validator._runtime[network] = topology.runtime
    return asset, validator


def test_authored_oncoming_stream_is_static_valid(site):
    asset, validator = site
    validator.validate(asset.scenario, asset.expected)
    assert [v['vehicle_id'] for v in asset.scenario['vehicles'][-3:]] == [
        'oncoming_01', 'oncoming_02', 'oncoming_03']
    assert all('native_lane_change_enabled' not in v['initial_physical_state']
               for v in asset.scenario['vehicles'][-3:])


def test_world_event_for_native_peer_is_rejected(site):
    asset, validator = site
    scenario = copy.deepcopy(asset.scenario)
    scenario['experiment_world_events'] = [{
        'at_s': 1, 'entity_id': 'oncoming_01',
        'action': 'set_vehicle_speed', 'speed_kmh': 0}]
    with pytest.raises(SceneValidationError, match='world events'):
        validator.validate(scenario, asset.expected)


def test_runtime_parser_rejects_world_control_of_sumo_npc(site):
    asset, _ = site
    scenario = copy.deepcopy(asset.scenario)
    scenario['experiment_world_events'] = [{
        'at_s': 1, 'entity_id': 'oncoming_01',
        'action': 'set_vehicle_speed', 'speed_kmh': 0}]
    with pytest.raises(ValueError, match='Unknown scenario fields'):
        MultiScenario.from_dict(scenario)


def test_frozen_full_network_tasks_have_three_authored_encounters():
    root = ROOT / 'vehiclearena/evaluation/experiments/scenarios'
    catalog = json.loads((root / 'catalog.json').read_text())
    entries = [e for e in catalog['entries'] if
               e['source_template_id'].split('__')[0] == 'chassis_full_network_route']
    assert len(entries) == 9
    for entry in entries:
        scenario = json.loads((root / entry['scenario']).read_text())
        expected = json.loads((root / entry['expected']).read_text())
        assertion = next(a for a in expected['setup_assertions']
                         if a['kind'] == 'native_full_network_oncoming_stream')
        assert assertion['minimum_runtime_encounters'] == 3
        cohort_ids = scenario['experiment_scene'][
            'related_background_traffic']['longitudinal_cohort_vehicle_ids']
        assert 3 <= len(cohort_ids) <= 8
        assert len(scenario['vehicles']) == 17 + len(cohort_ids)
        assert 'ego_centric_oncoming' in entry['tags']
        assert 'sumo_native_oncoming' in entry['tags']
        assert not scenario.get('experiment_world_events')


@pytest.mark.parametrize('scene_id', [
    'basic_037_full_network_route',
    'basic_038_full_network_route__chongqing_jiefangbei',
    'basic_039_full_network_route__wuhan_hankou',
    'basic_040_full_network_route__xian_zhonglou',
    'basic_188_full_network_route__expansion__haikou_longhua',
    'basic_189_full_network_route__expansion__lasa_chengguan',
    'basic_190_full_network_route__expansion__losangeles_downtown',
    'basic_191_full_network_route__expansion__nantong_chongchuan',
    'basic_192_full_network_route__expansion__shenzhen_nanshan',
])
def test_native_sumo_owns_each_oncoming_actor_after_initialization(scene_id):
    path = ROOT / 'vehiclearena/evaluation/experiments/scenarios/Basic' / scene_id / 'scenario.json'
    scene = json.loads(path.read_text())
    peers = [v['vehicle_id'] for v in scene['vehicles']
             if v['vehicle_id'].startswith('oncoming_')]
    scene.update(total_time_s=3, physics_only_mode=True,
                 enable_driving_evaluation=False, stop_when_all_vehicles_terminal=False)
    for vehicle in scene['vehicles']:
        vehicle['agent_config'] = {'type': 'sumo'}
    engine = ExperimentTrackedEngine(MultiScenario.from_dict(scene))
    result = engine.run({})
    assert not [event for event in result._physics_events
                if event['type'] == 'experiment_world_event']
    for entity in peers:
        rows = [row for row in result._vehicle_trajectory if row['vehicle_id'] == entity]
        assert any(row['speed_kmh'] > 0 for row in rows)
        assert max(row['distance_traveled_m'] for row in rows) > 0
