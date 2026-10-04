"""Platoon traffic must not become a permanently scripted obstacle."""
import copy
import json
from pathlib import Path

from evaluation.experiments.manifest import load_manifest
from evaluation.experiments.scene_validator import SceneValidator
from evaluation.experiments.scene_catalog import Topology, _platoon_pressure
from evaluation.experiments.time_window_calibration import (
    load_calibration_registry, scenario_physical_fingerprint,
)


def test_generator_does_not_reintroduce_scripted_lead_stop():
    root = Path(__file__).resolve().parents[1]
    topology = Topology('beijing_guomao', root/'simulation/road_networks')
    asset = _platoon_pressure(topology, 'Basic', 'native-platoon-test', title='Native traffic')
    assert not asset.scenario.get('experiment_world_events')
    assert not any(x['kind'] == 'scheduled_world_event' for x in asset.expected['setup_assertions'])
    assert not any(x['kind'] == 'scheduled_speed_change_observed' for x in asset.expected['runtime_acceptance'])


def test_platoon_catalog_manifests_and_baselines_use_native_traffic():
    root = Path(__file__).resolve().parents[1]
    cat = root/'evaluation/experiments/scenarios'
    entries = [e for e in json.loads((cat/'catalog.json').read_text())['entries']
               if 'platoon_pressure' in e['scene_id']]
    assert len(entries) == 9
    manifest = load_manifest(root/'evaluation/experiments/manifests/Basic.manifest.json')
    frozen = {v.base_scenario_id: v.scenario for v in manifest.variants}
    registry = load_calibration_registry()
    validator = SceneValidator(root/'simulation/road_networks')
    for entry in entries:
        scene = json.loads((cat/entry['scenario']).read_text())
        expected = json.loads((cat/entry['expected']).read_text())
        assert not scene.get('experiment_world_events')
        assert 'native_sumo_following' in entry['tags']
        assert 'lead_braking' not in scene['experiment_scene']['interaction_tags']
        assert not any(x['kind'] == 'scheduled_world_event' for x in expected['setup_assertions'])
        assert not any(x['kind'] == 'scheduled_speed_change_observed' for x in expected['runtime_acceptance'])
        for vehicle in scene['vehicles']:
            if vehicle['vehicle_id'] != 'ego':
                assert vehicle['agent_config']['type'] == 'sumo'
                assert not {'desired_speed_kmh', 'target_speed_kmh'} & vehicle['initial_physical_state'].keys()
        validator.validate(scene, expected)
        executable = copy.deepcopy(scene)
        executable['experiment_scene']['setup_assertions'] = expected['setup_assertions']
        assert frozen[entry['scene_id']] == executable
        calibration = registry['entries'][entry['scene_id']]
        assert calibration['physical_fingerprint'] == scenario_physical_fingerprint(scene)
        assert calibration['successful_case_count'] == calibration['case_count'] == 1
