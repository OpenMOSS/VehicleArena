"""Driver diversity stays reproducible, native and isolated from agent control."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from simulation.native_npc_behavior import (
    PARAMETERS, RANGES, sample_driver, validate_behavior, with_native_npc_behavior)
from simulation.sumo_traffic_manager import SumoPhysicsConfig, SumoTrafficManager
from simulation.multi_sim_engine import MultiScenario
from evaluation.experiments.telemetry import ExperimentTrackedEngine
from evaluation.experiments.time_window_calibration import scenario_physical_fingerprint

SCENE = Path(__file__).parent / 'fixtures/pre_pull_catalog/Basic/basic_005_lane_change/scenario.json'


def test_sampling_reproducible_independent_and_bounded():
    samples = [sample_driver(42, str(i)) for i in range(100)]
    assert samples == [sample_driver(42, str(i)) for i in range(100)]
    assert samples != [sample_driver(43, str(i)) for i in range(100)]
    assert {s['profile'] for s in samples} == set(RANGES)
    for s in samples:
        for key, (low, high) in zip(PARAMETERS, RANGES[s['profile']]):
            assert low <= s[key] <= high


@pytest.mark.parametrize('seed', [-1, 2**31, True, 1.2, '1'])
def test_bad_seeds_rejected(seed):
    with pytest.raises(ValueError):
        with_native_npc_behavior(json.loads(SCENE.read_text()), seed)


def test_configuration_and_reference_identity():
    raw = json.loads(SCENE.read_text())
    original = copy.deepcopy(raw)
    configured = with_native_npc_behavior(raw, 42)
    assert raw == original
    behavior = configured['sumo_config']['npc_behavior']
    assert 'ego' not in behavior['vehicle_ids']
    MultiScenario.from_dict(configured)
    reference = copy.deepcopy(configured)
    for v in reference['vehicles']:
        v['agent_config'] = {'type': 'sumo'}
    assert MultiScenario.from_dict(reference).sumo_config == MultiScenario.from_dict(configured).sumo_config
    assert scenario_physical_fingerprint(raw) != scenario_physical_fingerprint(configured)
    assert scenario_physical_fingerprint(configured) != scenario_physical_fingerprint(with_native_npc_behavior(raw, 43))
    behavior['vehicle_ids'].append('ego')
    with pytest.raises(ValueError, match='existing SUMO'):
        MultiScenario.from_dict(configured)


def test_no_actuator_commands_or_agent_changes():
    manager = object.__new__(SumoTrafficManager)
    raw = with_native_npc_behavior(json.loads(SCENE.read_text()), 42)
    manager.sumo_config = SumoPhysicsConfig.from_dict(raw['sumo_config'])
    manager.npc_behavior_assignments = {}
    manager._sumo = NS(vehicle=Mock())
    for vid, llm, crashed in [('ego', True, False), ('ego', False, False), ('slow_lead', False, True)]:
        manager._initialize_native_driver(NS(vehicle_id=vid, is_llm=llm, is_crashed=crashed))
    assert not manager._sumo.vehicle.method_calls
    manager._initialize_native_driver(NS(vehicle_id='slow_lead', is_llm=False, is_crashed=False))
    names = {c[0] for c in manager._sumo.vehicle.method_calls}
    assert names == {'setSpeedFactor', 'setTau', 'setMinGap', 'setImperfection', 'setParameter'}


def test_real_sumo_seeded_motion_and_effective_parameters(monkeypatch):
    initialize = SumoTrafficManager._initialize_native_driver
    def checked_initialize(manager, vehicle):
        initialize(manager, vehicle)
        vid = vehicle.vehicle_id
        if vid in manager.npc_behavior_assignments:
            expected = manager.npc_behavior_assignments[vid]
            api = manager._sumo.vehicle
            assert api.getTau(vid) == pytest.approx(expected['tau'])
            assert api.getSpeedFactor(vid) == pytest.approx(expected['speedFactor'])
            assert api.getMinGap(vid) == pytest.approx(expected['minGap'])
            assert api.getImperfection(vid) == pytest.approx(expected['sigma'])
            # SUMO generic parameter strings use its two-decimal output precision.
            assert float(api.getParameter(vid, 'laneChangeModel.lcSpeedGain')) == pytest.approx(expected['lcSpeedGain'], abs=0.0051)
            assert float(api.getParameter(vid, 'laneChangeModel.lcCooperative')) == pytest.approx(expected['lcCooperative'], abs=0.0051)
            assert api.getSpeedMode(vid) == 31
            assert api.getLaneChangeMode(vid) == 1621
    monkeypatch.setattr(SumoTrafficManager, '_initialize_native_driver', checked_initialize)
    trajectories = []
    for seed in (42, 42, 43):
        raw = with_native_npc_behavior(json.loads(SCENE.read_text()), seed)
        raw.update(total_time_s=5, physics_only_mode=True, enable_driving_evaluation=False)
        engine = ExperimentTrackedEngine(MultiScenario.from_dict(raw))
        result = engine.run({})
        assert not engine.agent_callback_errors
        assignments = engine.traffic_mgr.npc_behavior_assignments
        assert assignments['slow_lead']['seed'] == seed
        assert 'ego' not in assignments
        assert result.npc_behavior_assignments == assignments
        trajectories.append([(r['vehicle_id'], r['time_s'], r['pose_x_m'], r['pose_y_m'], r['speed_kmh'])
                             for r in engine._vehicle_trajectory])
    assert trajectories[0] == trajectories[1]
    assert trajectories[0] != trajectories[2]


@pytest.mark.parametrize('value', [None, [], {'seed': 1}, {'schema': 'native-npc-mixed-v1', 'seed': 1, 'vehicle_ids': ['x', 'x']}])
def test_invalid_behavior_rejected(value):
    with pytest.raises(ValueError):
        validate_behavior(value)


def test_prepare_catalog_is_isolated_and_drops_stale_summary(tmp_path):
    from prepare_native_npc_catalog import prepare, check_native_calibration
    from evaluation.experiments.time_window_calibration import CALIBRATION_SCHEMA
    catalog = Path(__file__).parent / 'experiments/scenarios'
    target = tmp_path / 'new'
    scene_id = 'basic_001_crosswalk'
    original = (catalog / 'Basic' / scene_id / 'scenario.json').read_bytes()
    assert prepare(catalog, target, 42, [scene_id]) == 1
    metadata = json.loads((target / 'catalog.json').read_text())
    assert metadata['scene_count'] == 1
    assert metadata['counts'] == {'Basic': 1}
    raw = json.loads((target / 'Basic' / scene_id / 'scenario.json').read_text())
    assert 'time_window_calibration' not in raw
    MultiScenario.from_dict(raw)
    assert (catalog / 'Basic' / scene_id / 'scenario.json').read_bytes() == original
    with pytest.raises(FileExistsError):
        prepare(catalog, target, 42, [scene_id])
    with pytest.raises(ValueError, match='calibration-registry'):
        check_native_calibration(target, None)
    registry = tmp_path / 'registry.json'
    baseline = {'physical_fingerprint': scenario_physical_fingerprint(raw),
                'successful_case_count': 1, 'time_limit_s': raw['total_time_s']}
    registry.write_text(json.dumps({'schema': CALIBRATION_SCHEMA, 'entries': {scene_id: baseline}}))
    check_native_calibration(target, registry)
    raw['sumo_config']['npc_behavior']['seed'] = 43
    (target / 'Basic' / scene_id / 'scenario.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='Fresh matching'):
        check_native_calibration(target, registry)
