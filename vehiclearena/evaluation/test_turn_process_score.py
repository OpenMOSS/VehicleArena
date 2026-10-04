"""Connector signalling and geometry-bounded braking regression tests."""
import math
from types import SimpleNamespace as NS

import pytest

from evaluation.driving_process_score import (
    DrivingProcessScoreConfig, DrivingProcessScoreTracker,
    _connector_context, _curve_reference_speed, calculate_driving_process_score,
)
from evaluation.test_driving_process_score import _report, _sample


def turn_samples(direction='left', signal='correct'):
    result = []
    for i in range(71):
        on = 5 <= i < 20
        if signal == 'late':
            on = 8 <= i < 20
        if signal == 'held':
            on = i >= 5
        lamps = {direction + '_indicator': on}
        if signal == 'missing':
            lamps = {}
        if signal == 'wrong':
            lamps = {('right' if direction == 'left' else 'left')+'_indicator': on}
        if signal == 'both':
            lamps = {'left_indicator': on, 'right_indicator': on}
        result.append(_sample(i/10, active_connector_id='turn-a' if 10 <= i < 20 else '',
                              active_connector_turn=direction if 10 <= i < 20 else None, **lamps))
    return result


@pytest.mark.parametrize('direction', ['left', 'right'])
@pytest.mark.parametrize('signal,points', [('correct', 0), ('missing', 10), ('wrong', 10), ('both', 10), ('late', 10), ('held', 2)])
def test_turn_direction_lead_and_cancellation(direction, signal, points):
    result = calculate_driving_process_score(_report(), samples=turn_samples(direction, signal))
    assert result['deduction_total'] == points
    assert result['observed_turn_count'] == 1
    assert result['turn_events'][0]['end_time_s'] == 2
    if points:
        assert result['deductions'][0]['evidence']['connector_id'] == 'turn-a'


def test_straight_connector_does_not_require_a_turn_signal():
    result = calculate_driving_process_score(_report(), samples=turn_samples('straight', 'missing'))
    assert result['observed_turn_count'] == 0 and result['deduction_total'] == 0


def test_entry_sample_requires_signal_to_remain_on():
    samples = turn_samples()
    samples[10]['left_indicator'] = False
    assert calculate_driving_process_score(_report(), samples=samples)['deduction_total'] == 10


def test_initial_connector_or_insufficient_lead_history_is_not_invented_violation():
    for samples in (turn_samples(signal='missing')[10:25], turn_samples(signal='missing')[8:25]):
        assert calculate_driving_process_score(_report(), samples=samples)['deduction_total'] == 0


def test_reentering_same_connector_counts_as_another_turn():
    samples = turn_samples(signal='missing')
    for i in range(40, 50):
        samples[i].update(active_connector_id='turn-a', active_connector_turn='left')
    result = calculate_driving_process_score(_report(), samples=samples)
    assert result['observed_turn_count'] == 2 and result['deduction_total'] == 20


@pytest.mark.parametrize('planned,distance,expected', [('left', 15, 0), ('left', 100, 2), ('right', 15, 2), (None, None, 2)])
def test_lane_change_can_keep_signal_for_nearby_same_direction_turn(planned, distance, expected):
    samples = [_sample(i/10, left_indicator=True,
                       is_changing_lane=i == 6, current_lane=0, target_lane=1,
                       planned_connector_turn=planned,
                       distance_to_planned_connector_m=distance) for i in range(40)]
    result = calculate_driving_process_score(_report(), samples=samples)
    assert result['deduction_total'] == expected


def test_lane_change_then_turn_has_one_final_cancellation_check():
    samples = turn_samples(signal='held')
    samples[6].update(is_changing_lane=True, target_lane=1)
    for s in samples:
        s['left_indicator'] = True
    result = calculate_driving_process_score(_report(), samples=samples)
    assert result['deduction_total'] == 2
    assert result['deductions'][0]['evidence']['source'] == 'connector_turn'


def test_connector_lane_number_change_is_not_a_lane_change():
    samples = [_sample(0, current_segment='road', current_lane_id='lane-a', current_lane=0),
               _sample(.1, current_segment='road', current_lane_id='lane-b', current_lane=1,
                       active_connector_id='turn-a', active_connector_turn='left')]
    result = calculate_driving_process_score(_report(), samples=samples)
    assert result['observed_lane_change_count'] == 0


def curve_fixture(distance=10, speed=36, active=False, straight=False):
    points = [[10*math.cos(i*math.pi/20), 10*math.sin(i*math.pi/20)] for i in range(11)]
    if straight:
        points = [[0, 0], [10, 0], [20, 0]]
    connector = {'id': 'c', 'turn': 'left', 'from_lane': 'lane', 'centerline_xy': points}
    runtime = NS(_connector_by_id={'c': connector},
                 _lane_by_id={'lane': {'segment_id': 'road', 'direction': 1, 'length_m': 200}})
    vehicle = NS(active_connector_id='c' if active else '', planned_connector_id='c',
                 current_lane_id='lane', edge_progress=1-distance/200, current_speed_kmh=speed,
                 length_m=4.6, acceleration_mps2=-5, signal_state=NS(), current_segment='', current_lane=-1)
    manager = NS(_lane_geometry=runtime, get_state=lambda vid: vehicle,
                 _build_env_view=lambda *args: NS(speed_limit_kmh=60),
                 get_driving_awareness=lambda *args, **kwargs: {})
    return manager, vehicle, connector


@pytest.mark.parametrize('distance,speed,active,straight,justified', [
    (10, 36, False, False, True), (80, 36, False, False, False),
    (10, 10, False, False, False), (0, 36, True, False, True),
    (0, 10, True, False, False), (10, 36, False, True, False),
])
def test_curve_braking_is_bounded_by_speed_distance_and_geometry(distance, speed, active, straight, justified):
    manager, vehicle, _ = curve_fixture(distance, speed, active, straight)
    context = _connector_context(manager, vehicle, DrivingProcessScoreConfig(), {})
    assert context['curve_braking_context']['justified'] is justified
    tracker = DrivingProcessScoreTracker(manager, {'ego': NS()}, ['ego'])
    for i in range(6):
        tracker.observe(i/10)
    sample = tracker._traces['ego'].samples[0]
    assert sample['unnecessary_hard_brake'] is not justified
    result = tracker.finalize({'ego': _report()})['ego']
    assert bool(result['curve_braking_exemptions']) is justified
    assert ('unnecessary_hard_brake' in [d['type'] for d in result['deductions']]) is not justified


def test_curve_reference_uses_real_curvature_and_ignores_missing_geometry():
    _, _, connector = curve_fixture()
    config = DrivingProcessScoreConfig()
    assert _curve_reference_speed(connector, config) == pytest.approx(18)
    assert _curve_reference_speed({}, config) is None


def test_stale_plan_on_another_road_does_not_excuse_braking():
    manager, vehicle, _ = curve_fixture()
    manager._lane_geometry._lane_by_id['other'] = {'segment_id': 'elsewhere', 'direction': 1, 'length_m': 200}
    vehicle.current_lane_id = 'other'
    assert not _connector_context(manager, vehicle, DrivingProcessScoreConfig(), {})['curve_braking_context']['justified']


def test_hard_acceleration_threshold_scales_with_chassis_capability():
    # compact (3.2 m/s²): threshold 2.88 — full-throttle 3.0 still flags.
    manager, vehicle, _ = curve_fixture()
    vehicle.max_acceleration_mps2 = 3.2
    vehicle.acceleration_mps2 = 3.0
    tracker = DrivingProcessScoreTracker(manager, {'ego': NS()}, ['ego'])
    tracker.observe(0.0)
    sample = tracker._traces['ego'].samples[0]
    assert sample['hard_acceleration'] is True
    assert sample['hard_acceleration_threshold_mps2'] == pytest.approx(2.88)

    # freight (1.8 m/s²): threshold 1.62 — 1.5 m/s² no longer flags.
    manager, vehicle, _ = curve_fixture()
    vehicle.max_acceleration_mps2 = 1.8
    vehicle.acceleration_mps2 = 1.5
    tracker = DrivingProcessScoreTracker(manager, {'ego': NS()}, ['ego'])
    tracker.observe(0.0)
    sample = tracker._traces['ego'].samples[0]
    assert sample['hard_acceleration'] is False
    assert sample['hard_acceleration_threshold_mps2'] == pytest.approx(1.62)

    # Chassis capability unknown: fixed config fallback applies.
    manager, vehicle, _ = curve_fixture()
    vehicle.acceleration_mps2 = 2.9
    tracker = DrivingProcessScoreTracker(manager, {'ego': NS()}, ['ego'])
    tracker.observe(0.0)
    sample = tracker._traces['ego'].samples[0]
    assert sample['hard_acceleration'] is True
    assert sample['hard_acceleration_threshold_mps2'] == 2.8
