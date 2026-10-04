"""Request-only six-grade scoring: no live model calls."""
import copy
import json
from types import SimpleNamespace

import pytest

from evaluation.request_satisfaction import (
    GRADE_SCORES, REVISION, score_submission, validate_contract,
    validate_judge_trigger, available_judge_triggers,
)
from evaluation.personal_agent import (
    PersonalAgentRuntime, PassengerJudgeRuntime,
    _validate_trigger_horizon,
    aggregate_passenger_judgements,
)
from test_passenger_orchestration import Client, response, make_callback, wake
from vehiclearena import VehicleWorld


def contract(kind='one_shot'):
    return dict(core=['Close the window'], secondary=[], request_kind=kind,
                expected_response_s=1.0, valid_for_s=5.0)


def verdict(grade, kind='one_shot', fulfilled=0.5):
    return response('submit_passenger_judgement', dict(
        grade=grade, reason='Window state observed in physical evidence',
        core_statuses=['met'] if grade in ('A', 'B', 'C') else ['unmet'],
        secondary_statuses=[],
        fulfilled_at_s=fulfilled, request_kind=kind,
        completion_status='completed' if grade in ('A','B') else 'pending',
        completion_reason='Window state inspected'))


def check(core=None, secondary=None, fulfilled=None):
    arguments = dict(
        reason='Intermediate physical evidence inspected',
        core_statuses=core or ['met'],
        secondary_statuses=secondary or [],
    )
    if fulfilled is not None:
        arguments['fulfilled_at_s'] = fulfilled
    return response('submit_passenger_check', arguments)


@pytest.mark.parametrize('grade,points', GRADE_SCORES.items())
def test_fixed_grade_mapping(grade, points):
    result = score_submission(dict(grade=grade, reason='evidence'))
    assert result['overall_score_100'] == points
    assert result['dimension_scores_100'] == {'request_response': points}


@pytest.mark.parametrize('args', [dict(grade='A+', reason='x'), dict(grade='NA',reason='x'),
                                dict(grade='A',reason=''), dict(request_response_score=90,reason='x')])
def test_invalid_submissions_rejected(args):
    with pytest.raises(ValueError): score_submission(args)


def test_pa_requires_and_freezes_contract():
    immediate = dict(core=['Close the window'], secondary=[])
    client = Client([response('send_immediate_request', dict(
        message='Could you close the window?', immediate_request=immediate,
        request_kind='one_shot',
        expected_response_s=1.0, valid_for_s=5.0))])
    pa = PersonalAgentRuntime('ego',client)
    request = pa.generate({'sim_time_s':0})
    immediate['core'][0] = 'changed'
    assert request.as_dict()['acceptance_criteria']['core'] == ['Close the window']
    schema = client.tool_schemas[0][0]['function']['parameters']
    assert 'immediate_request' in schema['required']
    assert 'acceptance_criteria' not in schema['properties']
    pa = PersonalAgentRuntime('ego',Client([
        response('send_passenger_request_removed',dict(message='No contract')),
        response('finish', {}),
    ]))
    assert pa.generate({'sim_time_s':0}) is None


@pytest.mark.parametrize('trigger', [
    {'trigger_type': 'time', 'condition': 'after_delay',
     'after_s': 5.0, 'timeout_s': 10.0},
    {'trigger_type': 'map', 'condition': 'exit_next_intersection',
     'timeout_s': 20.0},
    {'trigger_type': 'vehicle_state', 'condition': 'vehicle_stopped',
     'hold_for_s': 0.5, 'timeout_s': 10.0},
])
def test_supported_judge_triggers_validate(trigger):
    validate_judge_trigger(trigger)


@pytest.mark.parametrize('trigger', [
    {'trigger_type': 'map', 'condition': 'after_delay',
     'after_s': 1.0, 'timeout_s': 2.0},
    {'trigger_type': 'time', 'condition': 'after_delay',
     'after_s': 3.0, 'timeout_s': 2.0},
    {'trigger_type': 'map', 'condition': 'exit_next_intersection',
     'after_s': 1.0, 'timeout_s': 2.0},
])
def test_invalid_judge_triggers_are_rejected(trigger):
    with pytest.raises(ValueError):
        validate_judge_trigger(trigger)


def _triggered_request_response(*, message=(
        'The cabin is chilly, so warm it a little now; in two seconds, turn on '
        'the air conditioning and set it to 22 degrees.'), after_s=2.0,
                                trigger_type='time', condition='after_delay'):
    trigger = {
        'trigger_type': trigger_type, 'condition': condition,
        'timeout_s': 10.0,
    }
    if condition == 'after_delay':
        trigger['after_s'] = after_s
    return response('send_triggered_request', {
        'message': message,
        'immediate_request': {
            'core': ['Warm the cabin a little'], 'secondary': []},
        'triggered_request': {
            'core': ['Turn on the air conditioning and set 22 degrees'],
            'secondary': [],
        },
        'request_kind': 'one_shot',
        'expected_response_s': 1.0, 'valid_for_s': 5.0,
        'judge_trigger': trigger,
    })


def test_pa_preserves_natural_message_while_trigger_remains_structured():
    client = Client([_triggered_request_response()])
    pa = PersonalAgentRuntime('ego', client)
    request = pa.generate({
        'sim_time_s': 1.0,
        'temporal_task_context': {'remaining_episode_s': 10.0},
    })

    assert request is not None
    assert request.message == (
        'The cabin is chilly, so warm it a little now; in two seconds, turn on '
        'the air conditioning and set it to 22 degrees.')
    properties = client.tool_schemas[0][1]['function']['parameters'][
        'properties']
    assert 'triggered_request' in properties
    assert request.criterion_phases['triggered']['core_indices'] == [1]


def test_pa_rejects_time_trigger_beyond_remaining_episode():
    client = Client([_triggered_request_response(after_s=8.0),
                     _triggered_request_response(after_s=8.0)])
    pa = PersonalAgentRuntime('ego', client)
    request = pa.generate({
        'sim_time_s': 1.0,
        'temporal_task_context': {'remaining_episode_s': 5.0},
    })

    assert request is None
    receipt = json.loads(pa.state['turn_log'][-1]['tool_receipts'][0][
        'content'])
    assert receipt['error'] == 'invalid_passenger_request'
    assert 'remaining episode' in receipt['detail']
    assert len(client.calls) == 2


def test_pa_does_not_parse_message_timing_when_trigger_is_structured():
    submitted = _triggered_request_response(
        message='When the vehicle stops, please help with the cabin.',
        trigger_type='map', condition='exit_next_intersection')
    client = Client([submitted])
    pa = PersonalAgentRuntime('ego', client)
    request = pa.generate({
        'sim_time_s': 0.0,
        'temporal_task_context': {'remaining_episode_s': 20.0},
    })

    assert request is not None
    assert request.judge_trigger['condition'] == 'exit_next_intersection'
    assert len(client.calls) == 1
    receipt = json.loads(pa.state['turn_log'][-1]['tool_receipts'][0]['content'])
    assert receipt['success'] is True


def test_pa_does_not_infer_a_trigger_from_message_timing():
    submitted = response('send_immediate_request', {
        'message': ('After the next intersection, turn on the steering '
                    'wheel heater.'),
        'immediate_request': {
            'core': [('After the next intersection, turn on the steering '
                      'wheel heater.')], 'secondary': []},
        'request_kind': 'one_shot',
        'expected_response_s': 2.0, 'valid_for_s': 8.0,
    })
    client = Client([submitted])
    pa = PersonalAgentRuntime('ego', client)

    request = pa.generate({
        'sim_time_s': 0.0,
        'temporal_task_context': {'remaining_episode_s': 20.0},
    })

    assert request is not None
    assert request.judge_trigger is None
    assert request.message == (
        'After the next intersection, turn on the steering wheel heater.')
    assert len(client.calls) == 1


@pytest.mark.parametrize('message', [
    'Please drive carefully through the intersection.',
    'We have been stopped for a while. Please turn on the heater.',
])
def test_pa_allows_non_temporal_intersection_and_stop_context(message):
    valid = response('send_immediate_request', {
        'message': message,
        'immediate_request': {
            'core': [message], 'secondary': [],
        },
        'request_kind': 'one_shot',
        'expected_response_s': 2.0, 'valid_for_s': 8.0,
    })
    request = PersonalAgentRuntime('ego', Client([valid])).generate({
        'sim_time_s': 0.0,
        'temporal_task_context': {'remaining_episode_s': 20.0},
    })

    assert request is not None
    assert request.judge_trigger is None


def test_pa_retries_once_after_validation_error_and_accepts_repair():
    invalid = _triggered_request_response(after_s=8.0)
    repaired = _triggered_request_response(after_s=2.0)
    client = Client([invalid, repaired])
    pa = PersonalAgentRuntime('ego', client)

    request = pa.generate({
        'sim_time_s': 0.0,
        'temporal_task_context': {'remaining_episode_s': 5.0},
    })

    assert request is not None
    assert len(client.calls) == 2
    first_receipt = json.loads(
        pa.state['turn_log'][-1]['tool_receipts'][0]['content'])
    assert first_receipt['retry_allowed'] is True
    final_receipt = json.loads(
        pa.state['turn_log'][-1]['tool_receipts'][-1]['content'])
    assert final_receipt == {
        'success': True, 'request_id': 'pa-ego-0001',
        'sheet_action': 'create', 'replaces_request_id': None}


def test_pa_enforces_hard_request_design_outcome_count():
    too_easy = response('send_passenger_request', {
        'immediate_request': {
            'message': 'Turn on the air conditioning.',
            'core': ['Air conditioning is on'], 'secondary': []},
        'request_kind': 'one_shot',
        'expected_response_s': 2.0, 'valid_for_s': 5.0,
    })
    repaired = response('send_passenger_request', {
        'immediate_request': {
            'message': ('Turn on the air conditioning, set it to 23 degrees, '
                        'and lower the music volume.'),
            'core': ['Air conditioning is on',
                     'Air conditioning temperature is 23 degrees',
                     'Music volume is lower'],
            'secondary': []},
        'request_kind': 'one_shot',
        'expected_response_s': 2.0, 'valid_for_s': 5.0,
    })
    client = Client([too_easy, repaired])
    request = PersonalAgentRuntime('ego', client).generate({
        'sim_time_s': 0.0,
        'request_design': {
            'difficulty': 'hard', 'pattern': 'compound_immediate',
            'min_explicit_outcomes': 3, 'judge_trigger': 'optional'},
    })
    assert request is not None
    assert len(request.acceptance_criteria['core']) == 3
    assert len(client.calls) == 2


@pytest.mark.parametrize('message', [
    'Later, turn the music down.',
    '稍后，调低音乐音量。',
    'When stopped, close the window.',
])
def test_pa_accepts_natural_timing_words_without_text_parsing(message):
    client = Client([response('send_passenger_request', {
        'immediate_request': {
            'message': message, 'core': ['Requested state is achieved'],
            'secondary': []},
        'request_kind': 'one_shot',
        'expected_response_s': 2.0, 'valid_for_s': 8.0,
    })])
    request = PersonalAgentRuntime('ego', client).generate({'sim_time_s': 0.0})
    assert request is not None
    assert request.message == message


def test_environment_trigger_requires_a_currently_observable_horizon():
    with pytest.raises(ValueError, match='next intersection'):
        _validate_trigger_horizon({
            'trigger_type': 'map', 'condition': 'exit_next_intersection',
            'timeout_s': 10.0,
        }, {'temporal_task_context': {
            'remaining_episode_s': 20.0,
            'intersection_ahead_known': False,
        }})
    _validate_trigger_horizon({
        'trigger_type': 'vehicle_state', 'condition': 'vehicle_stopped',
        'hold_for_s': 0.5, 'timeout_s': 10.0,
    }, {'temporal_task_context': {
        'remaining_episode_s': 20.0, 'vehicle_stopped': False,
    }})


@pytest.mark.parametrize('trigger', [
    {'trigger_type': 'map', 'condition': 'approach_next_intersection',
     'distance_m': 30, 'timeout_s': 10},
    {'trigger_type': 'map', 'condition': 'enter_next_intersection',
     'timeout_s': 10},
    {'trigger_type': 'vehicle_state', 'condition': 'vehicle_resumed_moving',
     'hold_for_s': 1, 'timeout_s': 10},
    {'trigger_type': 'vehicle_state', 'condition': 'speed_threshold_held',
     'comparison': 'above', 'speed_kmh': 40, 'hold_for_s': 1,
     'timeout_s': 10},
    {'trigger_type': 'map', 'condition': 'distance_to_destination_below',
     'distance_m': 100, 'timeout_s': 10},
    {'trigger_type': 'environment', 'condition': 'weather_changed_to',
     'weather_condition': 'foggy', 'timeout_s': 10},
    {'trigger_type': 'environment', 'condition': 'daynight_became_dark',
     'timeout_s': 10},
])
def test_new_structured_triggers_validate(trigger):
    validate_judge_trigger(trigger)


def test_pa_sees_only_currently_registerable_design_triggers():
    context = {'remaining_episode_s': 20, 'max_after_delay_s': 5,
               'intersection_ahead_known': False,
               'remaining_distance_m': None,
               'future_weather_conditions': [],
               'future_dark_transition': False}
    design = {'judge_trigger': 'required',
              'allowed_trigger_conditions': [
                  'enter_next_intersection', 'vehicle_stopped',
                  'weather_changed_to']}
    options = available_judge_triggers(context, design)
    assert [option['condition'] for option in options] == ['vehicle_stopped']
    observation = {'sim_time_s': 0, 'temporal_task_context': context,
                   'request_design': design,
                   'available_judge_triggers': options}
    client = Client([response('finish', {})])
    assert PersonalAgentRuntime('ego', client).generate(observation) is None
    schema = client.tool_schemas[0][1]['function']['parameters']['properties']['judge_trigger']
    assert schema['properties']['condition']['enum'] == ['vehicle_stopped']
    assert set(schema['properties']) == {
        'trigger_type', 'condition', 'timeout_s', 'hold_for_s'}
    with pytest.raises(ValueError, match='not available'):
        _validate_trigger_horizon({
            'trigger_type': 'environment', 'condition': 'weather_changed_to',
            'weather_condition': 'rainy', 'timeout_s': 10}, observation)


def test_distance_trigger_offered_only_before_threshold_and_weather_only_if_scheduled():
    context = {'remaining_episode_s': 30, 'max_after_delay_s': 10,
               'intersection_ahead_known': True,
               'inside_intersection': False,
               'next_intersection_distance_m': 80,
               'remaining_distance_m': 200,
               'future_weather_conditions': ['rainy'],
               'future_dark_transition': True}
    options = available_judge_triggers(context)
    names = {option['condition'] for option in options}
    assert {'approach_next_intersection', 'enter_next_intersection',
            'vehicle_resumed_moving', 'speed_threshold_held',
            'distance_to_destination_below', 'weather_changed_to',
            'daynight_became_dark'} <= names
    assert next(option for option in options if option['condition'] ==
                'approach_next_intersection')['distance_m']['max_exclusive'] == 80
    with pytest.raises(ValueError, match='range'):
        _validate_trigger_horizon({
            'trigger_type': 'map', 'condition': 'approach_next_intersection',
            'distance_m': 80, 'timeout_s': 10}, {
                'sim_time_s': 0, 'temporal_task_context': context,
                'available_judge_triggers': options})


def test_production_trigger_options_require_route_time_after_activation():
    context = {
        'remaining_episode_s': 12.0,
        'estimated_free_flow_remaining_s': 8.0,
        'estimated_free_flow_speed_mps': 10.0,
        'judge_observation_window_s': 3.0,
        'max_after_delay_s': 4.5,
        'intersection_ahead_known': True,
        'inside_intersection': False,
        'next_intersection_distance_m': 10.0,
        'next_intersection_exit_distance_m': 20.0,
        'remaining_distance_m': 80.0,
        'future_weather_events': [
            {'weather_condition': 'rainy', 'after_s': 6.0}],
        'future_dark_after_s': 6.0,
        'guaranteed_trigger_conditions': {},
    }
    names = {item['condition'] for item in available_judge_triggers(context)}
    assert {'enter_next_intersection', 'exit_next_intersection',
            'approach_next_intersection', 'distance_to_destination_below'} <= names
    assert not {'vehicle_stopped', 'vehicle_resumed_moving',
                'speed_threshold_held', 'weather_changed_to',
                'daynight_became_dark'} & names
    distance = next(item for item in available_judge_triggers(context)
                    if item['condition'] == 'distance_to_destination_below')
    assert distance['distance_m']['min_exclusive'] == 30.5

    context.update(remaining_distance_m=25.0,
                   estimated_free_flow_remaining_s=2.5)
    assert available_judge_triggers(context) == []


def test_intermediate_tool_has_no_grade_and_computes_one_shot_timing():
    client = Client([check(fulfilled=2)])
    judge = PassengerJudgeRuntime('ego',client)
    result = judge.judge(dict(schema_version='passenger-judge-evidence-v3',
        request=dict(request_id='r',created_at_s=0,acceptance_criteria=contract()),
        judged_at_s=3,final_window=False))
    assert result['judged'] is True
    assert result['grade'] == 'B'
    assert result['completion_status'] == 'completed'
    assert result['promoted_to_final'] is True
    fields = client.tool_schemas[0][0]['function']['parameters']['properties']
    required = client.tool_schemas[0][0]['function']['parameters']['required']
    assert client.tool_schemas[0][0]['function']['name'] == 'submit_passenger_check'
    assert 'grade' not in fields
    assert {'core_statuses', 'secondary_statuses'} <= set(required)
    assert 'driving_rationality_score' not in fields
    assert 'environmental_impact_score' not in fields


def test_trigger_activation_is_the_response_clock_origin():
    client = Client([check(fulfilled=10.5)])
    judge = PassengerJudgeRuntime('ego', client)
    result = judge.judge(dict(schema_version='passenger-judge-evidence-v3',
        request=dict(request_id='r', created_at_s=0,
                     acceptance_criteria=contract()),
        response_origin_s=10.0, judged_at_s=10.6, final_window=False,
        judge_trigger={'status': 'activated', 'activated_at_s': 10.0}))
    assert result['judged'] is True
    assert result['grade'] == 'A'
    assert result['fulfilled_at_s'] == 10.5


def test_intermediate_early_execution_is_timing_failure_not_judge_error():
    client = Client([check(fulfilled=9.3)])
    judge = PassengerJudgeRuntime('ego', client)
    result = judge.judge(dict(schema_version='passenger-judge-evidence-v3',
        request=dict(request_id='r', created_at_s=0,
                     acceptance_criteria=contract()),
        response_origin_s=10.0, judged_at_s=10.1, final_window=False,
        judge_trigger={'status': 'activated', 'activated_at_s': 10.0}))
    assert result['judged'] is True
    assert result['completion_status'] == 'pending'
    assert result['trigger_timing_status'] == 'unmet'
    assert result['fulfilled_at_s'] == 9.3
    assert not judge.state['errors']


def test_final_early_execution_is_deterministically_capped_at_d():
    client = Client([verdict('A', fulfilled=9.3)])
    judge = PassengerJudgeRuntime('ego', client)
    result = judge.judge(dict(schema_version='passenger-judge-evidence-v3',
        request=dict(request_id='r', created_at_s=0,
                     acceptance_criteria=contract()),
        response_origin_s=10.0, judged_at_s=12.0, final_window=True,
        terminal=False,
        judge_trigger={'status': 'activated', 'activated_at_s': 10.0}))
    assert result['judged'] is True
    assert result['grade'] == 'D'
    assert result['overall_score_100'] == 40
    assert result['completion_status'] == 'pending'
    assert result['trigger_timing_status'] == 'unmet'
    assert result['criterion_statuses']['core'] == ['unmet']
    assert result['completion_reason'].startswith(
        'Temporal constraint unmet:')
    assert result['judge_reason_raw'] == \
        'Window state observed in physical evidence'
    assert result['score_constraints'] == [
        'grade_normalized_from_trigger_timing']
    assert not judge.state['errors']


def test_unactivated_trigger_cannot_leave_full_credit_unscored():
    client = Client([verdict('A', fulfilled=2.0)])
    judge = PassengerJudgeRuntime('ego', client)
    result = judge.judge(dict(
        schema_version='passenger-judge-evidence-v3',
        request=dict(request_id='r', created_at_s=0,
                     acceptance_criteria=contract()),
        response_origin_s=0.0, judged_at_s=5.0, final_window=True,
        terminal=False,
        judge_trigger={'status': 'timed_out', 'activated_at_s': None}))
    assert result['judged'] is True
    assert result['grade'] == 'D'
    assert result['overall_score_100'] == 40
    assert result['completion_status'] == 'pending'
    assert result['trigger_timing_status'] == 'unmet'
    assert result['criterion_statuses']['core'] == ['unmet']
    assert result['completion_reason'] == (
        'Temporal condition unmet: the frozen judge trigger never activated '
        'before the final observation window closed.')
    assert result['judge_reason_raw'] == \
        'Window state observed in physical evidence'
    assert result['score_constraints'] == [
        'grade_normalized_from_unactivated_trigger']
    assert not judge.state['errors']


def test_mixed_phase_request_does_not_apply_trigger_clock_to_immediate_work():
    criteria = dict(
        core=['Air conditioning is on'],
        secondary=['Sunshade opens after the trigger'],
        request_kind='one_shot', expected_response_s=2.0,
        valid_for_s=5.0)
    request = dict(
        request_id='r', created_at_s=0,
        acceptance_criteria=criteria,
        judge_trigger={'trigger_type': 'time', 'condition': 'after_delay',
                       'after_s': 10, 'timeout_s': 12},
        criterion_phases={
            'immediate': {'core_indices': [0], 'secondary_indices': []},
            'triggered': {'core_indices': [], 'secondary_indices': [0]},
        })
    client = Client([response('submit_passenger_judgement', {
        'grade': 'C', 'reason': 'Immediate core met; triggered secondary unmet',
        'core_statuses': ['met'], 'secondary_statuses': ['unmet'],
        'immediate_fulfilled_at_s': 0.1,
        'request_kind': 'one_shot', 'completion_status': 'pending',
        'completion_reason': 'The delayed sunshade outcome was not performed.',
    })])
    result = PassengerJudgeRuntime('ego', client).judge(dict(
        schema_version='passenger-judge-evidence-v3', request=request,
        response_origin_s=10.0, judged_at_s=12.0, final_window=True,
        terminal=False,
        judge_trigger={'status': 'activated', 'activated_at_s': 10.0}))

    assert result['judged'] is True
    assert result['grade'] == 'C'
    assert result['score_constraints'] == []
    assert 'trigger_timing_status' not in result


def test_mixed_phase_request_scores_each_fulfillment_from_own_origin():
    criteria = dict(
        core=['Air conditioning is on', 'Sunshade is open'], secondary=[],
        request_kind='one_shot', expected_response_s=1.0,
        valid_for_s=3.0)
    request = dict(
        request_id='r', created_at_s=0,
        acceptance_criteria=criteria,
        judge_trigger={'trigger_type': 'time', 'condition': 'after_delay',
                       'after_s': 10, 'timeout_s': 12},
        criterion_phases={
            'immediate': {'core_indices': [0], 'secondary_indices': []},
            'triggered': {'core_indices': [1], 'secondary_indices': []},
        })
    client = Client([response('submit_passenger_judgement', {
        'grade': 'A', 'reason': 'Both phase outcomes are directly observed',
        'core_statuses': ['met', 'met'], 'secondary_statuses': [],
        'immediate_fulfilled_at_s': 0.2,
        'triggered_fulfilled_at_s': 10.4,
        'request_kind': 'one_shot', 'completion_status': 'completed',
        'completion_reason': 'Both phase outcomes are complete.',
    })])
    result = PassengerJudgeRuntime('ego', client).judge(dict(
        schema_version='passenger-judge-evidence-v3', request=request,
        response_origin_s=10.0, judged_at_s=10.5, final_window=True,
        terminal=False,
        judge_trigger={'status': 'activated', 'activated_at_s': 10.0}))

    assert result['judged'] is True
    assert result['grade'] == 'A'
    assert result['fulfilled_at_s'] == 10.4
    assert result['immediate_fulfilled_at_s'] == 0.2
    assert result['triggered_fulfilled_at_s'] == 10.4


def test_final_grade_bucket_is_derived_from_each_phase_clock():
    criteria = dict(
        core=['Air conditioning is on', 'Seat heating is on'], secondary=[],
        request_kind='one_shot', expected_response_s=2.0,
        valid_for_s=5.0)
    request = dict(
        request_id='r', created_at_s=0, acceptance_criteria=criteria,
        criterion_phases={
            'immediate': {'core_indices': [0], 'secondary_indices': []},
            'triggered': {'core_indices': [1], 'secondary_indices': []},
        })
    client = Client([response('submit_passenger_judgement', {
        # The model makes the old bug: it compares 11s against creation time
        # and submits B, even though the triggered phase took only one second.
        'grade': 'B', 'reason': 'Both outcomes are directly observed',
        'core_statuses': ['met', 'met'], 'secondary_statuses': [],
        'immediate_fulfilled_at_s': 0.0,
        'triggered_fulfilled_at_s': 11.0,
        'request_kind': 'one_shot', 'completion_status': 'completed',
        'completion_reason': 'Both outcomes are complete.',
    })])
    result = PassengerJudgeRuntime('ego', client).judge(dict(
        schema_version='passenger-judge-evidence-v3', request=request,
        response_origin_s=10.0, judged_at_s=11.1, final_window=True,
        terminal=True,
        judge_trigger={'status': 'activated', 'activated_at_s': 10.0}))
    assert result['judged'] is True
    assert result['grade'] == 'A'
    assert result['score_constraints'] == [
        'grade_normalized_from_phase_timing']


def test_judge_adds_generic_numeric_trace_extrema_without_changing_evidence():
    evidence = dict(request=dict(request_id='r', created_at_s=0), judged_at_s=2,
        observation_when_requested={'cabin': {'shared_settings': {'arbitrary': 3}}},
        physical_execution={'ego_motion_trace': [
            {'time_s': 0, 'arbitrary_measure': 2.0},
            {'time_s': 1, 'arbitrary_measure': -7.0},
            {'time_s': 2, 'arbitrary_measure': 4.0},
        ]})
    original = copy.deepcopy(evidence)
    client = Client([response('submit_passenger_judgement', {
        'grade': 'D', 'reason': 'Observed counterexample'})])
    PassengerJudgeRuntime('ego', client).judge(evidence)
    sent = json.loads(client.calls[0][1]['content'])
    assert sent['observation_when_requested']['cabin']['current_state'] == {'arbitrary': 3}
    summary = sent['physical_execution']['numeric_trace_summary']['arbitrary_measure']
    assert summary['min'] == -7
    assert summary['max'] == 4
    assert summary['min_occurrences'] == [{'index': 1, 'time_s': 1}]
    assert evidence == original


def test_grade_is_normalized_from_generic_criterion_statuses():
    criteria = contract()
    criteria['secondary'] = ['Explicit secondary outcome']
    client = Client([response('submit_passenger_judgement', {
        'grade': 'A', 'reason': 'Core met; secondary unverified',
        'core_statuses': ['met'], 'secondary_statuses': ['unverified'],
        'fulfilled_at_s': 0.5, 'request_kind': 'one_shot',
        'completion_status': 'completed', 'completion_reason': 'Core observed'})])
    result = PassengerJudgeRuntime('ego', client).judge(dict(
        schema_version='passenger-judge-evidence-v3',
        request=dict(request_id='r', created_at_s=0, acceptance_criteria=criteria),
        judged_at_s=1, final_window=True))
    assert result['judged'] is True
    assert result['grade'] == 'C'
    assert result['overall_score_100'] == 60
    assert result['criterion_statuses']['secondary'] == ['unverified']
    assert result['score_constraints'] == ['grade_normalized_from_criterion_statuses']


def test_evidence_backed_unsupported_refusal_counts_as_resolved():
    criteria = dict(
        core=['Set temperature to 24', 'Activate washer spray'],
        secondary=[], request_kind='one_shot',
        expected_response_s=1.0, valid_for_s=5.0)
    client = Client([response('submit_passenger_judgement', {
        'grade': 'D',
        'reason': (
            'Temperature met; washer is unavailable and was explicitly '
            'refused after capability inspection.'),
        'core_statuses': ['met', 'unsupported_refused'],
        'secondary_statuses': [],
        'immediate_fulfilled_at_s': 0.5,
        'request_kind': 'one_shot', 'completion_status': 'completed',
        'completion_reason': 'All requested outcomes were handled.',
    })])
    result = PassengerJudgeRuntime('ego', client).judge(dict(
        schema_version='passenger-judge-evidence-v3',
        request=dict(
            request_id='r', created_at_s=0,
            acceptance_criteria=criteria,
            criterion_phases={
                'immediate': {
                    'core_indices': [0, 1], 'secondary_indices': []},
                'triggered': {
                    'core_indices': [], 'secondary_indices': []},
            }),
        judged_at_s=1, final_window=True))

    assert result['judged'] is True
    assert result['grade'] == 'A'
    assert result['overall_score_100'] == 100
    assert result['criterion_statuses']['core'] == [
        'met', 'unsupported_refused']
    assert result['fulfilled_at_s'] == 0.5
    assert result['score_constraints'] == [
        'grade_normalized_from_phase_timing']
    schema = client.tool_schemas[0][0]['function']['parameters']
    assert 'unsupported_refused' in schema['properties'][
        'core_statuses']['items']['enum']
    assert 'authoritative capability evidence' in client.calls[0][0]['content']
    assert 'Internal Todo closure is bookkeeping' in \
        client.calls[0][0]['content']


def test_masking_an_effect_is_not_documented_as_physical_fulfillment():
    client = Client([verdict('D')])
    PassengerJudgeRuntime('ego', client).judge(dict(
        schema_version='passenger-judge-evidence-v3',
        request=dict(request_id='r', created_at_s=0,
                     acceptance_criteria=contract()),
        judged_at_s=1, final_window=True))
    assert 'only masks an effect' in client.calls[0][0]['content']


def test_core_unmet_caps_an_inconsistent_high_grade_at_d():
    criteria = contract()
    client = Client([response('submit_passenger_judgement', {
        'grade': 'C', 'reason': 'Useful partial result, but core counterexample observed',
        'core_statuses': ['met', 'unmet'], 'secondary_statuses': [],
        'request_kind': 'one_shot', 'completion_status': 'pending',
        'completion_reason': 'Core remains unmet'})])
    result = PassengerJudgeRuntime('ego', client).judge(dict(
        schema_version='passenger-judge-evidence-v3',
        request=dict(request_id='r', created_at_s=0, acceptance_criteria=criteria),
        judged_at_s=1, final_window=True))
    assert result['judged'] is True
    assert result['grade'] == 'D'
    assert result['overall_score_100'] == 40
    assert result['criterion_statuses']['core'] == ['met', 'unmet']


def test_terminal_window_can_finalize_ongoing_request_before_original_expiry():
    client = Client([verdict('A', kind='ongoing', fulfilled=0.5)])
    result = PassengerJudgeRuntime('ego', client).judge(dict(
        schema_version='passenger-judge-evidence-v3',
        request=dict(request_id='r', created_at_s=0,
                     acceptance_criteria=contract('ongoing')),
        judged_at_s=2, final_window=True, terminal=True))
    assert result['judged'] is True
    assert result['grade'] == 'A'
    assert result['completion_status'] == 'completed'


def test_ongoing_request_checks_wait_until_frozen_deadline():
    kind, grade, fulfilled = 'ongoing', 'A', 0.5
    cb,pa,judge,drivers = make_callback([
        response('send_passenger_request',dict(message='Close the window',acceptance_criteria=contract(kind)))],
        [check(fulfilled=fulfilled), check(fulfilled=fulfilled),
         verdict(grade,kind,fulfilled)])
    vw = VehicleWorld()
    wake(cb,vw,0,'personal_agent_due')
    assert cb.request_records[0]['acceptance_deadline_s'] == 5
    for t,next_t in [(2,4),(4,5)]:
        wake(cb,vw,t,'passenger_judge_due')
        assert cb.pending_judge_due_at_s == next_t
        assert not cb.judge.state['judgements']
    wake(cb,vw,5,'passenger_judge_due')
    assert cb.pending_judge_due_at_s is None
    assert len(cb.judge.state['judgements']) == 1
    assert cb.judge.state['judgements'][0]['overall_score_100'] == GRADE_SCORES[grade]
    assert all(call[0]['function']['name'] == 'submit_passenger_check'
               for call in judge.tool_schemas[:2])
    assert judge.tool_schemas[2][0]['function']['name'] == 'submit_passenger_judgement'
    final_evidence = json.loads(judge.calls[2][1]['content'])
    assert len(final_evidence['prior_checks']) == 2
    assert all(item['check_only'] is True
               for item in final_evidence['prior_checks'])


def test_full_timely_request_closes_early_without_driver_reply():
    cb,pa,judge,drivers = make_callback([
        response('send_passenger_request',dict(message='Close window',acceptance_criteria=contract()))],
        [check(fulfilled=0.5)])
    vw=VehicleWorld(); wake(cb,vw,0,'personal_agent_due'); wake(cb,vw,2,'passenger_judge_due')
    assert cb.pending_judge_due_at_s is None
    assert cb.judge.state['judgements'][0]['overall_score_100'] == 100


def test_summary_counts_each_request_once_and_excludes_na_and_failures():
    def item(r,g): return dict(request_id=r,judged=True,**score_submission(dict(
        grade=g, reason='evidence',na_reason='insufficient_evidence')))
    summary=aggregate_passenger_judgements([item('r1','D'),item('r1','B'),item('r2','F'),
        item('r3','NA'),dict(request_id='r4',judged=False,error='API failure')],request_count=5)
    assert summary['overall_score_100'] == 40
    assert summary['scored_count'] == 2
    assert summary['na_count'] == summary['judge_failed_count'] == summary['unjudged_count'] == 1
    assert summary['grade_counts']['B'] == summary['grade_counts']['F'] == 1
    assert summary['metric_revision'] == REVISION
    assert summary['score_coverage_rate'] == 0.4
    assert summary['coverage_adjusted_score_100'] == 16.0
    assert summary['unresolved_request_count'] == 3


@pytest.mark.parametrize('key,value', [('valid_for_s',0),('valid_for_s',0.5),
                                    ('expected_response_s',True),('core',[]),('request_kind','unknown')])
def test_bad_contract_rejected(key,value):
    c=contract(); c[key]=value
    with pytest.raises(ValueError): validate_contract(c)


def test_scene_average_weights_requests_not_vehicles():
    from simulation.multi_sim_engine import MultiSimResult, VehicleResult
    def evaluation(vid,grades):
        return aggregate_passenger_judgements([dict(request_id=f'{vid}-{i}',judged=True,
            **score_submission(dict(grade=g,reason='evidence'))) for i,g in enumerate(grades)])
    first = VehicleResult(vehicle_id='a',is_evaluated=True)
    second = VehicleResult(vehicle_id='b',is_evaluated=True)
    first.passenger_evaluation = evaluation('a',['A'])
    second.passenger_evaluation = evaluation('b',['F','F','F'])
    result = MultiSimResult(scenario_id='test',vehicle_results={'a':first,'b':second})
    summary = result.to_dict()['evaluation']['passenger_interaction']
    assert summary['overall_score_100'] == 25
    assert summary['scored_count'] == 4
    assert summary['grade_counts']['F'] == 3


def test_task_passenger_summary_uses_focal_and_preserves_all_agents():
    from evaluation.experiments.batch_runner import _task_passenger_summary
    result = {'evaluation': {'passenger_interaction': {
        'overall_score_100': 80,
        'vehicles': {
            'ego': {'overall_score_100': 40, 'scored_count': 1},
            'peer': {'overall_score_100': 100, 'scored_count': 3},
        },
    }}}
    focal, all_agents = _task_passenger_summary(
        result, focal_id='ego', focal_only=True)
    assert focal['overall_score_100'] == 40
    assert all_agents['overall_score_100'] == 80
    missing, _ = _task_passenger_summary(
        result, focal_id='missing', focal_only=True)
    assert missing == {}
