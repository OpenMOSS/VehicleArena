"""Frozen request contracts and discrete, request-only passenger scoring."""
import math

REVISION = 'request_satisfaction_six_grade_v2'
GRADE_SCORES = {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'E': 20, 'F': 0}
CRITERION_STATUSES = (
    'met', 'unsupported_refused', 'unmet', 'unverified',
)
RESOLVED_CRITERION_STATUSES = frozenset({'met', 'unsupported_refused'})
RUBRIC = (
    'Evaluate ONLY fulfillment of this passenger request, not general driving '
    'rationality, safety or external traffic impact. Use the frozen acceptance '
    'criteria; never invent new requirements. Choose A=100: all requirements '
    'resolved correctly within expected_response_s; B=80: all resolved but '
    'late, before valid_for_s; C=60: all core requirements resolved with minor '
    'secondary omissions; D=40: useful partial core fulfillment or inadequate '
    'maintenance; E=20: evidenced relevant physical action but no useful outcome; '
    'F=0: no relevant action, unrelated action or opposite outcome. A verbal '
    'promise, Todo edit or merely queued command is not an effective attempt. '
    'Do not force intermediate grades for atomic requests. Grade NA only for '
    'a malformed or unsafe request, or genuinely '
    'insufficient evidence, never for observed non-fulfillment. Cite concrete '
    'evidence in reason. No spoken reply is needed for physical requests; an '
    'information request needs an actual answer. Already satisfied state counts. '
    'A capability-limited requested outcome remains evaluable: classify it as '
    'unsupported_refused only when authoritative capability evidence proves '
    'that the current vehicle lacks the requested operation and the Driving '
    'Agent clearly tells the passenger that it cannot perform that outcome. '
    'The Agent must not claim completion or substitute a different operation. '
    'A statement that the requested operation is unavailable or cannot be '
    'performed is an explicit refusal even if it does not use the word '
    '"refuse". Closing or cancelling an internal Todo is bookkeeping and '
    'does not negate an otherwise truthful refusal unless the Agent also '
    'claims that the unavailable physical outcome occurred. '
    'A refusal without capability evidence is unverified; silence, a false '
    'completion claim, or an attempted substitute is unmet. A properly refused '
    'unsupported outcome is resolved for grading, but it is not evidence that '
    'the requested physical state occurred. Do not use NA solely because one '
    'requested operation is unavailable on the current vehicle. '
    'For timing, use evidence of first fulfillment, not just the current check '
    'time. For A/B supply fulfilled_at_s from that evidence. Do not claim A if timely fulfillment cannot be established. '
    'Sustained requests require evidence throughout the supplied observation '
    'period. A normal intermediate check must not replace the frozen period; '
    'when the vehicle or episode terminates, the complete actually observable '
    'window is the final period. '
    '\nBefore selecting a grade, audit EVERY core and secondary criterion '
    'against the supplied evidence. In reason, concisely identify which '
    'criteria are met, unsupported_refused, unmet, or unverified and cite '
    'the relevant state/time. '
    'Do not silently ignore secondary criteria because they are not core: '
    'A/B require every applicable criterion to be resolved. C requires ALL '
    'core outcomes to be resolved; any contradicted core outcome rules out A/B/C. '
    'A/B also require supported timing, not a guessed fulfillment timestamp. '
    'Match each criterion to positive evidence that directly supports it. '
    'A successful command proves only what its recorded result and subsequent '
    'observable state establish; do not infer additional unobserved attributes. '
    'An operation that only masks an effect is not evidence that the requested '
    'underlying state changed. Likewise, a nearby or substitute operation does '
    'not satisfy a different requested outcome. '
    'Treat simulator observations as authoritative for the state they name. '
    'Do not reinterpret an observed value using assumptions about real-world '
    'implementations or demand an extra sensor absent from this environment. '
    'A direct before/after change in the requested state is positive evidence. '
    'Do not downgrade an observed state to merely a setting, intention, or '
    'proxy unless the supplied evidence explicitly labels it that way. '
    'Unknown state is not proof of failure, but cannot support full credit. '
    'When useful core fulfillment is established and only secondary evidence '
    'is missing, choose C and explicitly say it is unverified, not observed '
    'noncompliance. Reserve NA for evidence too weak to judge the request. '
    'For sustained requests, inspect the ENTIRE supplied interval, especially '
    'contradictory motion after a seemingly correct initial command. A promise '
    'or soft acceleration command cannot outweigh later observed harsh motion. '
    'For any requirement that must hold over an interval, one clear observed '
    'counterexample makes that requirement unmet unless the request explicitly '
    'allows exceptions. Verify the full chronology before claiming that no '
    'counterexample occurred; do not cherry-pick favorable samples. '
    'When the passenger identifies an observed earlier event as the unwanted '
    'behavior to avoid, use that event as the evidence-grounded reference. A '
    'materially similar event after the request is a counterexample; do not '
    'relabel it acceptable using an invented threshold. '
    'If meaningful compliant behavior alternates with contrary behavior, D '
    'may apply; E means only an actual attempt without useful fulfillment. '
    'Read the original passenger message alongside the frozen checklist. '
    'If the checklist adds an unrequested condition or reduces a requested '
    'physical outcome to merely promising it, flag this contract mismatch '
    'in reason. Do not silently redefine or repair frozen criteria; use '
    'NA/invalid_request when that mismatch makes a fair grade impossible. '
    'Finally check that your grade and completion_status do not contradict '
    'your own evidence. Previous check verdicts are not evidence and must '
    'not override observed contradictions. '
)
CONTRACT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'core': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': {'type': 'string', 'minLength': 1}},
        'secondary': {'type': 'array', 'maxItems': 6, 'items': {'type': 'string', 'minLength': 1}},
        'request_kind': {'type': 'string', 'enum': ['one_shot', 'ongoing']},
        'expected_response_s': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 30},
        'valid_for_s': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 60},
    },
    'required': ['core', 'secondary', 'request_kind', 'expected_response_s', 'valid_for_s'],
}

TRIGGER_TYPES = {
    'after_delay': 'time',
    'exit_next_intersection': 'map',
    'approach_next_intersection': 'map',
    'enter_next_intersection': 'map',
    'distance_to_destination_below': 'map',
    'vehicle_stopped': 'vehicle_state',
    'vehicle_resumed_moving': 'vehicle_state',
    'speed_threshold_held': 'vehicle_state',
    'weather_changed_to': 'environment',
    'daynight_became_dark': 'environment',
}
TRIGGER_WEATHER_CONDITIONS = ('rainy', 'heavy_rain', 'foggy')
TRIGGER_DESCRIPTIONS = {
    'after_delay': 'After after_s seconds of simulation time.',
    'exit_next_intersection': 'On exiting the current or next intersection.',
    'approach_next_intersection': 'On first reaching the chosen distance before the next intersection.',
    'enter_next_intersection': 'On entering the next intersection.',
    'distance_to_destination_below': 'When navigation distance first falls below distance_m.',
    'vehicle_stopped': 'After a continuous stop lasting hold_for_s seconds.',
    'vehicle_resumed_moving': 'When moving again after a qualifying continuous stop.',
    'speed_threshold_held': 'After speed stays strictly above or below the threshold.',
    'weather_changed_to': 'On a future weather-change event matching the chosen condition.',
    'daynight_became_dark': 'On a future light-to-dark transition.',
}

JUDGE_TRIGGER_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'trigger_type': {
            'type': 'string',
            'enum': sorted(set(TRIGGER_TYPES.values())),
        },
        'condition': {
            'type': 'string',
            'enum': list(TRIGGER_TYPES),
        },
        'after_s': {
            'type': 'number', 'exclusiveMinimum': 0, 'maximum': 60,
        },
        'hold_for_s': {
            'type': 'number', 'exclusiveMinimum': 0, 'maximum': 10,
        },
        'distance_m': {
            'type': 'number', 'exclusiveMinimum': 0, 'maximum': 500,
        },
        'speed_kmh': {
            'type': 'number', 'exclusiveMinimum': 0,
            'exclusiveMaximum': 120,
        },
        'comparison': {'type': 'string', 'enum': ['above', 'below']},
        'weather_condition': {
            'type': 'string', 'enum': list(TRIGGER_WEATHER_CONDITIONS),
        },
        'timeout_s': {
            'type': 'number', 'exclusiveMinimum': 0, 'maximum': 60,
        },
    },
    'required': ['trigger_type', 'condition', 'timeout_s'],
}


def validate_contract(contract):
    if not isinstance(contract, dict) or set(contract) != set(CONTRACT_SCHEMA['required']):
        raise ValueError('invalid acceptance criteria fields')
    for key in ('core', 'secondary'):
        values = contract[key]
        if (not isinstance(values, list) or len(values) > 6
                or (key == 'core' and not values)
                or any(not isinstance(v, str) or not v.strip() or len(v) > 300 for v in values)):
            raise ValueError('invalid acceptance requirements')
    if len(contract['core']) + len(contract['secondary']) > 6:
        raise ValueError('at most six acceptance requirements are allowed')
    for key, maximum in [('expected_response_s', 30), ('valid_for_s', 60)]:
        v = contract[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= maximum:
            raise ValueError('invalid request timing')
    if contract['expected_response_s'] > contract['valid_for_s'] or contract['request_kind'] not in ('one_shot', 'ongoing'):
        raise ValueError('invalid request timing or kind')


def validate_judge_trigger(trigger):
    """Validate one bounded, declarative evaluator trigger.

    The Personal Agent may choose only these predicates.  It never supplies
    executable code, map identifiers, or evaluator implementation details.
    """
    if not isinstance(trigger, dict):
        raise ValueError('judge trigger must be an object')
    allowed = set(JUDGE_TRIGGER_SCHEMA['properties'])
    if (not {'trigger_type', 'condition', 'timeout_s'}.issubset(trigger)
            or not set(trigger).issubset(allowed)):
        raise ValueError('invalid judge trigger fields')
    trigger_type = trigger.get('trigger_type')
    condition = trigger.get('condition')
    if (not isinstance(condition, str)
            or not isinstance(trigger_type, str)
            or TRIGGER_TYPES.get(condition) != trigger_type):
        raise ValueError('invalid judge trigger type/condition')
    timeout = trigger.get('timeout_s')
    if (type(timeout) not in (int, float) or not math.isfinite(timeout)
            or not 0 < timeout <= 60):
        raise ValueError('invalid judge trigger timeout')
    required = {
        'after_delay': {'after_s'},
        'vehicle_stopped': set(),
        'vehicle_resumed_moving': set(),
        'speed_threshold_held': {'speed_kmh', 'comparison', 'hold_for_s'},
        'approach_next_intersection': {'distance_m'},
        'distance_to_destination_below': {'distance_m'},
        'weather_changed_to': {'weather_condition'},
    }.get(condition, set())
    optional = {'hold_for_s'} if condition in {
        'vehicle_stopped', 'vehicle_resumed_moving'} else set()
    parameters = set(trigger) - {'trigger_type', 'condition', 'timeout_s'}
    if not required.issubset(parameters) or not parameters.issubset(required | optional):
        raise ValueError('invalid judge trigger parameters for condition')
    after = trigger.get('after_s')
    hold = trigger.get('hold_for_s')
    if condition == 'after_delay':
        if (type(after) not in (int, float) or not math.isfinite(after)
                or not 0 < after <= 60 or after > timeout):
            raise ValueError('time trigger requires after_s within timeout')
    if condition in ('vehicle_stopped', 'vehicle_resumed_moving'):
        if hold is None:
            hold = 0.5
        if (type(hold) not in (int, float) or not math.isfinite(hold)
                or not 0 < hold <= min(10, timeout)):
            raise ValueError('vehicle-state trigger has invalid hold_for_s')
    if condition == 'speed_threshold_held':
        speed = trigger['speed_kmh']
        if (type(speed) not in (int, float) or not math.isfinite(speed)
                or not 0 < speed < 120 or trigger['comparison'] not in ('above', 'below')):
            raise ValueError('invalid speed threshold')
        if (type(hold) not in (int, float) or not math.isfinite(hold)
                or not 0 < hold <= min(10, timeout)):
            raise ValueError('invalid speed threshold hold_for_s')
    if condition in ('approach_next_intersection', 'distance_to_destination_below'):
        distance = trigger['distance_m']
        if (type(distance) not in (int, float) or not math.isfinite(distance)
                or not 0 < distance <= 500):
            raise ValueError('invalid distance threshold')
    if (condition == 'weather_changed_to'
            and trigger['weather_condition'] not in TRIGGER_WEATHER_CONDITIONS):
        raise ValueError('invalid target weather condition')


def available_judge_triggers(context, design=None):
    """One source of truth for PA-visible and accepted trigger choices.

    Production contexts are route-aware: a delayed predicate is offered only
    when it can activate before the route endpoint and still leave the full
    Judge observation window. State predicates require an already-observed
    guarantee; they are never offered just because their schema is valid.
    """
    context = context or {}
    design = design or {}
    if design.get('judge_trigger') == 'forbidden':
        return []
    allowed = design.get('allowed_trigger_conditions')
    remaining = context.get('remaining_episode_s')
    judge_window = context.get('judge_observation_window_s', 3.0)
    judge_window = (float(judge_window)
                    if type(judge_window) in (int, float)
                    and math.isfinite(judge_window) else 3.0)
    judge_window = max(0.1, judge_window)
    horizons = []
    if type(remaining) in (int, float) and math.isfinite(remaining):
        horizons.append(float(remaining))
    free_flow = context.get('estimated_free_flow_remaining_s')
    if type(free_flow) in (int, float) and math.isfinite(free_flow):
        horizons.append(float(free_flow))
    endpoint_horizon = min(horizons) if horizons else None
    timeout_max = min(
        60.0,
        (endpoint_horizon - judge_window - 0.1)
        if endpoint_horizon is not None else 60.0)
    if timeout_max <= 0.5:
        return []
    candidates = []

    def add(condition, **parameters):
        if isinstance(allowed, list) and condition not in allowed:
            return
        timeout_bounds = parameters.pop(
            'timeout_s', {'min_exclusive': 0, 'max': timeout_max})
        candidates.append({
            'trigger_type': TRIGGER_TYPES[condition],
            'condition': condition,
            'meaning': TRIGGER_DESCRIPTIONS[condition],
            'timeout_s': timeout_bounds,
            **parameters,
        })
    delay_max = context.get('max_after_delay_s', timeout_max)
    if (type(delay_max) in (int, float) and math.isfinite(delay_max)
            and delay_max > 0):
        add('after_delay', timeout_s={'min_exclusive': 0, 'max': timeout_max},
            after_s={
            'min_exclusive': 0, 'max': min(timeout_max, float(delay_max))})

    remaining_distance = context.get('remaining_distance_m')
    route_distance = (float(remaining_distance)
                      if type(remaining_distance) in (int, float)
                      and math.isfinite(remaining_distance)
                      and remaining_distance >= 0.0 else None)
    cruise_mps = context.get('estimated_free_flow_speed_mps')
    cruise_mps = (float(cruise_mps)
                  if type(cruise_mps) in (int, float)
                  and math.isfinite(cruise_mps) and cruise_mps > 0.0 else None)

    def leaves_judge_window_after(event_distance_m):
        if route_distance is None or cruise_mps is None:
            # Compatibility for direct unit/runtime callers that do not have
            # the production route instrumentation. Engine contexts always
            # include both values and therefore take the strict branch.
            return True
        return (route_distance - float(event_distance_m)
                >= cruise_mps * judge_window + 0.5)

    intersection = context.get('intersection_ahead_known', True)
    inside = context.get('inside_intersection', False)
    if intersection:
        entry_distance = context.get('next_intersection_distance_m')
        entry_distance = (float(entry_distance)
                          if type(entry_distance) in (int, float)
                          and math.isfinite(entry_distance)
                          and entry_distance >= 0.0 else None)
        exit_distance = context.get('next_intersection_exit_distance_m')
        exit_distance = (float(exit_distance)
                         if type(exit_distance) in (int, float)
                         and math.isfinite(exit_distance)
                         and exit_distance >= 0.0 else None)
        if exit_distance is not None and leaves_judge_window_after(exit_distance):
            add('exit_next_intersection')
        if not inside and entry_distance is not None \
                and leaves_judge_window_after(entry_distance):
            add('enter_next_intersection')
            if entry_distance > 5.0:
                add('approach_next_intersection', distance_m={
                    'min_exclusive': 0,
                    **({'max': 500.0} if entry_distance > 500.0 else
                       {'max_exclusive': entry_distance})})
    minimum_destination_distance = (
        cruise_mps * judge_window + 0.5 if cruise_mps is not None else 0.0)
    if route_distance is not None and route_distance > max(
            5.0, minimum_destination_distance + 1e-9):
        add('distance_to_destination_below', distance_m={
            'min_exclusive': minimum_destination_distance,
            **({'max': 500.0} if route_distance > 500.0 else
               {'max_exclusive': route_distance})})

    guarantees = context.get('guaranteed_trigger_conditions')
    if isinstance(guarantees, dict):
        stopped = guarantees.get('vehicle_stopped')
        maximum_hold = (stopped.get('max_hold_for_s')
                        if isinstance(stopped, dict) else None)
        if (type(maximum_hold) in (int, float)
                and math.isfinite(maximum_hold) and maximum_hold >= 0.5):
            add('vehicle_stopped', hold_for_s={
                'min_exclusive': 0,
                'max': min(10.0, timeout_max, float(maximum_hold)),
            })
    else:
        # Non-engine callers retain the historical generic tool surface.
        hold = {'min_exclusive': 0, 'max': min(10.0, timeout_max)}
        add('vehicle_stopped', hold_for_s=hold)
        add('vehicle_resumed_moving', hold_for_s=hold)
        add('speed_threshold_held', comparison=['above', 'below'],
            speed_kmh={'min_exclusive': 0, 'max_exclusive': 120},
            hold_for_s=hold)

    future_weather_events = context.get('future_weather_events')
    if isinstance(future_weather_events, list):
        future_weather = sorted({
            item.get('weather_condition') for item in future_weather_events
            if isinstance(item, dict)
            and item.get('weather_condition') in TRIGGER_WEATHER_CONDITIONS
            and type(item.get('after_s')) in (int, float)
            and math.isfinite(item['after_s'])
            and 0.0 < float(item['after_s']) <= timeout_max
        })
    else:
        future_weather = sorted(set(context.get('future_weather_conditions', []))
                                & set(TRIGGER_WEATHER_CONDITIONS))
    if future_weather:
        add('weather_changed_to', weather_condition=future_weather)
    dark_after = context.get('future_dark_after_s')
    if ((type(dark_after) in (int, float) and math.isfinite(dark_after)
         and 0.0 < float(dark_after) <= timeout_max)
            or (dark_after is None and context.get('future_dark_transition'))):
        add('daynight_became_dark')
    return candidates


def validate_available_judge_trigger(trigger, options):
    """Reject an option/threshold the PA was not offered at this wake."""
    matches = [option for option in options
               if option['condition'] == trigger['condition']]
    if not matches:
        raise ValueError('judge trigger was not available at this wake')
    option = matches[0]
    for key, limits in option.items():
        if key in ('trigger_type', 'condition', 'meaning') or key not in trigger:
            continue
        value = trigger[key]
        if isinstance(limits, list):
            if value not in limits:
                raise ValueError(f'{key} was not available at this wake')
        elif isinstance(limits, dict):
            if ('min_exclusive' in limits and value <= limits['min_exclusive']
                    or 'max_exclusive' in limits and value >= limits['max_exclusive']
                    or 'min' in limits and value < limits['min']
                    or 'max' in limits and value > limits['max'] + 1e-9):
                raise ValueError(f'{key} exceeds available trigger range')


def score_submission(arguments):
    grade = arguments.get('grade')
    reason = arguments.get('reason')
    if grade not in (*GRADE_SCORES, 'NA') or not isinstance(reason, str) or not reason.strip():
        raise ValueError('invalid request satisfaction grade or evidence')
    if grade == 'NA' and arguments.get('na_reason') not in ('invalid_request', 'insufficient_evidence'):
        raise ValueError('NA requires a reason category')
    score = GRADE_SCORES.get(grade)
    return {'metric_revision': REVISION, 'grade': grade, 'applicable': grade != 'NA',
            'na_reason': arguments.get('na_reason') if grade == 'NA' else None,
            'dimension_scores_100': {'request_response': score} if score is not None else {},
            'overall_score_100': score, 'reason': reason[:600],
            'judge_reason_raw': reason[:600], 'score_constraints': []}
