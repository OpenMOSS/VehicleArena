"""Focused contract tests for the deduction-based driving process score."""

import pytest

from evaluation.driving_process_score import (
    _equipment_snapshot,
    calculate_driving_process_score,
)
from evaluation.trajectory_filters import (
    windowed_longitudinal_acceleration,
)
from evaluation.layer_scoring import single_vehicle_layer_score_100
from evaluation.multi_agent_runner import _events_visible_to_multimodal_driver
from evaluation.multi_agent_runner import (
    make_noop_agent,
)
from module.weather import Weather
from simulation.multi_sim_engine import _LLM_SENSOR_OR_LIFECYCLE_WAKE_EVENTS
from simulation.scenario_generator import (
    WEATHER_ACTIVE,
    WEATHER_CALM,
    WEATHER_CONDITIONS,
)
from weather_safety import (
    SUPPORTED_WEATHER_CONDITIONS,
    apply_weather_safety_transition,
    weather_score_requirements,
)
from vehicleworld import VehicleWorld


def _report(*, arrived=True, decisions=(), events=(), metrics=None):
    base_metrics = {
        "arrived": arrived,
        "collision_count": 0,
        "red_light_violations": 0,
    }
    base_metrics.update(metrics or {})
    return {
        "applicable": True,
        "task_completed": arrived,
        "metrics": base_metrics,
        "hard_violations": [],
        "decision_episodes": list(decisions),
        "events": list(events),
    }


def _sample(time_s, *, overspeed=False, speed=20.0, limit=30.0,
            equipment=None, **values):
    sample = {
        "time_s": float(time_s),
        "speed_kmh": float(speed),
        "speed_limit_kmh": float(limit),
        "overspeed": bool(overspeed),
        "unexplained_stop": False,
        "intersection_blocked": False,
        "hard_acceleration": False,
        "unnecessary_hard_brake": False,
        "high_jerk": False,
        "high_beam_misuse": False,
        "is_changing_lane": False,
        "current_lane": 0,
        "target_lane": -1,
        "left_indicator": False,
        "right_indicator": False,
        "equipment": equipment or {},
    }
    sample.update(values)
    return sample


def test_weather_and_daynight_events_reach_multimodal_driver():
    kinds = [
        "weather_initialized", "weather_changed",
        "daynight_initialized", "daynight_changed",
    ]
    visible = _events_visible_to_multimodal_driver([
        {"event_type": kind, "event_id": f"event-{index}"}
        for index, kind in enumerate(kinds)
    ])
    assert set(kinds) <= _LLM_SENSOR_OR_LIFECYCLE_WAKE_EVENTS
    assert [item["event_type"] for item in visible] == kinds


def test_weather_transition_profiles_are_selective_and_persistent():
    assert SUPPORTED_WEATHER_CONDITIONS == {
        "sunny", "cloudy", "rainy", "heavy_rain",
        "foggy", "snowy", "heavy_snow", "hail",
    }

    _, _, state, updates = apply_weather_safety_transition(
        "sunny", "sunny")
    assert updates == {}

    _, _, state, updates = apply_weather_safety_transition(
        "sunny", "foggy", state)
    assert updates == {
        "low_beam_on": True,
        "high_beam_on": False,
        "front_fog_on": True,
        "rear_fog_on": True,
        "position_light_on": True,
    }

    _, _, state, updates = apply_weather_safety_transition(
        "foggy", "rainy", state)
    assert updates == {
        "front_fog_on": False,
        "rear_fog_on": False,
        "position_light_on": False,
        "front_wiper_on": True,
        "all_windows_closed": True,
        "sunroof_closed": True,
    }
    assert state.low_beam_on is True

    _, _, state, updates = apply_weather_safety_transition(
        "rainy", "hail", state)
    assert updates["front_wiper_on"] is False
    assert state.low_beam_on is True
    assert state.all_windows_closed is True
    assert state.sunroof_closed is True

    requirements = weather_score_requirements("foggy", "sunny")
    assert requirements["fog_and_position_lights_must_be_off"] is True
    assert requirements["front_wiper_must_be_off"] is False


def test_weather_transition_profiles_cover_rain_aliases_and_snow():
    _, _, state, updates = apply_weather_safety_transition(
        "sunny", "cloudy")
    assert updates == {}

    _, _, state, updates = apply_weather_safety_transition(
        "cloudy", "heavy_rain", state)
    assert updates == {
        "front_wiper_on": True,
        "all_windows_closed": True,
        "sunroof_closed": True,
    }

    _, _, state, updates = apply_weather_safety_transition(
        "heavy_rain", "snowy", state)
    assert updates["front_wiper_on"] is True
    assert updates["low_beam_on"] is True
    assert updates["all_windows_closed"] is True
    assert updates["sunroof_closed"] is True

    _, _, state, updates = apply_weather_safety_transition(
        "snowy", "heavy_snow", state)
    assert updates["front_fog_on"] is True
    assert updates["rear_fog_on"] is True
    assert updates["position_light_on"] is True
    assert updates["high_beam_on"] is False

    _, _, state, updates = apply_weather_safety_transition(
        "heavy_snow", "cloudy", state)
    assert updates == {
        "front_fog_on": False,
        "rear_fog_on": False,
        "position_light_on": False,
        "front_wiper_on": False,
    }
    assert state.low_beam_on is True
    assert state.all_windows_closed is True
    assert state.sunroof_closed is True


def test_weather_module_and_generator_expose_all_eight_conditions():
    expected = {
        "sunny", "cloudy", "rainy", "heavy_rain",
        "foggy", "snowy", "heavy_snow", "hail",
    }
    assert {condition.value for condition in Weather.Condition} == expected
    assert set(WEATHER_CONDITIONS) == expected
    assert WEATHER_CALM == ["sunny", "cloudy"]
    assert set(WEATHER_ACTIVE) == expected - set(WEATHER_CALM)










def test_non_evaluated_sumo_background_keeps_noop_cabin_policy():
    vw = VehicleWorld()
    callback = make_noop_agent("background")
    actions = callback(
        vw, 1.0, [], None, 1,
        _wake_events=[{
            "event_id": "rain-start",
            "event_type": "weather_changed",
            "details": {
                "previous_condition": "sunny",
                "condition": "rainy",
            },
        }],
    )
    assert actions == []
    assert vw.wiper.front_wiper.is_on is False
    assert not hasattr(callback, "_rule_action_log")




def test_new_weather_event_supersedes_old_response_deadline():
    samples = [
        _sample(
            index / 10,
            equipment={
                "front_wiper_on": 5 <= index < 25,
                "front_fog_on": index >= 25,
                "rear_fog_on": index >= 25,
                "low_beam_on": index >= 25,
                "position_light_on": index >= 25,
                "all_windows_closed": True,
                "sunroof_closed": True,
            },
        )
        for index in range(61)
    ]
    result = calculate_driving_process_score(
        _report(),
        samples=samples,
        delivered_events=[
            {
                "event_id": "sunny-initial",
                "event_type": "weather_initialized",
                "delivered_at_s": 0.0,
                "details": {"condition": "sunny"},
            },
            {
                "event_id": "rain-start",
                "event_type": "weather_changed",
                "delivered_at_s": 0.5,
                "details": {"condition": "rainy"},
            },
            {
                "event_id": "fog-start",
                "event_type": "weather_changed",
                "delivered_at_s": 2.5,
                "details": {"condition": "foggy"},
            },
        ],
    )
    assert not [
        item for item in result["deductions"]
        if "weather" in item["type"] or "fog" in item["type"]
    ]


def test_soft_rule_deductions_are_not_hard_gates():
    decisions = [
        {
            "type": kind,
            "start_time_s": index,
            "end_time_s": index + 0.6,
            "reasonable": False,
            "details": {"hazard": kind},
        }
        for index, kind in enumerate((
            "leader_response", "connector_response", "pedestrian_response"))
    ]
    events = [
        {"type": "near_miss", "time_s": 4.0, "hazard": "leader:npc"},
        {"type": "near_miss", "time_s": 5.0,
         "hazard": "connector:npc"},
        {"type": "near_miss", "time_s": 6.0,
         "hazard": "pedestrian:ped"},
        {"type": "unnecessary_horn", "time_s": 7.0},
    ]
    report = _report(
        decisions=decisions,
        events=events,
        metrics={"unsafe_lane_changes": 1},
    )
    report["hard_violations"] = [{
        "type": "unsafe_lane_change", "count": 1,
    }]

    result = calculate_driving_process_score(report)

    assert result["hard_gate_triggered"] is False
    assert result["deduction_total"] == 82
    assert result["driving_process_score_100"] == 18.0
    assert [item["points"] for item in result["deductions"]] == [
        10, 10, 10, 15, 15, 20, 2,
    ]


def test_process_score_keeps_documented_leader_risk_threshold():
    report = _report(decisions=[{
        "type": "leader_response",
        "start_time_s": 0.0,
        "end_time_s": 0.6,
        "reasonable": False,
        "details": {
            "trigger": "startup_closing_leader",
            "entry_ttc_s": 5.5,
        },
    }])
    result = calculate_driving_process_score(report)
    assert result["deduction_total"] == 0


def test_short_and_long_overspeed_are_separated_at_five_seconds():
    short_samples = [
        _sample(index / 10, overspeed=True, speed=34.0)
        for index in range(13)
    ] + [_sample(1.3)]
    short = calculate_driving_process_score(
        _report(), samples=short_samples)
    assert short["hard_gate_triggered"] is False
    assert short["deduction_total"] == 5
    assert short["deductions"][0]["type"] == "short_overspeed"

    long_samples = [
        _sample(index / 10, overspeed=True, speed=34.0)
        for index in range(51)
    ] + [_sample(5.1)]
    long = calculate_driving_process_score(
        _report(), samples=long_samples)
    assert long["hard_gate_triggered"] is False
    assert long["driving_process_score_100"] == 80.0
    assert long["hard_gate_events"] == []
    assert long["deduction_total"] == 20
    assert long["deductions"][0]["type"] == "long_overspeed"


def test_collision_and_red_light_zero_the_score_but_unfinished_task_does_not():
    collision = calculate_driving_process_score(
        _report(metrics={"collision_count": 1}))
    red_light = calculate_driving_process_score(
        _report(metrics={"red_light_violations": 1}))
    unfinished = calculate_driving_process_score(_report(arrived=False))

    assert collision["hard_gate_triggered"] is True
    assert red_light["hard_gate_triggered"] is True
    assert [item["type"] for item in collision["hard_gate_events"]] == [
        "collision"]
    assert [item["type"] for item in red_light["hard_gate_events"]] == [
        "red_light_violation"]
    assert unfinished["hard_gate_events"] == []
    assert collision["deductions"] == []
    assert red_light["deductions"] == []
    assert unfinished["deductions"] == []
    assert collision["deduction_total"] == 0
    assert red_light["deduction_total"] == 0
    assert unfinished["deduction_total"] == 0
    assert collision["hard_gate_penalty_points"] == 100
    assert red_light["hard_gate_penalty_points"] == 100
    assert collision["driving_process_score_100"] == 0.0
    assert red_light["driving_process_score_100"] == 0.0
    assert red_light["metric_revision"] == (
        "at_fault_collision_red_light_hard_gate_v9")
    assert "red_light_points" not in red_light["thresholds_and_points"]
    assert single_vehicle_layer_score_100({
        "driving_process": red_light}) == 0.0
    assert unfinished["driving_process_score_100"] == 100.0


def test_collision_and_red_light_gate_only_once_even_with_other_deductions():
    report = _report(
        metrics={"collision_count": 2, "red_light_violations": 2},
        events=[
            {"type": "red_light_entry", "time_s": 1.0},
            {"type": "red_light_entry", "time_s": 2.0},
        ],
    )
    samples = [
        _sample(index / 10, overspeed=True, speed=34.0)
        for index in range(13)
    ] + [_sample(1.3)]
    result = calculate_driving_process_score(report, samples=samples)

    assert result["hard_gate_triggered"] is True
    assert [(item["type"], item["count"])
            for item in result["hard_gate_events"]] == [
                ("collision", 2), ("red_light_violation", 2)]
    assert result["hard_gate_penalty_points"] == 100
    assert result["deduction_total"] == 5
    assert result["rule_score_100"] == 95.0
    assert [item["type"] for item in result["deductions"]] == [
        "short_overspeed"]
    assert result["driving_process_score_100"] == 0.0


@pytest.mark.parametrize("at_fault_count, red_count, gate_types", [
    (0, 0, []),
    (1, 0, ["collision"]),
    (0, 1, ["red_light_violation"]),
])
def test_passive_collision_preserves_other_penalties(
        at_fault_count, red_count, gate_types):
    report = _report(metrics={
        "collision_count": 2,
        "at_fault_collision_count": at_fault_count,
        "red_light_violations": red_count,
    })
    samples = [
        _sample(index / 10, overspeed=True, speed=34.0)
        for index in range(13)
    ] + [_sample(1.3)]
    result = calculate_driving_process_score(report, samples=samples)
    assert [item["type"] for item in result["hard_gate_events"]] == gate_types
    if at_fault_count:
        assert result["hard_gate_events"][0]["count"] == at_fault_count
    assert result["deduction_total"] == 5
    assert result["driving_process_score_100"] == (0.0 if gate_types else 95.0)


def test_pedestrian_collision_hard_violation_triggers_gate_without_metric():
    report = _report()
    report["hard_violations"] = [{
        "type": "vehicle_pedestrian_collision", "time_s": 4.0,
    }]
    result = calculate_driving_process_score(report)

    assert result["hard_gate_triggered"] is True
    assert result["hard_gate_events"][0]["type"] == "collision"
    assert result["driving_process_score_100"] == 0.0


@pytest.mark.parametrize("stop_duration_s, charge_times_s", [
    (3.0, []),
    (3.1, [3.0]),
    (7.9, [3.0]),
    (8.0, [3.0, 8.0]),
    (13.0, [3.0, 8.0, 13.0]),
])
def test_long_unexplained_stop_charges_every_five_seconds(
        stop_duration_s, charge_times_s):
    samples = [
        _sample(index / 10, speed=0, unexplained_stop=True)
        for index in range(int(round(stop_duration_s * 10)))
    ] + [_sample(stop_duration_s, speed=0)]

    result = calculate_driving_process_score(_report(), samples=samples)
    deductions = [item for item in result["deductions"]
                  if item["type"] == "unexplained_stop"]

    assert [item["start_time_s"] for item in deductions] == charge_times_s
    assert [item["end_time_s"] for item in deductions] == charge_times_s
    assert [item["evidence"]["block_index"] for item in deductions] == (
        list(range(1, len(charge_times_s) + 1)))
    assert all(item["evidence"]["episode_start_time_s"] == 0.0
               and item["evidence"]["episode_end_time_s"] == stop_duration_s
               for item in deductions)
    assert result["deduction_total"] == 5 * len(charge_times_s)
    assert result["driving_process_score_100"] == 100 - result[
        "deduction_total"]


def test_same_rule_has_three_second_cooldown_and_red_light_is_only_a_gate():
    result = calculate_driving_process_score(
        _report(
            decisions=[
                {
                    "type": "leader_response",
                    "start_time_s": 1.0,
                    "end_time_s": 1.0,
                    "reasonable": False,
                    "details": {},
                },
                {
                    "type": "leader_response",
                    "start_time_s": 2.0,
                    "end_time_s": 2.0,
                    "reasonable": False,
                    "details": {},
                },
                {
                    "type": "leader_response",
                    "start_time_s": 4.0,
                    "end_time_s": 4.0,
                    "reasonable": False,
                    "details": {},
                },
                {
                    "type": "connector_response",
                    "start_time_s": 2.0,
                    "end_time_s": 2.0,
                    "reasonable": False,
                    "details": {},
                },
            ],
            events=[
                {"type": "red_light_entry", "time_s": 1.0,
                 "connector_id": "red-a"},
                {"type": "red_light_entry", "time_s": 2.0,
                 "connector_id": "red-b"},
            ],
        ))

    types = [item["type"] for item in result["deductions"]]
    assert types.count("leader_response_late") == 2
    assert types.count("connector_response_late") == 1
    assert "red_light_violation" not in types
    assert result["deduction_total"] == 30
    assert result["rule_score_100"] == 70.0
    assert result["hard_gate_triggered"] is True
    assert result["hard_gate_events"][0]["count"] == 2
    assert result["driving_process_score_100"] == 0.0
    assert result["rule_cooldown_s"] == 3.0
    assert result["rule_cooldown_exempt_types"] == []


def test_lane_change_deductions_use_adjusted_ten_point_values():
    samples = [
        _sample(0.0),
        _sample(
            0.1,
            is_changing_lane=True,
            current_lane=0,
            target_lane=1,
            target_lane_gap_m=5.0,
            target_lane_rear_ttc_s=3.0,
        ),
        _sample(0.2, is_changing_lane=False),
    ]
    result = calculate_driving_process_score(_report(), samples=samples)
    assert result["deduction_total"] == 20
    assert {
        item["type"]: item["points"] for item in result["deductions"]
    } == {
        "unsignaled_lane_change": 10,
        "unsafe_lane_change": 10,
    }

    scored_report = _report()
    scored_report["hard_safety_passed"] = False
    scored_report["driving_process"] = result
    assert single_vehicle_layer_score_100(scored_report) == 80.0


def test_native_sumo_lane_transition_is_observed_without_false_signal_fault():
    samples = [
        _sample(
            index / 10,
            current_segment="road-a",
            current_lane_id="road-a::lane_0",
            current_lane=0,
            left_indicator=index > 0,
        )
        for index in range(6)
    ]
    samples.append(_sample(
        0.6,
        current_segment="road-a",
        current_lane_id="road-a::lane_1",
        current_lane=1,
        left_indicator=False,
        current_lane_gap_m=20.0,
        current_lane_rear_ttc_s=4.0,
    ))

    result = calculate_driving_process_score(_report(), samples=samples)

    assert result["observed_lane_change_count"] == 1
    assert result["lane_change_events"] == [{
        "time_s": 0.6,
        "direction": "left",
        "source_lane": 0,
        "target_lane": 1,
        "source": "sumo_native_lane_transition",
    }]
    assert not any(
        item["type"] == "unsignaled_lane_change"
        for item in result["deductions"])


def test_windowed_acceleration_filters_ten_hertz_acceleration_noise():
    # The speed follows a smooth 2 m/s² ramp even if a simulator's reported
    # instantaneous acceleration jitters between adjacent 0.1-second steps.
    history = [
        (index / 10, 18.0 + 3.6 * 2.0 * index / 10)
        for index in range(11)
    ]
    acceleration = windowed_longitudinal_acceleration(
        history, 1.1, 18.0 + 3.6 * 2.0 * 1.1, 1.0)
    assert acceleration is not None
    assert abs(acceleration - 2.0) < 1e-9


def test_environment_deductions_use_delivered_time_and_actual_state():
    off = {
        "front_wiper_on": False,
        "front_fog_on": False,
        "rear_fog_on": False,
        "low_beam_on": False,
        "position_light_on": False,
        "all_windows_closed": True,
        "sunroof_closed": True,
    }
    samples = [_sample(0.0, equipment=off), _sample(2.0, equipment=off)]
    events = [
        {
            "event_id": "weather-1",
            "event_type": "weather_changed",
            "delivered_at_s": 0.0,
            "details": {"condition": "rainy"},
        },
        {
            "event_id": "daynight-1",
            "event_type": "daynight_changed",
            "delivered_at_s": 0.0,
            "details": {"period": "night"},
        },
    ]
    result = calculate_driving_process_score(
        _report(), samples=samples, delivered_events=events)
    # Weather and day/night compliance is reported by cabin_layer_score and
    # must not affect the 100-point driving-process score.
    assert result["deduction_total"] == 0
    assert result["deductions"] == []


def test_fog_requires_all_visibility_lights_and_selective_cleanup():
    lights_off = {
        "front_wiper_on": False,
        "front_fog_on": False,
        "rear_fog_on": False,
        "low_beam_on": False,
        "position_light_on": False,
        "all_windows_closed": False,
        "sunroof_closed": False,
    }
    fog_lights_on = {
        **lights_off,
        "front_fog_on": True,
        "rear_fog_on": True,
        "low_beam_on": True,
        "position_light_on": True,
    }
    after_fog = {
        **fog_lights_on,
        "front_fog_on": False,
        "rear_fog_on": False,
        "position_light_on": False,
    }
    samples = [
        _sample(0.0, equipment=lights_off),
        _sample(2.0, equipment=lights_off),
        _sample(3.0, equipment=fog_lights_on),
        _sample(5.0, equipment=after_fog),
    ]
    result = calculate_driving_process_score(
        _report(), samples=samples, delivered_events=[
            {
                "event_id": "fog-start",
                "event_type": "weather_changed",
                "delivered_at_s": 0.0,
                "details": {
                    "previous_condition": "sunny",
                    "condition": "foggy",
                },
            },
            {
                "event_id": "fog-clear",
                "event_type": "weather_changed",
                "delivered_at_s": 3.0,
                "details": {
                    "previous_condition": "foggy",
                    "condition": "sunny",
                },
            },
        ])
    assert result["deductions"] == []
    assert after_fog["low_beam_on"] is True
    assert after_fog["all_windows_closed"] is False
    assert after_fog["sunroof_closed"] is False


def test_weather_cleanup_keeps_position_light_required_by_dark_period():
    equipment = {
        "front_wiper_on": False,
        "front_fog_on": False,
        "rear_fog_on": False,
        "low_beam_on": True,
        "position_light_on": True,
        "all_windows_closed": True,
        "sunroof_closed": True,
    }
    samples = [
        _sample(0.0, equipment=equipment),
        _sample(4.0, equipment=equipment),
    ]
    result = calculate_driving_process_score(
        _report(), samples=samples, delivered_events=[
            {
                "event_id": "night-start",
                "event_type": "daynight_changed",
                "delivered_at_s": 0.0,
                "details": {"period": "night"},
            },
            {
                "event_id": "fog-clear",
                "event_type": "weather_changed",
                "delivered_at_s": 2.0,
                "details": {
                    "previous_condition": "foggy",
                    "condition": "sunny",
                },
            },
        ])
    assert result["deductions"] == []


def test_equipment_snapshot_reads_real_vehicle_modules():
    vw = VehicleWorld()
    baseline = _equipment_snapshot(vw)
    assert baseline == {
        "front_wiper_on": False,
        "front_fog_on": False,
        "rear_fog_on": False,
        "low_beam_on": False,
        "position_light_on": False,
        "all_windows_closed": True,
        "sunroof_closed": True,
    }

    vw.wiper.carcontrol_wiperBlade_switch(True, "front")
    vw.lowBeamHeadlight.switch("on")
    changed = _equipment_snapshot(vw)
    assert changed["front_wiper_on"] is True
    assert changed["low_beam_on"] is True


def test_equipment_snapshot_uses_sumo_npc_weather_state():
    vw = VehicleWorld()
    equipment = _equipment_snapshot(vw, {
        "front_wiper_on": True,
        "front_fog_on": True,
        "rear_fog_on": True,
        "low_beam_on": True,
        "position_light_on": True,
        "all_windows_closed": True,
        "sunroof_closed": True,
    })
    assert equipment == {
        "front_wiper_on": True,
        "front_fog_on": True,
        "rear_fog_on": True,
        "low_beam_on": True,
        "position_light_on": True,
        "all_windows_closed": True,
        "sunroof_closed": True,
    }


def test_short_threshold_fluctuation_does_not_duplicate_dynamic_episode():
    samples = [
        _sample(index / 10, hard_acceleration=True)
        for index in range(6)
    ]
    samples.append(_sample(0.6))
    samples.extend(
        _sample(index / 10, hard_acceleration=True)
        for index in range(9, 15)
    )
    samples.append(_sample(1.5))

    result = calculate_driving_process_score(_report(), samples=samples)
    assert [
        item["type"] for item in result["deductions"]
    ] == ["hard_acceleration"]
    assert result["deduction_total"] == 3


def test_soft_deduction_total_is_not_capped():
    events = [
        {
            "type": "near_miss",
            # Each event follows a complete safe interval, so these are six
            # independent encounters rather than one crowded encounter.
            "time_s": float(index * 2),
            "hazard": f"pedestrian:ped-{index}",
        }
        for index in range(6)
    ]
    result = calculate_driving_process_score(_report(events=events))
    assert result["hard_gate_triggered"] is False
    assert result["deduction_total"] == 60
    assert result["driving_process_score_100"] == 40.0


def test_crowded_pedestrian_findings_charge_once_per_continuous_encounter():
    decisions = [
        {
            "type": "pedestrian_response",
            "start_time_s": 9.3,
            "end_time_s": 9.9,
            "reasonable": False,
            "details": {"hazard": f"pedestrian:ped-{index}"},
        }
        for index in range(8)
    ]
    events = [
        {
            "type": "near_miss",
            "time_s": 9.4,
            "hazard": f"pedestrian:ped-{index}",
        }
        for index in range(8)
    ]

    result = calculate_driving_process_score(
        _report(decisions=decisions, events=events))

    pedestrian_deductions = [
        item for item in result["deductions"]
        if item["type"] in (
            "pedestrian_response_late", "pedestrian_near_miss")
    ]
    assert len(pedestrian_deductions) == 1
    assert pedestrian_deductions[0]["type"] == "pedestrian_near_miss"
    assert pedestrian_deductions[0]["points"] == 20
    assert pedestrian_deductions[0]["start_time_s"] == 9.3
    assert pedestrian_deductions[0]["end_time_s"] == 9.9
    aggregation = pedestrian_deductions[0]["evidence"][
        "pedestrian_encounter_aggregation"]
    assert aggregation["source_finding_count"] == 16
    assert aggregation["suppressed_finding_count"] == 15
    assert len(aggregation["pedestrian_hazards"]) == 8
    assert len(aggregation["source_findings"]) == 16


def test_pedestrian_warning_and_later_body_proximity_share_one_deduction():
    decision = {
        "type": "pedestrian_response",
        "start_time_s": 2.6,
        "end_time_s": 3.2,
        "reasonable": False,
        "details": {
            "hazard": "pedestrian:pedestrian",
            "pedestrian_encounter_id": "pedestrian:pedestrian:encounter:1",
        },
    }
    late_only = calculate_driving_process_score(_report(decisions=[decision]))
    assert late_only["deduction_total"] == 10
    assert late_only["deductions"][0]["type"] == "pedestrian_response_late"

    near_miss = {
        "type": "near_miss",
        "time_s": 5.1,
        "hazard": "pedestrian:pedestrian",
        "pedestrian_encounter_id": "pedestrian:pedestrian:encounter:1",
        "body_clearance_m": 1.389,
        "vehicle_speed_kmh": 9.12,
        "vehicle_ttc_s": 0.0,
    }
    combined = calculate_driving_process_score(_report(
        decisions=[decision], events=[near_miss]))
    assert combined["deduction_total"] == 20
    deduction = combined["deductions"][0]
    assert deduction["type"] == "pedestrian_near_miss"
    assert deduction["thresholds"]["body_clearance_m"] == 1.5
    assert deduction["thresholds"]["min_vehicle_speed_kmh"] == 5.0
    assert deduction["evidence"]["body_clearance_m"] == 1.389
    assert deduction["evidence"][
        "pedestrian_encounter_aggregation"]["source_finding_count"] == 2


def test_pedestrian_risk_can_be_charged_again_after_safe_clear_interval():
    events = [
        {"type": "near_miss", "time_s": 2.3,
         "hazard": "pedestrian:ped-1"},
        {"type": "near_miss", "time_s": 2.3,
         "hazard": "pedestrian:ped-2"},
        {"type": "near_miss", "time_s": 3.301,
         "hazard": "pedestrian:ped-3"},
    ]

    result = calculate_driving_process_score(_report(events=events))

    pedestrian_deductions = [
        item for item in result["deductions"]
        if item["type"] == "pedestrian_near_miss"
    ]
    assert len(pedestrian_deductions) == 1
    assert result["deduction_total"] == 20


@pytest.mark.parametrize("second_time", [3.6, 3.601, 9.5, 90.0])
def test_legacy_same_pedestrian_uses_temporal_continuity_only(second_time):
    events = [
        {"type": "near_miss", "time_s": 2.6,
         "hazard": "pedestrian:ped-1"},
        # Without lifecycle evidence, the same entity ID is not sufficient
        # to link findings beyond the one-second clear interval.
        {"type": "near_miss", "time_s": second_time,
         "hazard": "pedestrian:ped-1"},
    ]

    result = calculate_driving_process_score(_report(events=events))

    expected_count = 1 if second_time < 5.6 else 2
    pedestrian_deductions = [
        item for item in result["deductions"]
        if item["type"] == "pedestrian_near_miss"
    ]
    assert len(pedestrian_deductions) == expected_count
    assert result["deduction_total"] == 20 * expected_count


@pytest.mark.parametrize("first_id,second_id", [
    (None, None),
    ("", ""),
    ("pedestrian:ped-1:encounter:1", None),
    (None, "pedestrian:ped-1:encounter:1"),
    ("pedestrian:ped-1", None),
])
def test_missing_encounter_identity_cannot_link_distant_findings(
    first_id, second_id,
):
    events = [
        {"type": "near_miss", "time_s": time_s,
         "hazard": "pedestrian:ped-1",
         "pedestrian_encounter_id": encounter_id}
        for time_s, encounter_id in ((2.6, first_id), (90.0, second_id))
    ]

    result = calculate_driving_process_score(_report(events=events))

    assert result["deduction_total"] == 40


def test_same_explicit_pedestrian_encounter_is_charged_once_across_ttc_gap():
    events = [
        {"type": "near_miss", "time_s": time_s,
         "hazard": "pedestrian:ped-1",
         "pedestrian_encounter_id": "pedestrian:ped-1:encounter:1"}
        for time_s in (2.6, 9.5)
    ]

    result = calculate_driving_process_score(_report(events=events))

    pedestrian_deductions = [
        item for item in result["deductions"]
        if item["type"] == "pedestrian_near_miss"
    ]
    assert len(pedestrian_deductions) == 1
    assert result["deduction_total"] == 20
    aggregation = pedestrian_deductions[0]["evidence"][
        "pedestrian_encounter_aggregation"]
    assert aggregation["source_finding_count"] == 2
    assert aggregation["pedestrian_hazards"] == ["pedestrian:ped-1"]


def test_distinct_physical_appearances_of_same_pedestrian_can_be_charged():
    events = [
        {"type": "near_miss", "time_s": 2.6,
         "hazard": "pedestrian:ped-1",
         "pedestrian_encounter_id": "pedestrian:ped-1:encounter:1"},
        {"type": "near_miss", "time_s": 9.5,
         "hazard": "pedestrian:ped-1",
         "pedestrian_encounter_id": "pedestrian:ped-1:encounter:2"},
    ]

    result = calculate_driving_process_score(_report(events=events))

    pedestrian_deductions = [
        item for item in result["deductions"]
        if item["type"] == "pedestrian_near_miss"
    ]
    assert len(pedestrian_deductions) == 2
    assert result["deduction_total"] == 40
