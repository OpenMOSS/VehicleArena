"""Regression checks for fixed SUMO-reference experiment deadlines."""

from types import SimpleNamespace

from evaluation.experiments.time_window_calibration import (
    NPC_ARRIVAL_MARGIN_S,
    _actor_configuration,
    _last_scheduled_event_s,
    _peer_in_cabin_runtime_configs,
    _public_peer_protocol,
    _should_retry,
    _persistent_obstacle_vehicle_ids,
    _terminal_time_by_vehicle,
    _time_limit_from_npc_arrival,
)
from simulation.multi_sim_engine import MultiSimEngine


def test_time_limit_is_npc_arrival_plus_ten_simulation_seconds():
    assert NPC_ARRIVAL_MARGIN_S == 10.0
    assert _time_limit_from_npc_arrival(21.3) == 31.3
    assert _time_limit_from_npc_arrival(0.0) == 10.0


def test_multillm_calibration_enables_pa_and_judge_for_llm_peers():
    peer = {
        "api_base": "https://qwen.invalid/v1",
        "model": "Qwen3.8-27B",
        "temperature": 0.7,
        "context_window_tokens": 1000000,
        "max_tokens": 32768,
        "thinking_mode": "enabled",
        "reasoning_effort": "xhigh",
        "chat_template_enable_thinking": True,
    }

    personal, judge = _peer_in_cabin_runtime_configs(peer)

    assert personal == {
        **{key: value for key, value in peer.items() if key != "temperature"},
        "enabled": True,
        "temperature": 0.7,
        "trigger_mode": "event_random",
        "seed": 0,
    }
    assert judge == {
        **{key: value for key, value in peer.items() if key != "temperature"},
        "enabled": True,
        "temperature": 0.0,
        "check_offsets_s": [0.1, 1.0, 3.0],
        "max_checks": 3,
        "acceptance_timeout_s": 3.0,
    }
    protocol = _public_peer_protocol(peer)
    assert protocol["in_cabin_agents"]["scope"] == (
        "all_fixed_peer_llm_vehicles")
    assert protocol["in_cabin_agents"]["personal_agent"][
        "reasoning_effort"] == "xhigh"
    assert protocol["in_cabin_agents"]["passenger_judge"][
        "temperature"] == 0.0
    assert _actor_configuration("MultiLLM", peer).startswith(
        "focal_sumo_fixed_peers:")


def test_persistent_obstacles_are_declared_by_initial_crash_state():
    scenario = {
        "vehicles": [
            {"vehicle_id": "trip", "initial_physical_state": {}},
            {"vehicle_id": "wreck", "initial_physical_state": {
                "crashed": True}},
            {"vehicle_id": "ordinary", "initial_physical_state": {}},
        ],
    }
    assert _persistent_obstacle_vehicle_ids(scenario) == {"wreck"}


def test_late_pedestrian_entry_uses_minutes_in_calibration_schedule():
    assert _last_scheduled_event_s({
        "pedestrians": [{"start_time": 2.5}],
    }) == 150.0


def test_sumo_arrival_and_collision_are_both_calibration_terminals():
    kinds, times = _terminal_time_by_vehicle(
        {"arrived", "crashed"},
        {
            "arrived": SimpleNamespace(arrived=True, arrival_time_s=12.3),
            "crashed": SimpleNamespace(arrived=False, arrival_time_s=None),
        },
        {
            "arrived": SimpleNamespace(
                is_crashed=False, route_failed=False),
            "crashed": SimpleNamespace(
                is_crashed=True, route_failed=False),
        },
        [SimpleNamespace(
            entity_a="crashed", entity_b="peer", time_s=8.4)],
    )
    assert kinds == {"arrived": "arrived", "crashed": "crashed"}
    assert times == {"arrived": 12.3, "crashed": 8.4}


def test_unfinished_or_route_failed_vehicle_is_not_successful_terminal():
    kinds, times = _terminal_time_by_vehicle(
        {"failed", "moving"}, {},
        {
            "failed": SimpleNamespace(
                is_crashed=False, route_failed=True),
            "moving": SimpleNamespace(
                is_crashed=False, route_failed=False),
        }, [],
    )
    assert kinds == {"failed": "route_failed", "moving": "unfinished"}
    assert times == {}


def test_calibration_can_stop_on_sumo_subset_without_waiting_for_llm_peer():
    states = {
        "sumo": SimpleNamespace(
            arrived=False, route_failed=False, is_crashed=True),
        "llm_peer": SimpleNamespace(
            arrived=False, route_failed=False, is_crashed=False),
    }
    engine = SimpleNamespace(
        scenario=SimpleNamespace(
            terminal_vehicle_ids=["sumo"],
            vehicles=[
                SimpleNamespace(vehicle_id="sumo"),
                SimpleNamespace(vehicle_id="llm_peer"),
            ],
        ),
        traffic_mgr=SimpleNamespace(get_state=states.get),
    )
    assert MultiSimEngine._all_scenario_vehicles_terminal(engine) is True


def test_collision_does_not_block_retry_when_other_sumo_remain_unfinished():
    outcome = {
        "success": False,
        "collision_count": 2,
        "run_duration_s": 100.0,
        "error": "",
    }
    assert _should_retry(outcome, 1200.0) is True
    outcome["success"] = True
    assert _should_retry(outcome, 1200.0) is False
