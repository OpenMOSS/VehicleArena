"""Offline overlap, owner-thread, commit-barrier and failure-isolation tests."""
import threading
import time
import copy
from types import SimpleNamespace

import pytest
from greenlet import getcurrent

from simulation.model_concurrency import ModelCallScheduler, model_io, on_simulation_owner
from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from evaluation.agent_client import AgentClient


def test_model_calls_overlap_but_callbacks_and_rendering_keep_owner():
    owner_thread, owner_greenlet = threading.get_ident(), getcurrent()
    barrier = threading.Barrier(2, timeout=5)
    clients = []

    def create(**kwargs):
        assert threading.get_ident() != owner_thread
        barrier.wait()
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")])

    for _ in range(2):
        client = AgentClient(api_key="offline-test", model="Qwen3.8-27B")
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        clients.append(client)

    def owner_render():
        assert threading.get_ident() == owner_thread
        assert getcurrent() is owner_greenlet
        return "image"

    def callback(client):
        for _ in range(2):
            assert threading.get_ident() == owner_thread
            assert on_simulation_owner(owner_render) == "image"
            message, *_ = client.chat_with_tools([], [])
            assert message.content == "ok"
            assert client.last_call_metadata["ok"]
        return client.model

    with ModelCallScheduler(2) as scheduler:
        assert scheduler.run([lambda c=c: callback(c) for c in clients]) == ["Qwen3.8-27B"] * 2


def test_parallelism_cap_serial_switch_and_exception_drain():
    active = peak = 0
    lock = threading.Lock()
    completed = []

    @model_io
    def request(index):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
        time.sleep(0.03)
        with lock:
            active -= 1
        if index == 0:
            raise ValueError("failed request")
        completed.append(index)
        return index

    with ModelCallScheduler(2) as scheduler:
        with pytest.raises(ValueError, match="failed request"):
            scheduler.run([lambda i=i: request(i) for i in range(5)])
    assert peak == 2
    assert sorted(completed) == [1, 2, 3, 4]
    owner = threading.get_ident()

    @model_io
    def serial_request():
        return threading.get_ident()

    with ModelCallScheduler(1) as scheduler:
        assert scheduler.run([serial_request, serial_request]) == [owner, owner]


def test_owner_failure_reaches_only_its_callback_and_scheduler_is_reusable():
    def fail_render():
        raise RuntimeError("render failed")

    def job():
        with pytest.raises(RuntimeError, match="render failed"):
            on_simulation_owner(fail_render)
        return "recovered"

    with ModelCallScheduler(2) as scheduler:
        assert scheduler.run([job, lambda: "other"]) == ["recovered", "other"]
        assert scheduler.run([lambda: "next boundary"]) == ["next boundary"]


def test_scenario_parallelism_defaults_and_validation():
    raw = {"scenario_id": "config", "road_network_id": "beijing_guomao", "vehicles": []}
    assert MultiScenario.from_dict(raw).max_parallel_model_calls == 4
    assert MultiScenario.from_dict({**raw, "max_parallel_model_calls": 1}).max_parallel_model_calls == 1
    for value in (0, -1, True, 1.5, "4"):
        with pytest.raises(ValueError, match="positive integer"):
            MultiScenario.from_dict({**raw, "max_parallel_model_calls": value})


@pytest.mark.parametrize("fail_first", [False, True])
def test_two_vehicle_frozen_batch_commits_in_scenario_order(monkeypatch, tmp_path, fail_first):
    owner_thread, owner_greenlet = threading.get_ident(), getcurrent()
    rendered = []

    class Camera:
        def render(self, engine, vehicle_id, t, tick):
            assert threading.get_ident() == owner_thread
            assert getcurrent() is owner_greenlet
            rendered.append(vehicle_id)
            return None

        def close(self):
            assert getcurrent() is owner_greenlet

    monkeypatch.setattr("visualization.web3d_camera.Web3DCameraRenderer", Camera)
    scenario = MultiScenario.from_dict({
        "scenario_id": "parallel_frozen_batch", "road_network_id": "beijing_guomao",
        "total_time_s": 0.1, "tick_interval_s": 100,
        "sumo_config": {"cache_root": str(tmp_path), "suppress_warnings": True},
        "vehicles": [
            {"vehicle_id": vid, "initial_node": node, "destination_node": "n35722739",
             "agent_config": {"type": "llm"}, "disable_modules": ["lidar"]}
            for vid, node in [("first", "n33399858"), ("second", "n35553582")]
        ]})
    engine = MultiSimEngine(scenario)
    barrier = threading.Barrier(2, timeout=5)
    observed = []
    committed = []
    original_commit = engine._commit_agent_commands

    def commit(t):
        committed.extend(item["entity_id"] for item in engine._agent_command_queue)
        original_commit(t)

    monkeypatch.setattr(engine, "_commit_agent_commands", commit)

    @model_io
    def request(vid):
        barrier.wait()
        if vid == "first":
            time.sleep(0.03)  # Deliberately finish second first.

    def make_callback(vid):
        def callback(vw, t, messages, memory, tick, **kwargs):
            if t != 0:
                return []
            before = {other: copy.deepcopy(engine.traffic_mgr.get_state(other).llm_control_command)
                      for other in ("first", "second")}
            request(vid)
            assert threading.get_ident() == owner_thread
            assert all(engine.traffic_mgr.get_state(other).llm_control_command == before[other]
                       for other in before)
            ack = vw.navigation.navigation_set_speed(20, reason=vid)
            assert ack["queued_for_world_commit"]
            request(vid)
            assert all(engine.traffic_mgr.get_state(other).llm_control_command == before[other]
                       for other in before)
            observed.append((vid, t))
            if fail_first and vid == "first":
                raise RuntimeError("intentional failure after sibling queued commands")
            return []
        return callback

    engine.run({vid: make_callback(vid) for vid in ("first", "second")})
    assert sorted(observed) == [("first", 0), ("second", 0)], engine.agent_callback_errors
    assert set(rendered) == {"first", "second"}
    assert committed == (["second"] if fail_first else ["first", "second"])
    assert engine.traffic_mgr.get_state("second").llm_control_command["reason"] == "second"
    assert bool(engine.agent_callback_errors) == fail_first
