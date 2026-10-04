"""Regressions from the ten-task Qwen sample; these tests make no API calls."""
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from simulation.sumo_map import SumoMapConverter
from simulation.sumo_traffic_manager import SumoTrafficManager
from evaluation.experiments.batch_runner import ExperimentBatchRunner, load_run
from evaluation.experiments.manifest import ExperimentManifest, ExperimentVariant
from evaluation.experiments.telemetry import ExperimentTrackedEngine


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "evaluation/fixtures/pre_pull_catalog"
HK_MAP = ROOT / "simulation/road_networks/hongkong_central_lane_level.json"
NEEDS_SUMO = pytest.mark.skipif(
    not HK_MAP.exists() or not shutil.which("netconvert") or not shutil.which("sumo"),
    reason="SUMO and offline maps required")


@pytest.fixture
def no_camera(monkeypatch):
    import visualization.web3d_camera

    class NoCamera:
        def render(self, *args):
            return None

        def close(self):
            pass

    monkeypatch.setattr(visualization.web3d_camera, "Web3DCameraRenderer", NoCamera)


@NEEDS_SUMO
def test_hongkong_compiler_does_not_consume_authored_approaches():
    bundle = SumoMapConverter().convert(HK_MAP)
    root = ET.parse(bundle.net_file).getroot()
    source = {lane["id"]: lane for lane in json.loads(HK_MAP.read_text())["lanes"]}
    for lane_id in ("n1615603704_n9231274165::lane_0",
                    "n988470348_n988470363::lane_0",
                    "n2496629720_n988470348::lane_1"):
        lane = root.find(f"edge[@id='{bundle.edge_by_lane[lane_id]}']/"
                         f"lane[@index='{bundle.lane_index_by_lane[lane_id]}']")
        assert float(lane.get("length")) >= source[lane_id]["length_m"] * 0.95


@NEEDS_SUMO
@pytest.mark.parametrize("parallel", [1, 4])
@pytest.mark.parametrize("study,name,drive", [
    ("Basic", "basic_010_signalized_intersection__hongkong_central", True),
    ("MultiLLM", "multi_006_signal_queue_start__hongkong_central", True),
    ("MultiLLM", "multi_010_two_lane_merge__hongkong_central", False),
    ("MultiLLM", "multi_014_unsignalized_four_way__hongkong_central", False),
])
def test_failed_hongkong_physics_with_real_actuator_authority(no_camera, parallel, study, name, drive):
    raw = json.loads((CATALOG / study / name / "scenario.json").read_text())
    raw.update(total_time_s=8.0 if drive else 2.0, max_parallel_model_calls=parallel)
    scenario = MultiScenario.from_dict(raw)
    engine = MultiSimEngine(scenario)
    initial = {}

    def make_callback(vid):
        def callback(vw, t, *args, **kwargs):
            if vid not in initial:
                vehicle = engine.traffic_mgr.get_state(vid)
                initial[vid] = (t, vehicle.current_speed_kmh)
                assert vehicle.route_control_authority == "llm_maneuver"
                if drive:
                    assert vw.navigation.navigation_select_maneuver("straight")["success"]
                    assert vw.navigation.navigation_set_speed(32)["success"]
            return []
        return callback

    engine.run({v.vehicle_id: make_callback(v.vehicle_id)
                for v in scenario.vehicles if v.agent_type == "llm"})
    assert not engine.agent_callback_errors
    assert set(initial) == {v.vehicle_id for v in scenario.vehicles if v.agent_type == "llm"}
    for item in raw["vehicles"]:
        if item["vehicle_id"] not in initial:
            continue
        vid = item["vehicle_id"]
        assert initial[vid] == (0.0, item["initial_physical_state"]["speed_kmh"])
        state = engine.traffic_mgr.get_state(vid)
        if drive:
            assert state.arrived
            assert state.current_node == state.destination_node
        else:
            assert not state.arrived
            assert state.present_in_physics_world
            assert not hasattr(engine.traffic_mgr, "_sumo_navigation_holds")
            assert state.active_connector_id or state.planned_maneuver_source == "default_straight"


@pytest.mark.parametrize("selected", ["active", "planned", "none"])
@pytest.mark.parametrize("installed,destination", [(["in", "out"], "goal"),
                                                   (["in"], "goal"),
                                                   (["in", "out"], "elsewhere")])
def test_arrival_between_samples_requires_connector_and_installed_destination(selected, installed, destination):
    manager = object.__new__(SumoTrafficManager)
    manager.sumo_map = SimpleNamespace(edge_by_lane={"incoming": "in", "target": "out"})
    manager._sumo_installed_routes = {"ego": installed}
    manager._lane_geometry = SimpleNamespace(
        _node_to_junction={},
        _lane_by_id={"incoming": {"end_node": "junction"},
                     "target": {"end_node": "goal", "index": 0, "segment_id": "s"}},
        connector_record=lambda cid: {"from_lane": "incoming", "to_lane": "target"})
    vehicle = SimpleNamespace(vehicle_id="ego", current_lane_id="incoming",
        destination_node=destination, current_speed_kmh=32,
        active_connector_id="c" if selected == "active" else "",
        planned_connector_id="c" if selected == "planned" else "",
        active_control_commands={"route_maneuver": {}})
    valid = selected != "none" and len(installed) == 2 and destination == "goal"
    assert manager._accept_llm_route_arrival(vehicle) is valid
    assert vehicle.current_lane_id == ("target" if valid else "incoming")
    assert vehicle.current_speed_kmh == 32


def test_failed_batch_preserves_partial_evidence_and_retries(tmp_path, monkeypatch):
    import evaluation.experiments.batch_runner as batch

    callback = lambda *args, **kwargs: []
    callback._state = {"tool_call_log": [{"tool": "navigation_set_speed"}],
                       "model_call_log": [{"prompt_tokens": 20}],
                       "personal_agent": {"api_key": "do-not-persist", "requests": ["AC"]}}
    monkeypatch.setattr(batch, "resolve_agent_specs", lambda *args, **kwargs: {})
    monkeypatch.setattr(batch, "apply_resolved_agent_authorities", lambda *args: None)
    monkeypatch.setattr(batch, "build_callbacks", lambda *args, **kwargs: ({"ego": callback}, {}))
    # No actual agent specs are needed; only the logging state is installed
    # inside the mock run, after output-directory setup.
    state = callback._state
    del callback._state

    class BrokenEngine:
        attempts = 0
        failure_snapshot = ExperimentTrackedEngine.failure_snapshot

        def __init__(self, scenario):
            self._vehicle_trajectory = [{"vehicle_id": "ego", "time_s": 0.1}]
            self._pedestrian_trajectory = []
            self._physics_events = [{"time_s": 0.1, "type": "brake_light_changed"}]
            self._sim_time = 0.2

        def run(self, callbacks):
            BrokenEngine.attempts += 1
            callbacks["ego"]._state = copy.deepcopy(state)
            raise RuntimeError("injected physics failure")

    monkeypatch.setattr(batch, "ExperimentTrackedEngine", BrokenEngine)
    variant = ExperimentVariant(variant_id="probe", base_scenario_id="probe",
        factors={}, scenario={"scenario_id": "probe", "road_network_id": "beijing_guomao"})
    manifest = ExperimentManifest(experiment_id="Basic", description="test", variants=[variant])
    runner = ExperimentBatchRunner(manifest, tmp_path, resume=True)
    result = runner.run()["variants"]["probe"]
    assert result["status"] == "failed"
    assert result["error"] == "RuntimeError: injected physics failure"
    first = Path(result["diagnostic_output"])
    data = load_run(first)
    assert data["run"]["partial"] and not data["run"]["evaluated"]
    assert data["trajectories"]["vehicles"][0]["time_s"] == 0.1
    assert data["callbacks"]["ego"]["model_call_log"][0]["prompt_tokens"] == 20
    assert "do-not-persist" not in json.dumps(data)
    assert not (tmp_path / "probe.json.gz").exists()
    del callback._state
    second = runner.run()["variants"]["probe"]
    assert BrokenEngine.attempts == 2
    assert second["diagnostic_output"] != str(first)
    assert first.exists()


def test_preflight_registers_llm_callbacks(monkeypatch):
    scripts = ROOT.parent / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import preflight_llm_physics as preflight
    raw = json.loads((CATALOG / "MultiLLM/multi_010_two_lane_merge__hongkong_central/scenario.json").read_text())

    class Engine:
        def __init__(self, scenario):
            assert scenario.physics_only_mode
            assert scenario.total_time_s == 0.2
            self.traffic_mgr = SimpleNamespace(get_state=lambda vid: SimpleNamespace(route_control_authority="llm_maneuver"))

        def run(self, callbacks):
            assert set(callbacks) == {"ego", "merging_vehicle"}

    monkeypatch.setattr(preflight, "MultiSimEngine", Engine)
    assert preflight.run_physics_preflight(raw)["status"] == "passed"


@NEEDS_SUMO
def test_invalid_initial_projection_fails_before_any_driver_call(no_camera, monkeypatch):
    raw = json.loads((CATALOG / "MultiLLM/multi_010_two_lane_merge__hongkong_central/scenario.json").read_text())
    raw["total_time_s"] = 0.2
    # Reproduce the old compiler's endpoint projection without corrupting any
    # offline source map or compiled cache on disk.
    monkeypatch.setattr(SumoTrafficManager, "_project_position_onto_shape",
                        staticmethod(lambda point, shape, length: length))
    # Exercise a genuinely closed actuator boundary (no unique default).
    monkeypatch.setattr(SumoTrafficManager, "_prepare_default_continuation",
                        lambda self, vehicle: False)
    engine = ExperimentTrackedEngine(MultiScenario.from_dict(raw))
    calls = []
    with pytest.raises(ValueError, match="Initial LLM placement projects onto a closed route endpoint"):
        engine.run({v.vehicle_id: lambda *args, **kwargs: calls.append(True)
                    for v in engine.scenario.vehicles})
    assert calls == []
    assert engine.failure_snapshot()["trajectories"]["vehicles"] == []


@NEEDS_SUMO
def test_mid_step_sumo_failure_saves_last_completed_frames(tmp_path, no_camera, monkeypatch):
    import evaluation.experiments.batch_runner as batch
    raw = json.loads((CATALOG / "Basic/basic_010_signalized_intersection__hongkong_central/scenario.json").read_text())
    raw.update(total_time_s=0.4, physics_only_mode=True)
    callback = lambda *args, **kwargs: []
    callback._state = {"model_call_log": [], "tool_call_log": []}
    monkeypatch.setattr(batch, "build_callbacks", lambda *args, **kwargs: ({"queue_1": callback}, {}))
    original = ExperimentTrackedEngine._log_per_substep

    def fail_after_frame(self, time_s, tick, events):
        original(self, time_s, tick, events)
        if time_s >= 0.2:
            raise RuntimeError("injected mid-step failure")

    monkeypatch.setattr(ExperimentTrackedEngine, "_log_per_substep", fail_after_frame)
    variant = ExperimentVariant(variant_id="mid_step", base_scenario_id=raw["scenario_id"],
        factors={}, scenario=raw, requires_llm=True)
    manifest = ExperimentManifest(experiment_id="Basic", description="failure probe", variants=[variant])
    result = ExperimentBatchRunner(manifest, tmp_path, allow_llm=True).run()["variants"]["mid_step"]
    assert result["status"] == "failed", result
    assert result["error"] == "RuntimeError: injected mid-step failure"
    data = load_run(Path(result["diagnostic_output"]))
    assert data["run"]["status"] == "failed"
    assert {row["time_s"] for row in data["trajectories"]["vehicles"]} >= {0.1, 0.2}
    assert data["last_public_time_s"] == 0.2
