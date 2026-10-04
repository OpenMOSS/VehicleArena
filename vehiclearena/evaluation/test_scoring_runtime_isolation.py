"""The new assessment must not change the native reference vehicle's motion."""
import copy
import json
from pathlib import Path

from evaluation.experiments.telemetry import ExperimentTrackedEngine
from simulation.multi_sim_engine import MultiScenario, MultiSimResult, VehicleResult


def test_multillm_process_score_uses_declared_focal_vehicle_only():
    result = MultiSimResult(
        scenario_id="multi_llm",
        vehicle_results={
            "ego": VehicleResult(
                vehicle_id="ego", is_evaluated=True,
                driving_evaluation={
                    "driving_process": {
                        "driving_process_score_100": 80.0,
                        "hard_gate_triggered": False,
                    },
                }),
            "fixed_peer": VehicleResult(
                vehicle_id="fixed_peer", is_evaluated=True,
                driving_evaluation={
                    "driving_process": {
                        "driving_process_score_100": 20.0,
                        "hard_gate_triggered": True,
                    },
                }),
        },
    )
    result._scenario_config = {
        "experiment_scene": {"focal_vehicle_id": "ego"},
        "vehicles": [
            {"vehicle_id": "ego", "agent_config": {"type": "llm"}},
            {"vehicle_id": "fixed_peer", "agent_config": {"type": "llm"}},
        ],
    }
    assert result.overall_single_vehicle_layer_score_100 == 80.0
    assert result.driving_process_hard_gate_triggered is False
    result.vehicle_results["ego"].driving_evaluation[
        "driving_process"]["hard_gate_triggered"] = True
    assert result.driving_process_hard_gate_triggered is True


def test_process_score_is_attached_without_changing_physics(monkeypatch):
    class NoCamera:
        def render(self, *args, **kwargs):
            return None

        def close(self):
            pass

    monkeypatch.setattr("visualization.web3d_camera.Web3DCameraRenderer", NoCamera)
    path = Path(__file__).parent / (
        "experiments/scenarios/Basic/basic_009_signalized_intersection/scenario.json")
    raw = json.loads(path.read_text())
    focal_id = raw["experiment_scene"]["focal_vehicle_id"]
    raw.update(total_time_s=3.0, stop_when_all_vehicles_terminal=False)
    for vehicle in raw["vehicles"]:
        vehicle["agent_config"] = {"type": "sumo"}
    trajectories = []
    for enabled in (False, True):
        scene = copy.deepcopy(raw)
        scene["enable_driving_evaluation"] = enabled
        engine = ExperimentTrackedEngine(MultiScenario.from_dict(scene))
        result = engine.run({})
        assert not engine.agent_callback_errors
        assert not hasattr(engine.traffic_mgr, "_evaluated_npc_should_hold_for_pedestrian")
        assert not hasattr(engine.traffic_mgr, "_evaluated_npc_should_hold_for_shared_corridor")
        fields = ("vehicle_id", "time_s", "pose_x_m", "pose_y_m", "speed_kmh")
        trajectories.append([tuple(row[k] for k in fields)
                             for row in result._vehicle_trajectory])
        if enabled:
            report = result.vehicle_results[focal_id].driving_evaluation
            process = report["driving_process"]
            assert process["evaluation_type"] == "exam_style_deduction_v1"
            assert process["hard_gate_triggered"] is False
            assert not any(e["type"] in {"collision", "not_arrived"}
                           for e in process["deductions"])
    assert trajectories[0] == trajectories[1]
