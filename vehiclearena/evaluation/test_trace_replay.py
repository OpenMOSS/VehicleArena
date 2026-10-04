"""Contract tests for portable replay data and its two synchronized cameras."""

import gzip
import json

from scripts.visualization import export_trace_replay as replay


def test_export_preserves_environment_presence_and_pedestrians(tmp_path, monkeypatch):
    lane = {"id": "a", "centerline_xy": [[0, 0], [20, 0]],
            "left_boundary_xy": [[0, 2], [20, 2]],
            "right_boundary_xy": [[0, -2], [20, -2]]}
    stop = {"id": "stop", "lane_id": "a", "line_xy": [[19, 2], [19, -2]]}
    net = {"lanes": [lane], "connectors": [], "stop_lines": [stop],
           "crosswalks": [{"id": "cw", "polygon_xy": [[15, -2], [16, -2], [16, 2], [15, 2]]}]}
    (tmp_path / "test_lane_level.json").write_text(json.dumps(net))
    monkeypatch.setattr(replay, "NETDIR", tmp_path)
    rows = [{"vehicle_id": "ego", "time_s": t, "pose_x_m": t, "pose_y_m": 0,
             "yaw_rad": 0, "speed_kmh": 10, "current_lane_id": "a",
             "active_connector_id": "", "present_in_physics_world": present}
            for t, present in [(0, True), (.1, True), (.2, False), (.3, False)]]
    trace = {"variant": {"variant_id": "test", "scenario": {"road_network_id": "test"}},
             "result": {"vehicles": {}}, "trajectories": {"vehicles": rows,
             "pedestrians": [{"ped_id": "p", "time_s": 0, "pose_x_m": None,
                               "pose_y_m": None, "spawned": False},
                              {"ped_id": "p", "time_s": 1, "pose_x_m": 15,
                               "pose_y_m": 1, "spawned": True}]}}
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(trace, f)
    payload = replay.build_payload(path)
    assert payload["environment"]["stop_lines"] == [stop]
    assert payload["environment"]["lanes"][0]["left_boundary_xy"] == lane["left_boundary_xy"]
    assert payload["vehicles"]["ego"][-1][0] == .3
    assert payload["vehicles"]["ego"][1][7] is False
    assert payload["pedestrians"]["p"] == [[0, None, None, False], [1, 15, 1, True]]

    # A clean safety event list can still have process deductions (case 191).
    trace["result"]["vehicles"] = {"ego": {"driving_evaluation": {
        "events": [], "decision_episodes": [], "single_vehicle_layer_score_100": 84,
        "driving_process": {"deduction_total": 16, "deductions": [
            {"type": "unsignaled_turn", "start_time_s": .1, "end_time_s": .1, "points": 10},
            {"type": "hard_acceleration", "start_time_s": .2, "end_time_s": .3, "points": 3},
            *[{"type": "high_longitudinal_jerk", "start_time_s": 0, "end_time_s": .1, "points": 1} for _ in range(3)],
        ]}}}}
    with gzip.open(path, "wt") as f:
        json.dump(trace, f)
    payload = replay.build_payload(path)
    assert len(payload["penalties"]) == 5
    assert sum(item["points"] for item in payload["penalties"]) == 16
    assert payload["evaluations"]["ego"]["deduction_total"] == 16
    assert payload["evaluation_events"] == []


def test_deductions_do_not_double_count_evidence_or_invent_aggregate_time():
    report = {"events": [{"type": "red_light_entry", "time_s": 5}],
              "decision_episodes": [{"type": "signal_compliance", "reasonable": False}],
              "driving_process": {"deductions": [{"type": "red_light_violation",
                  "points": 20, "start_time_s": 0, "end_time_s": 0, "evidence": {"count": 1}}]}}
    deductions = replay._recorded_deductions("ego", report)
    assert len(deductions) == 1
    assert deductions[0]["points"] == 20
    assert deductions[0]["t"] is None
    assert replay._recorded_deductions("ego", {"events": report["events"]}) == []


def test_signal_phase_boundaries():
    state, _ = replay._build_signal_lookup({"signal_plans": [{"phases": [
        {"connector_ids": ["left"], "green_s": 5, "yellow_s": 3, "all_red_s": 2},
        {"connector_ids": ["straight"], "green_s": 7, "yellow_s": 3, "all_red_s": 2}]}]})
    assert state("left", 4.99) == "green"
    assert state("left", 5) == "yellow"
    assert state("left", 8) == "red"
    assert state("straight", 10) == "green"
    assert state("left", 22) == "green"
    assert state("missing", 1) == "unsignalized"
