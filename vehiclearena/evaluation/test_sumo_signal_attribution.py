"""Regressions for physical connector identity, independent of route intent."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.driving_evaluator import DrivingEvaluator
from simulation.road_networks import load_road_network
from simulation.sumo_traffic_manager import SumoTrafficManager


# Historical red-labelled entries: time, planned connector, physical connector.
CASES = {
    "losangeles_downtown": [
        (106.6, "n21420508::9", "n21420508::8"),
        (129.8, "n69091912::3", "n69091912::0"),
        (148.0, "n123611452::6", "n13411773158::2"),
        (286.9, "n123281798::5", "n250649832::12"),
        (326.3, "n21302766::21", "n21302766::18"),
        (377.1, "n18166171::7", "n18166168::8"),
    ],
    "nantong_chongchuan": [
        (104.8, "intersection::n1433677567::8",
         "intersection::n1433677567::7"),
    ],
}


@pytest.fixture(params=CASES)
def mapped_manager(request, tmp_path):
    network = request.param
    path = (Path(__file__).resolve().parents[1] / "simulation/road_networks"
            / f"{network}_lane_level.json")
    if not path.exists() or not all(shutil.which(x) for x in ("sumo", "netconvert")):
        pytest.skip("SUMO and offline maps are required")
    manager = SumoTrafficManager(
        load_road_network(network),
        sumo_config={"cache_root": str(tmp_path), "suppress_warnings": True})
    manager.close()
    return network, manager


def test_historical_entries_use_physical_lane_and_keep_identity_across_splits(mapped_manager):
    network, manager = mapped_manager
    geometry = manager._lane_geometry
    for time_s, planned, physical in CASES[network]:
        planned, physical = "connector::" + planned, "connector::" + physical
        connector = geometry.connector_record(physical)
        vehicle = SimpleNamespace(
            vehicle_id="ego", active_connector_id="",
            planned_connector_id=planned, planned_turn="right",
            lane_route_actions=[{"type": "connector", "connector_id": planned}],
            lane_route_action_index=0,
            pose_x_m=connector["centerline_xy"][0][0],
            pose_y_m=connector["centerline_xy"][0][1])
        manager._physics_time = time_s
        lanes = [lane for lane, cid in manager._sumo_connector_by_internal_lane.items()
                 if cid == physical]
        assert lanes
        assert geometry.signal_state(planned, time_s).signal == "red"
        for index, lane in enumerate(lanes):
            manager._sumo = SimpleNamespace(
                vehicle=SimpleNamespace(getLaneID=lambda _vid: lane))
            event = manager._sync_internal_lane(vehicle)
            assert vehicle.active_connector_id == physical
            assert vehicle.active_connector_from_lane_id == connector["from_lane"]
            assert vehicle.active_connector_to_lane_id == connector["to_lane"]
            if index == 0:
                assert event.details["connector_id"] == physical
                assert event.details["sumo_internal_lane_id"] == lane
                assert event.details["signal"] in ("green", "unsignalized")
            else:
                assert event is None  # A yield split is not a second entry.
        evaluator = DrivingEvaluator(manager, None, ["ego"])
        acc = evaluator._vehicles["ego"]
        evaluator._observe_signal_entry(acc, vehicle, time_s, 0.1)
        assert acc.red_light_entries == 0

    # Missing mapping must not fall back to the stale plan or "unsignalized".
    vehicle.active_connector_id = ""
    vehicle.planned_connector_id = planned
    manager._sumo = SimpleNamespace(
        vehicle=SimpleNamespace(getLaneID=lambda _vid: ":unknown_0"))
    with pytest.raises(RuntimeError, match="unmapped SUMO internal lane"):
        manager._sync_internal_lane(vehicle)
