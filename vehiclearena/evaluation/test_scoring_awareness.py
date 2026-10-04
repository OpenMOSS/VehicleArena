"""Focused tests for evaluator-only physical traffic observations."""
import math
from types import SimpleNamespace

from evaluation.driving_evaluator import (
    DrivingEvaluationConfig, DrivingEvaluator,
)
from evaluation.driving_process_score import (
    DrivingProcessScoreConfig, _legitimate_stop,
)
from evaluation.scoring_awareness import ScoringTrafficView
from simulation.lane_level_runtime import pose_at
from simulation.road_networks import load_road_network
from simulation.traffic_manager import TrafficCoordinator
from simulation.vehicle_state import VehicleState


def _vehicle(vehicle_id, *, x, y, yaw, connector_id):
    return SimpleNamespace(
        vehicle_id=vehicle_id,
        active_connector_id=connector_id,
        pose_x_m=float(x), pose_y_m=float(y), yaw_rad=float(yaw),
        width_m=1.9, length_m=4.6, z_level=0,
        arrived=False, is_crashed=False,
    )


def _view(*vehicles):
    manager = SimpleNamespace(
        vehicles={vehicle.vehicle_id: vehicle for vehicle in vehicles},
        _lane_geometry=SimpleNamespace(_connector_by_id={
            "ego_path": {"node_id": "junction"},
            "merge_path": {"node_id": "junction"},
            "other_junction_path": {"node_id": "other_junction"},
        }),
    )
    return ScoringTrafficView(manager)


def test_same_direction_merging_connector_body_is_a_blocker():
    ego = _vehicle("ego", x=0, y=0, yaw=0, connector_id="ego_path")
    blocker = _vehicle(
        "blocker", x=9, y=.5, yaw=.1, connector_id="merge_path")

    observed = _view(ego, blocker)._same_direction_connector_blocker_ahead(
        ego, apply_perception=False)

    assert observed == {
        "vehicle_id": "blocker",
        "gap_m": 4.4,
        "other_connector_id": "merge_path",
        "relationship": "same_direction_connector_ahead",
    }


def test_unrelated_connector_traffic_does_not_justify_stopping():
    ego = _vehicle("ego", x=0, y=0, yaw=0, connector_id="ego_path")
    adjacent = _vehicle(
        "adjacent", x=8, y=3.5, yaw=0, connector_id="merge_path")
    oncoming = _vehicle(
        "oncoming", x=8, y=0, yaw=math.pi, connector_id="merge_path")
    other_junction = _vehicle(
        "elsewhere", x=8, y=0, yaw=0,
        connector_id="other_junction_path")

    assert _view(
        ego, adjacent, oncoming, other_junction,
    )._same_direction_connector_blocker_ahead(
        ego, apply_perception=False) is None


def test_real_merging_connectors_change_both_stop_evaluations():
    """Different connector IDs can merge into one physical lane corridor."""
    manager = TrafficCoordinator(load_road_network('xian_zhonglou'))
    for vehicle_id, index, progress in (
            ('ego', 8, .5), ('blocker', 0, .95)):
        connector_id = (
            f'connector::intersection::n11738459709::{index}')
        connector = manager._lane_geometry.connector_record(connector_id)
        lane = manager._lane_geometry._lane_by_id[connector['from_lane']]
        vehicle = VehicleState(
            vehicle_id=vehicle_id,
            current_node=lane['start_node'],
            current_segment=lane['segment_id'],
            current_lane_id=lane['id'], current_lane=lane['index'],
            edge_progress=progress, destination_node=lane['end_node'],
            is_llm=vehicle_id == 'ego',
        )
        vehicle.active_connector_id = connector_id
        vehicle.active_connector_to_lane_id = connector['to_lane']
        vehicle.pose_x_m, vehicle.pose_y_m, vehicle.yaw_rad = pose_at(
            connector['centerline_xy'],
            progress * manager._lane_geometry.connector_length(
                connector_id),
        )
        manager.vehicles[vehicle_id] = vehicle

    ego = manager.vehicles['ego']
    env = manager._build_env_view(ego, 0)
    view = ScoringTrafficView(manager)
    evaluator = DrivingEvaluator.__new__(DrivingEvaluator)
    evaluator.config = DrivingEvaluationConfig()

    awareness = view.get_driving_awareness('ego', ground_truth=True)
    assert awareness['leader'] is None  # The route IDs are different.
    assert awareness['connector_path_blocker']['vehicle_id'] == 'blocker'
    assert _legitimate_stop(
        ego, env, awareness, DrivingProcessScoreConfig())
    assert evaluator._has_legitimate_stop_reason(ego, env, awareness)

    del manager.vehicles['blocker']
    awareness = view.get_driving_awareness('ego', ground_truth=True)
    assert awareness['connector_path_blocker'] is None
    assert not _legitimate_stop(
        ego, env, awareness, DrivingProcessScoreConfig())
    assert not evaluator._has_legitimate_stop_reason(ego, env, awareness)
