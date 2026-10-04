from types import SimpleNamespace

from evaluation.experiments.reference_protocol import (
    build_reference_protocol, protocol_hash,
)


def _spec(**overrides):
    values = {
        "agent_type": "llm",
        "model": "fixed-peer",
        "api_base": "https://endpoint-a.invalid/v1",
        "api_key": "secret-a",
        "max_turns": 10,
        "temperature": 0.7,
        "max_tokens": 4096,
        "thinking_mode": "disabled",
        "reasoning_effort": None,
        "chat_template_enable_thinking": False,
        "context_window_tokens": 1000000,
        "todo_max_ttl_s": 3600.0,
        "heartbeat_interval_s": 3.0,
        "driver_prompt": "fixed role",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _variant():
    return {
        "variant_id": "multi_001",
        "scenario_hash": "frozen-scenario",
        "scenario": {
            "road_network_id": "test_map",
            "experiment_scene": {
                "focal_vehicle_id": "ego",
                "fixed_peer_vehicle_ids": ["peer"],
            },
        },
    }


def _physics(map_manifest, **overrides):
    values = {
        "name": "sumo",
        "sumo_version": "SUMO 1.25.0",
        "step_length_s": 0.1,
        "vehicle_dynamics_authority": "sumo",
        "vehicle_pose_authority": "sumo",
        "vehicle_route_execution_authority": "sumo",
        "vehicle_collision_authority": "sumo",
        "vehicle_pedestrian_collision_authority": "sumo",
        "pedestrian_physics_authority": "sumo",
        "map_manifest": str(map_manifest),
    }
    values.update(overrides)
    return values


def test_protocol_ignores_endpoint_credentials_but_tracks_peer_behavior(
        tmp_path):
    network = tmp_path / "test_map_lane_level.json"
    network.write_text('{"schema":"test"}')
    map_manifest = tmp_path / "manifest.json"
    map_manifest.write_text(
        '{"format":"vehiclearena-sumo-map-v1","source_sha256":"compiled"}')
    first = build_reference_protocol(
        experiment_id="MultiLLM", variant=_variant(),
        network_path=network, physics_engine=_physics(map_manifest),
        resolved_specs={"peer": _spec()})
    moved_endpoint = build_reference_protocol(
        experiment_id="MultiLLM", variant=_variant(),
        network_path=network, physics_engine=_physics(map_manifest),
        resolved_specs={"peer": _spec(
            api_base="https://endpoint-b.invalid/v1", api_key="secret-b")})
    changed_model = build_reference_protocol(
        experiment_id="MultiLLM", variant=_variant(),
        network_path=network, physics_engine=_physics(map_manifest),
        resolved_specs={"peer": _spec(model="different-peer")})

    assert protocol_hash(first) == protocol_hash(moved_endpoint)
    assert protocol_hash(first) != protocol_hash(changed_model)
    assert first["reference_policy"] == "focal_sumo_fixed_peers"


def test_protocol_tracks_map_and_sumo_version(tmp_path):
    network = tmp_path / "test_map_lane_level.json"
    network.write_text('{"schema":"test"}')
    map_manifest = tmp_path / "manifest.json"
    map_manifest.write_text(
        '{"format":"vehiclearena-sumo-map-v1","source_sha256":"compiled"}')
    initial = build_reference_protocol(
        experiment_id="MultiLLM", variant=_variant(),
        network_path=network, physics_engine=_physics(map_manifest),
        resolved_specs={"peer": _spec()})
    newer_sumo = build_reference_protocol(
        experiment_id="MultiLLM", variant=_variant(),
        network_path=network,
        physics_engine=_physics(
            map_manifest, sumo_version="SUMO 1.26.0"),
        resolved_specs={"peer": _spec()})
    map_manifest.write_text(
        '{"format":"vehiclearena-sumo-map-v1",'
        '"source_sha256":"new-compiler-revision"}')
    changed_compiler = build_reference_protocol(
        experiment_id="MultiLLM", variant=_variant(),
        network_path=network, physics_engine=_physics(map_manifest),
        resolved_specs={"peer": _spec()})
    map_manifest.write_text(
        '{"format":"vehiclearena-sumo-map-v1",'
        '"source_sha256":"compiled"}')
    network.write_text('{"schema":"changed"}')
    changed_map = build_reference_protocol(
        experiment_id="MultiLLM", variant=_variant(),
        network_path=network, physics_engine=_physics(map_manifest),
        resolved_specs={"peer": _spec()})

    assert protocol_hash(initial) != protocol_hash(newer_sumo)
    assert protocol_hash(initial) != protocol_hash(changed_compiler)
    assert protocol_hash(initial) != protocol_hash(changed_map)
