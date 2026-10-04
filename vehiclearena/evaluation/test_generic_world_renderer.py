"""Smoke test generic current-frame, interval GIF and JSON replay rendering."""

from __future__ import annotations

import json
import os
import sys
import types

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
simulation_package = types.ModuleType("simulation")
simulation_package.__path__ = [os.path.join(ROOT, "simulation")]
sys.modules["simulation"] = simulation_package

from simulation.road_networks import get_key_node, load_road_network
from simulation.sumo_traffic_manager import SumoTrafficManager
from visualization.lane_world_renderer import (
    LaneWorldRenderer, Viewport, WorldRecording)

OUTPUT_DIR = os.path.join(ROOT, "evaluation", "outputs")


def main():
    network = load_road_network("beijing_guomao")
    manager = SumoTrafficManager(network)
    start = get_key_node("beijing_guomao", "guanghua_road")
    destination = get_key_node("beijing_guomao", "chaoyang_road")
    for index in range(3):
        vehicle = manager.register_vehicle(
            f"generic-{index + 1}", start, destination,
        )
        vehicle.edge_progress = 0.08 + index * 0.08

    viewport = Viewport.around_node(
        manager._lane_geometry.data, start, 180.0)
    renderer = LaneWorldRenderer(
        manager, viewport=viewport, width=480, height=480)
    recording = renderer.simulate_and_record(0.0, 3.0, 0.2)
    png_path = os.path.join(OUTPUT_DIR, "generic_snapshot_t2.png")
    gif_path = os.path.join(OUTPUT_DIR, "generic_interval_0_3.gif")
    json_path = os.path.join(OUTPUT_DIR, "generic_interval_0_3.json")
    replay_path = os.path.join(OUTPUT_DIR, "generic_interval_replay.gif")
    renderer.render_png(recording.frames[10], png_path)
    renderer.render_gif(recording, gif_path)
    recording.save_json(json_path)
    loaded = WorldRecording.load_json(json_path)
    renderer.render_gif(loaded, replay_path)

    assert len(recording.frames) == 16
    assert len(loaded.frames) == len(recording.frames)
    assert loaded.frames[10].vehicles == recording.frames[10].vehicles
    for path in (png_path, gif_path, json_path, replay_path):
        assert os.path.getsize(path) > 0
    print(json.dumps({
        "status": "PASS",
        "snapshot": png_path,
        "gif": gif_path,
        "recording": json_path,
        "replay_gif": replay_path,
        "frames": len(recording.frames),
        "time_range_s": [
            recording.frames[0].time_s,
            recording.frames[-1].time_s],
        "vehicles_per_frame": [
            len(frame.vehicles) for frame in recording.frames],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
