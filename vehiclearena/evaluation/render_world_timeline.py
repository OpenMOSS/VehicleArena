"""CLI for generic lane-world PNG snapshots and arbitrary-interval GIFs.

Vehicle JSON example:
[
  {"id": "car-1", "start": "n33399858", "destination": "n35722739"}
]
"""

from __future__ import annotations

import argparse
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

from simulation.road_networks import load_road_network
from simulation.sumo_traffic_manager import SumoTrafficManager
from visualization.lane_world_renderer import (
    LaneWorldRenderer, Viewport, WorldRecording)
from visualization.sumo_native_renderer import SumoNativeRenderer


DEFAULT_SUMO_GUI_SETTINGS = os.path.join(
    ROOT, "visualization", "sumo_vehiclearena.view.xml")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="beijing_guomao")
    parser.add_argument("--vehicles", help="JSON file containing vehicle list")
    parser.add_argument("--output", required=True, help=".png or .gif")
    parser.add_argument(
        "--renderer", choices=("vehiclearena", "sumo"),
        default="vehiclearena",
        help="drawing backend; sumo captures the live sumo-gui instance")
    parser.add_argument("--recording-json", help="save/load-independent frames")
    parser.add_argument("--replay-json", help="render an existing recording")
    parser.add_argument("--at", type=float, help="PNG snapshot simulation time")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--frame-interval", type=float, default=0.2)
    parser.add_argument("--center-node")
    parser.add_argument("--radius", type=float, default=150.0)
    parser.add_argument("--full-map", action="store_true")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--labels", action="store_true")
    parser.add_argument(
        "--sumo-cache-root", default="",
        help="optional directory for compiled SUMO network caches")
    parser.add_argument(
        "--sumo-gui-settings", default=DEFAULT_SUMO_GUI_SETTINGS,
        help="SUMO-GUI view-settings XML used by the native renderer")
    parser.add_argument(
        "--sumo-schema", default="vehiclearena",
        help="SUMO-GUI visualization scheme selected for native rendering")
    parser.add_argument(
        "--no-auto-xvfb", action="store_true",
        help="require an existing DISPLAY instead of starting Xvfb")
    parser.add_argument(
        "--vehicle-marker-radius", type=int, default=0,
        help="optional visibility halo in pixels; collision body stays exact")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.renderer == "sumo" and args.replay_json:
        raise ValueError("SUMO native rendering cannot replay VehicleArena JSON")
    if args.renderer == "sumo" and args.recording_json:
        raise ValueError(
            "--recording-json is available only with --renderer vehiclearena")
    if args.renderer == "sumo" and args.labels:
        raise ValueError("--labels is available only with --renderer vehiclearena")
    if not args.replay_json and not args.vehicles:
        raise ValueError("--vehicles is required unless --replay-json is used")
    vehicle_configs = []
    if args.vehicles:
        with open(args.vehicles, encoding="utf-8") as stream:
            vehicle_configs = json.load(stream)
        if (isinstance(vehicle_configs, dict)
                and "vehicles_detail" in vehicle_configs):
            vehicle_configs = [
                {
                    "id": vehicle_id,
                    "start": item["start_node"],
                    "destination": item["destination_node"],
                }
                for vehicle_id, item
                in vehicle_configs["vehicles_detail"].items()
            ]
    network = load_road_network(args.network)
    manager = SumoTrafficManager(
        network,
        sumo_config={
            "cache_root": args.sumo_cache_root,
            "gui": args.renderer == "sumo",
            "auto_start_xvfb": not args.no_auto_xvfb,
            "gui_width": args.width,
            "gui_height": args.height,
            "gui_settings_file": (
                args.sumo_gui_settings if args.renderer == "sumo" else ""),
            "gui_schema": (
                args.sumo_schema if args.renderer == "sumo" else ""),
        },
    )
    lane_map = manager._lane_geometry.data
    if args.full_map or not args.center_node:
        viewport = Viewport.full_map(lane_map)
    else:
        viewport = Viewport.around_node(
            lane_map, args.center_node, args.radius)
    renderer = (
        SumoNativeRenderer(
            manager, viewport, width=args.width, height=args.height)
        if args.renderer == "sumo"
        else LaneWorldRenderer(
            manager, viewport, width=args.width, height=args.height,
            vehicle_marker_radius_px=args.vehicle_marker_radius)
    )

    if args.replay_json:
        recording = WorldRecording.load_json(args.replay_json)
        renderer.set_viewport(recording.viewport)
        if not args.output.lower().endswith(".gif"):
            raise ValueError("replay output must be .gif")
        renderer.render_gif(recording, args.output, show_labels=args.labels)
        print(json.dumps({
            "output": os.path.abspath(args.output),
            "source": os.path.abspath(args.replay_json),
            "frames": len(recording.frames),
        }, ensure_ascii=False, indent=2))
        return

    try:
        for index, config in enumerate(vehicle_configs):
            manager.register_vehicle(
                config.get("id", f"vehicle-{index + 1}"),
                config["start"], config["destination"],
            )
    except Exception:
        manager.close()
        raise

    extension = os.path.splitext(args.output)[1].lower()
    try:
        if args.renderer == "sumo":
            if extension == ".png":
                at = args.at if args.at is not None else args.start
                result = renderer.render_png_at(at, args.output)
            elif extension == ".gif":
                if args.end is None:
                    raise ValueError("--end is required for GIF output")
                result = renderer.render_gif(
                    args.start, args.end, args.frame_interval, args.output)
            else:
                raise ValueError("--output extension must be .png or .gif")
            report = {
                "output": result.output,
                "recording_json": None,
                "network": args.network,
                "vehicles": len(vehicle_configs),
                "render_backend": "sumo-gui",
                "physics_engine": "sumo",
                "start_time_s": result.start_time_s,
                "end_time_s": result.end_time_s,
                "frames": result.frame_count,
                "viewport": as_viewport_dict(viewport),
            }
        else:
            if extension == ".png":
                at = args.at if args.at is not None else args.start
                recording = renderer.simulate_and_record(at, at)
                renderer.render_png(
                    recording.frames[0], args.output,
                    show_labels=args.labels)
            elif extension == ".gif":
                if args.end is None:
                    raise ValueError("--end is required for GIF output")
                recording = renderer.simulate_and_record(
                    args.start, args.end, args.frame_interval)
                renderer.render_gif(
                    recording, args.output, show_labels=args.labels)
            else:
                raise ValueError("--output extension must be .png or .gif")
            if args.recording_json:
                recording.save_json(args.recording_json)
            report = {
                "output": os.path.abspath(args.output),
                "recording_json": (
                    os.path.abspath(args.recording_json)
                    if args.recording_json else None),
                "network": args.network,
                "vehicles": len(vehicle_configs),
                "render_backend": "vehiclearena",
                "physics_engine": "sumo",
                "start_time_s": recording.frames[0].time_s,
                "end_time_s": recording.frames[-1].time_s,
                "frames": len(recording.frames),
                "viewport": as_viewport_dict(recording.viewport),
            }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        manager.close()


def as_viewport_dict(viewport):
    return {
        "min_x": viewport.min_x, "min_y": viewport.min_y,
        "max_x": viewport.max_x, "max_y": viewport.max_y,
    }


if __name__ == "__main__":
    main()
