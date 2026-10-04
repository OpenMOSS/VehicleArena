#!/usr/bin/env python3
"""Run one scenario as a SUMO reference and stream it to Web3D."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.request
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = WEB_ROOT.parent
PACKAGE_ROOT = REPOSITORY_ROOT / "vehiclearena"
for import_root in (REPOSITORY_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from visualization.web3d_live import Web3DFramePublisher


def _sumo_reference(payload: dict, duration_s: float | None) -> dict:
    result = copy.deepcopy(payload)
    for entity_group in ("vehicles", "pedestrians"):
        for entity in result.get(entity_group, []):
            entity.setdefault("agent_config", {})["type"] = "sumo"
    result["physics_only_mode"] = True
    if duration_s is not None:
        result["total_time_s"] = float(duration_s)
        result["stop_when_all_vehicles_terminal"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream an authoritative SUMO reference episode to Web3D. "
            "Use the experiment CLI when the driving agents must call LLMs."))
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument(
        "--stream-url",
        default="http://127.0.0.1:8765/api/live/frame")
    parser.add_argument("--session-id", default="live")
    parser.add_argument("--focus-entity", default="ego")
    parser.add_argument("--junction-id")
    parser.add_argument("--radius-m", type=float, default=220.0)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument(
        "--fast", action="store_true",
        help="Do not pace SUMO to real time; useful for transport tests.")
    args = parser.parse_args()

    health_url = args.stream_url.split("/api/", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=2.0) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
    except Exception as error:
        raise SystemExit(
            f"Web3D server is unavailable at {health_url}: {error}")

    payload = json.loads(args.scenario.read_text(encoding="utf-8"))
    scenario = MultiScenario.from_dict(
        _sumo_reference(payload, args.duration_s))
    publisher = Web3DFramePublisher(
        args.stream_url,
        session_id=args.session_id,
        focus_entity_id=args.focus_entity,
        junction_id=args.junction_id,
        radius_m=args.radius_m,
        realtime=not args.fast,
    )
    engine = MultiSimEngine(scenario)
    engine.add_world_observer(publisher)
    print(
        "Open: http://127.0.0.1:8765/"
        f"?session_id={args.session_id}&view=cockpit")
    result = engine.run({})
    print(json.dumps({
        "scenario_id": result.scenario_id,
        "physics_frames": publisher.encoder.frame_count,
        "published_frames": publisher.published_count,
        "dropped_frames": publisher.dropped_count,
        "failed_frames": publisher.failed_count,
        "last_error": publisher.last_error or None,
        "observer_errors": engine.world_observer_errors,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
