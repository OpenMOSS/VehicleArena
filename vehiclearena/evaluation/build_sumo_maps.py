"""Precompile and validate VehicleArena lane maps for the SUMO engine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from simulation.sumo_map import SumoMapConverter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "network_ids", nargs="*",
        help="network IDs to compile; omit to compile every installed map",
    )
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    road_dir = Path(__file__).resolve().parents[1] / "simulation" / "road_networks"
    selected = set(args.network_ids)
    paths = sorted(road_dir.glob("*_lane_level.json"))
    if selected:
        paths = [
            path for path in paths
            if path.name.removesuffix("_lane_level.json") in selected
        ]
        missing = selected - {
            path.name.removesuffix("_lane_level.json") for path in paths
        }
        if missing:
            raise SystemExit(
                f"unknown or unavailable network IDs: {sorted(missing)}")

    converter = SumoMapConverter()
    results = []
    failed = False
    for path in paths:
        started = time.monotonic()
        try:
            bundle = converter.convert(
                path, cache_root=args.cache_root or None, force=args.force)
            manifest = json.loads(Path(
                bundle.manifest_file).read_text(encoding="utf-8"))
            item = {
                "network_id": path.name.removesuffix("_lane_level.json"),
                "status": "passed",
                "elapsed_s": round(time.monotonic() - started, 3),
                "counts": manifest["counts"],
                "manifest": bundle.manifest_file,
            }
        except Exception as exc:
            failed = True
            item = {
                "network_id": path.name.removesuffix("_lane_level.json"),
                "status": "failed",
                "elapsed_s": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False))

    report = {
        "schema": "vehiclearena-sumo-map-build-report-v1",
        "passed": sum(item["status"] == "passed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
