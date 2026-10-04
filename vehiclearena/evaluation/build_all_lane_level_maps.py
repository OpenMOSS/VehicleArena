"""Build and validate lane-level companions for every source road network."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from simulation.lane_level_map import LaneLevelMapBuilder  # noqa: E402
from evaluation.build_lane_level_prototype import validate  # noqa: E402

NETWORK_DIR = ROOT / "simulation" / "road_networks"
DEFAULT_REPORT = HERE / "outputs" / "all_lane_level_maps_report.json"


def source_maps() -> list[Path]:
    return sorted(
        path for path in NETWORK_DIR.glob("*.json")
        if not path.stem.endswith("_lane_level"))


def build_one(source_path: Path) -> dict:
    started = time.monotonic()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    data = LaneLevelMapBuilder().build(source, source_path.stem)
    errors = validate(data)
    output_path = source_path.with_name(
        f"{source_path.stem}_lane_level.json")
    result = {
        "network_id": source_path.stem,
        "source": str(source_path),
        "output": str(output_path),
        "status": "PASS" if not errors else "FAIL",
        "validation_errors": errors,
        "lanes": len(data["lanes"]),
        "connectors": len(data["connectors"]),
        "connector_conflicts": len(data["connector_conflicts"]),
        "stop_lines": len(data["stop_lines"]),
        "crosswalks": len(data["crosswalks"]),
        "pedestrian_approaches": len(data["pedestrian_approaches"]),
        "signal_plans": len(data["signal_plans"]),
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    if not errors:
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8")
        result["bytes"] = output_path.stat().st_size
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network", action="append",
        help="build only this network ID; may be repeated")
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    selected = set(args.network or [])
    sources = [
        path for path in source_maps()
        if not selected or path.stem in selected]
    missing = selected - {path.stem for path in sources}
    if missing:
        raise SystemExit(f"unknown network IDs: {sorted(missing)}")

    started = time.monotonic()
    results = []
    for index, source_path in enumerate(sources, 1):
        try:
            result = build_one(source_path)
        except Exception as exc:
            result = {
                "network_id": source_path.stem,
                "source": str(source_path),
                "status": "ERROR",
                "validation_errors": [f"{type(exc).__name__}: {exc}"],
            }
        results.append(result)
        print(
            f"[{index:03d}/{len(sources):03d}] "
            f"{source_path.stem}: {result['status']} "
            f"lanes={result.get('lanes', 0)} "
            f"crosswalks={result.get('crosswalks', 0)}",
            flush=True)

    report = {
        "schema": "vehiclearena-lane-level-batch-report-v2",
        "source_maps": len(sources),
        "passed": sum(item["status"] == "PASS" for item in results),
        "failed": sum(item["status"] != "PASS" for item in results),
        "total_lanes": sum(item.get("lanes", 0) for item in results),
        "total_connectors": sum(
            item.get("connectors", 0) for item in results),
        "total_connector_conflicts": sum(
            item.get("connector_conflicts", 0) for item in results),
        "total_stop_lines": sum(
            item.get("stop_lines", 0) for item in results),
        "total_crosswalks": sum(
            item.get("crosswalks", 0) for item in results),
        "total_pedestrian_approaches": sum(
            item.get("pedestrian_approaches", 0) for item in results),
        "total_signal_plans": sum(
            item.get("signal_plans", 0) for item in results),
        "total_bytes": sum(item.get("bytes", 0) for item in results),
        "elapsed_s": round(time.monotonic() - started, 3),
        "maps": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(
        {key: value for key, value in report.items() if key != "maps"},
        ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["failed"] else 0)


if __name__ == "__main__":
    main()
