#!/usr/bin/env python3
"""Exercise LLM-authority physics startup without paying for model calls.

This is a diagnostic only: no maneuver is submitted, so an infeasible hold can
be detected even when a real driver might resolve it at its initial wake.
"""
import copy
import argparse
import gc
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vehiclearena"), str(ROOT / "scripts")]
from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from run_llm_suite import write_json


def run_physics_preflight(raw):
    scenario = MultiScenario.from_dict(copy.deepcopy(raw))
    scenario.total_time_s = 0.2
    scenario.physics_only_mode = True
    engine = MultiSimEngine(scenario)
    # Registration, not agent_type alone, activates llm_maneuver authority.
    # physics_only_mode suppresses invocation but must retain registration.
    callbacks = {v.vehicle_id: lambda *args, **kwargs: []
                 for v in scenario.vehicles if v.agent_type == "llm"}
    engine.run(callbacks)
    authorities = {vid: engine.traffic_mgr.get_state(vid).route_control_authority
                   for vid in callbacks}
    if any(value != "llm_maneuver" for value in authorities.values()):
        raise RuntimeError(f"Incorrect preflight actuator authority: {authorities}")
    return {"status": "passed", "authorities": authorities, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory containing frozen manifests")
    parser.add_argument("--report", type=Path, help="Separate report path for a new code revision")
    args = parser.parse_args()
    output = args.output
    report = {}
    for study in ("Basic", "MultiLLM"):
        manifest = json.loads((output / "manifests" / f"{study}.manifest.json").read_text())
        for variant in manifest["variants"]:
            raw = copy.deepcopy(variant["scenario"])
            try:
                result = run_physics_preflight(raw)
            except Exception as exc:
                result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()}
                print(variant["variant_id"], result["error"], flush=True)
            finally:
                gc.collect()
            report[variant["variant_id"]] = result
            write_json(args.report or output / "physics-preflight.json", report)
    print(json.dumps({"total": len(report),
                      "failed": sum(v["status"] == "failed" for v in report.values())}), flush=True)


if __name__ == "__main__":
    main()
