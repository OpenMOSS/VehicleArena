#!/usr/bin/env python3
"""Boot every frozen task without model calls or rendering; keep failures visible."""
import argparse
import gc
import json
from pathlib import Path

from evaluation.experiments.scene_catalog import iter_catalog_scenes
from simulation.multi_sim_engine import MultiScenario, MultiSimEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    catalog = root / "vehiclearena/evaluation/experiments/scenarios"
    rows = []
    for entry, scenario_path, expected_path in iter_catalog_scenes(catalog):
        engine = None
        row = {"scene_id": entry["scene_id"]}
        try:
            scene = json.loads(scenario_path.read_text())
            expected = json.loads(expected_path.read_text())
            scene.update(total_time_s=args.duration, physics_only_mode=True,
                         enable_driving_evaluation=False)
            engine = MultiSimEngine(MultiScenario.from_dict(scene))
            engine.run({})
            assert not engine.agent_callback_errors, engine.agent_callback_errors
            early = [e for e in engine.traffic_mgr.collision_log
                     if e.time_s <= scene.get("physics_step_s", 0.1) + 1e-9]
            smoke = expected.get("smoke", {})
            if early and not smoke.get("allow_initial_collision", False):
                raise AssertionError("unexpected initialization collision")
            assert len(early) >= smoke.get("minimum_initial_collisions", 0)
            row["passed"] = True
        except Exception as error:
            row.update(passed=False, error=f"{type(error).__name__}: {error}")
        finally:
            if engine is not None and getattr(engine, "traffic_mgr", None):
                engine.traffic_mgr.close()
            engine = None
            gc.collect()
        rows.append(row)
        if not row["passed"] or len(rows) % 10 == 0:
            print(json.dumps({"checked": len(rows), "latest": row},
                             ensure_ascii=False), flush=True)
    report = {"duration_s": args.duration, "model_calls": 0,
              "passed": sum(row["passed"] for row in rows),
              "failed": sum(not row["passed"] for row in rows), "tasks": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tasks"}), flush=True)
    raise SystemExit(bool(report["failed"]))


if __name__ == "__main__":
    main()
