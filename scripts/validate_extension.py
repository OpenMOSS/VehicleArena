#!/usr/bin/env python3
"""Load extension packages and validate their scenario/rule integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "vehiclearena"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension", action="append", default=[], metavar="IMPORT_PATH",
        help="Trusted extension package to import; repeat as needed.")
    parser.add_argument(
        "--scenario", type=Path,
        help="Optional scenario JSON to parse after loading extensions.")
    parser.add_argument(
        "--rules", type=Path,
        help="Optional rule directory to validate in isolation.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    from extensions import load_extensions

    reports = [report.to_dict()
               for report in load_extensions(args.extension)]
    output = {"status": "ok", "extensions": reports}

    if args.rules:
        from rules.rule_loader import RuleLoader
        loader = RuleLoader(args.rules)
        output["rules"] = {
            "directories": list(loader.rules_dirs),
            "environment_rules": len(loader.env_rules),
            "user_intent_rules": len(loader.user_intent_rules),
            "negative_checks": len(loader.negative_checks),
        }

    if args.scenario:
        from simulation.multi_sim_engine import MultiScenario
        payload = json.loads(args.scenario.read_text(encoding="utf-8"))
        scenario = MultiScenario.from_dict(payload)
        output["scenario"] = {
            "scenario_id": scenario.scenario_id,
            "vehicles": len(scenario.vehicles),
            "pedestrians": len(scenario.pedestrians),
            "physics_step_s": scenario.physics_step_s,
        }

    print(json.dumps(output, ensure_ascii=False, indent=2, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
