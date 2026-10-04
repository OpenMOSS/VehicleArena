#!/usr/bin/env python3
"""Tally LLM input/output tokens across VehicleArena run outputs.

Sums provider-reported prompt_tokens / completion_tokens from every
variant result ({variant_id}.json.gz) under each run directory, split by
role (vehicle_agent / personal_agent / passenger_judge).

Usage:
    python scripts/tally_run_tokens.py RUN_DIR [RUN_DIR ...]
"""

import gzip
import json
import sys
from pathlib import Path


def tally_file(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    roles: dict[str, dict[str, int]] = {}
    for entity_calls in (data.get("model_call_log") or {}).values():
        for call in entity_calls:
            role = str(call.get("role") or "unknown")
            bucket = roles.setdefault(
                role, {"calls": 0, "prompt_tokens": 0,
                       "completion_tokens": 0})
            bucket["calls"] += 1
            bucket["prompt_tokens"] += int(call.get("prompt_tokens", 0) or 0)
            bucket["completion_tokens"] += int(
                call.get("completion_tokens", 0) or 0)
    return roles


def tally_dir(run_dir: Path) -> dict:
    totals: dict[str, dict[str, int]] = {}
    variants = 0
    for path in sorted(run_dir.glob("*.json.gz")):
        if path.name == "status.json":
            continue
        variants += 1
        for role, bucket in tally_file(path).items():
            acc = totals.setdefault(
                role, {"calls": 0, "prompt_tokens": 0,
                       "completion_tokens": 0})
            for key in acc:
                acc[key] += bucket[key]
    status_path = run_dir / "status.json"
    status_counts = {}
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        for item in (status.get("variants") or {}).values():
            key = str(item.get("status"))
            status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "run_dir": str(run_dir),
        "result_files": variants,
        "status_counts": status_counts,
        "roles": totals,
    }


def print_report(report: dict) -> None:
    print(f"\n== {report['run_dir']} ==")
    print(f"result files: {report['result_files']}  "
          f"status: {report['status_counts'] or 'n/a'}")
    dir_prompt = dir_completion = dir_calls = 0
    for role in sorted(report["roles"]):
        bucket = report["roles"][role]
        dir_calls += bucket["calls"]
        dir_prompt += bucket["prompt_tokens"]
        dir_completion += bucket["completion_tokens"]
        print(f"  {role:<18} calls {bucket['calls']:>7}  "
              f"input {bucket['prompt_tokens']:>12,}  "
              f"output {bucket['completion_tokens']:>10,}")
    print(f"  {'TOTAL':<18} calls {dir_calls:>7}  "
          f"input {dir_prompt:>12,}  "
          f"output {dir_completion:>10,}")
    report["dir_prompt_tokens"] = dir_prompt
    report["dir_completion_tokens"] = dir_completion
    report["dir_calls"] = dir_calls


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    grand = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    reports = []
    for arg in argv:
        run_dir = Path(arg)
        if not run_dir.is_dir():
            print(f"skip (not a directory): {run_dir}", file=sys.stderr)
            continue
        report = tally_dir(run_dir)
        reports.append(report)
        print_report(report)
    if len(reports) > 1:
        for report in reports:
            grand["calls"] += report["dir_calls"]
            grand["prompt_tokens"] += report["dir_prompt_tokens"]
            grand["completion_tokens"] += report["dir_completion_tokens"]
        print("\n== GRAND TOTAL ==")
        print(f"  calls {grand['calls']:>7}  "
              f"input {grand['prompt_tokens']:>12,}  "
              f"output {grand['completion_tokens']:>10,}")
    out_path = Path(argv[-1]) / "token_tally.json"
    out_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
