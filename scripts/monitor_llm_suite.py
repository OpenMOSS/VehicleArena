#!/usr/bin/env python3
"""Print live scene progress without interpreting stack dumps as failures."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import time

from run_llm_suite import active_workers, compatible_hashes, inspect_run, load_manifest, write_json, read_model_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    config = json.loads((root / "configuration.json").read_text())
    selected = set(config.get("selected_variants", []))
    suite_path = root / "suite-status.json"
    suite = json.loads(suite_path.read_text()) if suite_path.exists() else {}
    active = active_workers(root)
    report = {"updated_at_epoch_s": time.time(), "studies": {}, "running": [], "failures": []}
    for study in ("Basic", "MultiLLM"):
        manifest = load_manifest(root / "manifests" / f"{study}.manifest.json")
        counts = collections.Counter()
        for variant in manifest.variants:
            key = variant.variant_id
            if selected and key not in selected:
                continue
            if (study, key) in active:
                counts["running"] += 1
                worker = active[(study, key)]
                lines = Path(worker["log"]).read_text(errors="replace").splitlines()
                wakes = [line.strip() for line in lines if "Tick " in line and "t=" in line]
                calls = [row for row in read_model_trace(worker["log"])
                         if row.get("event") == "model_call"]
                errors = [line.strip() for line in lines
                          if "Error code:" in line or "AgentClientError:" in line]
                report["running"].append({
                    "variant": key, "pid": worker["pid"],
                    "last_wake": wakes[-1] if wakes else "starting",
                    "model_calls_traced": len(calls),
                    "model_time_s": round(sum(call["elapsed_s"] for call in calls), 1),
                    "last_call": calls[-1] if calls else None,
                    "errors": errors[-2:],
                })
                continue
            info = inspect_run(root / study / f"{key}.json.gz", compatible_hashes(manifest))
            recorded = suite.get("variants", {}).get(key, {})
            if info["status"] == "missing" and recorded.get("status") in {
                    "failed", "controller_error", "invalid_infrastructure"}:
                info = dict(recorded)
            status = "pending" if info["status"] == "missing" else info["status"]
            counts[status] += 1
            if status not in ("completed", "pending"):
                report["failures"].append({"variant": key, **info})
        report["studies"][study] = dict(counts)
    write_json(root / "live-status.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
