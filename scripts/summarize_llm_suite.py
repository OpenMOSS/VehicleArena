#!/usr/bin/env python3
"""Audit suite completeness against frozen manifests, and aggregate results."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vehiclearena"), str(ROOT / "scripts")]

from evaluation.experiments.batch_runner import aggregate_run_directory
from evaluation.experiments.manifest import load_manifest
from run_llm_suite import compatible_hashes, inspect_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads((args.output / "configuration.json").read_text())
    selected = set(config.get("selected_variants", []))
    suite_path = args.output / "suite-status.json"
    suite = json.loads(suite_path.read_text()) if suite_path.exists() else {}
    summary = {"model": config["model"], "studies": {}, "issues": []}
    rows = []
    calls = collections.Counter()
    for study in ("Basic", "MultiLLM"):
        manifest = load_manifest(args.output / "manifests" / f"{study}.manifest.json")
        counts = collections.Counter()
        for variant in manifest.variants:
            if selected and variant.variant_id not in selected:
                continue
            path = args.output / study / f"{variant.variant_id}.json.gz"
            info = inspect_run(path, compatible_hashes(manifest))
            recorded = suite.get("variants", {}).get(variant.variant_id, {})
            if info["status"] == "missing" and recorded.get("status") in {
                    "failed", "controller_error", "invalid_infrastructure"}:
                info = dict(recorded)
            counts[info["status"]] += 1
            rows.append({"study": study, "variant": variant.variant_id,
                         "role_error_count": len(info.get("role_errors", [])),
                         **{key: value for key, value in info.items()
                            if key not in ("errors", "role_errors")}})
            if info["status"] != "completed":
                summary["issues"].append({"study": study, "variant": variant.variant_id, **info})
                continue
            with gzip.open(path, "rt") as stream:
                data = json.load(stream)
            for entity_id, metadata in data["agent_metadata"].items():
                if metadata["agent_type"] != "llm":
                    continue
                if metadata["model"] != config["model"]:
                    summary["issues"].append({"variant": variant.variant_id,
                        "entity": entity_id, "error": "wrong_model"})
                if study == "MultiLLM":
                    expected = variant.agent_overrides[entity_id]["driver_prompt"]
                    actual = data["agent_provenance"][entity_id]
                    if expected not in (actual.get("instruction") or ""):
                        # A body that crashed before its first wake legitimately
                        # has no built instruction; the configured role remains audited.
                        if actual.get("instruction") or metadata.get("driver_prompt") != expected:
                            summary["issues"].append({"variant": variant.variant_id,
                                "entity": entity_id, "error": "driver_role_mismatch"})
            for entity_calls in data.get("model_call_log", {}).values():
                for call in entity_calls:
                    calls[call.get("role", "unknown")] += 1
                    if call.get("model") != config["model"]:
                        summary["issues"].append({"variant": variant.variant_id,
                            "error": "wrong_call_model", "model": call.get("model")})
        summary["studies"][study] = {"expected": sum(counts.values()), "counts": dict(counts)}
        if counts["completed"]:
            aggregate_run_directory(args.output / study)
    summary["model_calls_by_role"] = dict(calls)
    summary["complete"] = not summary["issues"]
    summary["completed_scenes"] = sum(row["status"] == "completed" for row in rows)
    summary["total_scenes"] = len(rows)
    summary["prompt_tokens"] = sum(row.get("prompt_tokens", 0) for row in rows)
    summary["completion_tokens"] = sum(row.get("completion_tokens", 0) for row in rows)
    summary["role_error_count"] = sum(row.get("role_error_count", 0) for row in rows)
    write_json(args.output / "audit.json", summary)
    columns = sorted({key for row in rows for key in row})
    with (args.output / "scenes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "issues"},
                     ensure_ascii=False, indent=2))
    print("issues:", len(summary["issues"]))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
