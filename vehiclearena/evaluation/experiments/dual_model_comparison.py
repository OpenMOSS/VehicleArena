"""Compare two complete LLM experiment roots scene by scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from evaluation.experiments.batch_runner import (
    aggregate_run_directory, load_run,
)
from evaluation.experiments.reference_protocol import (
    reference_compatibility,
)


_SYSTEM_METRICS = (
    "arrival_rate", "throughput_per_min", "collision_count",
    "secondary_collision_count", "wait_mean_s", "wait_p90_s",
    "max_queue_length", "candidate_deadlock_count",
    "confirmed_deadlock_count", "queue_wait_mean_s",
)


def _mean(values: list[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 2)


def _score_text(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _rate_text(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def _model_usage(payload: dict) -> dict:
    calls = [
        item
        for items in payload.get("model_call_log", {}).values()
        for item in items
    ]
    return {
        "call_count": len(calls),
        "successful_call_count": sum(
            bool(item.get("ok", False)) for item in calls),
        "prompt_tokens": sum(
            int(item.get("prompt_tokens", 0) or 0) for item in calls),
        "completion_tokens": sum(
            int(item.get("completion_tokens", 0) or 0) for item in calls),
    }


def _raw_outcomes(payload: dict) -> dict:
    evaluated = [
        vehicle for vehicle in payload.get("result", {}).get(
            "vehicles", {}).values()
        if vehicle.get("is_evaluated")
    ]
    reports = [
        vehicle.get("driving_evaluation") or {}
        for vehicle in evaluated
    ]
    reasonable_flags = [
        item.get("reasonable_driving_pass")
        for item in reports
        if item.get("reasonable_driving_pass") is not None
    ]
    metadata = payload.get("agent_metadata", {})
    actual_models = sorted({
        str(item.get("model"))
        for item in metadata.values()
        if item.get("agent_type") == "llm" and item.get("model")
    })
    return {
        "run_status": payload.get("run", {}).get("status", "unknown"),
        "infrastructure_valid": (
            payload.get("run", {}).get("status") == "completed"
            and not payload.get("agent_infrastructure_errors")
        ),
        "actual_models": actual_models,
        "evaluated_entity_count": len(evaluated),
        "task_completed_rate": (
            sum(bool(item.get("task_completed")) for item in reports)
            / len(reports) if reports else None
        ),
        "hard_safety_passed": all(
            bool(item.get("hard_safety_passed")) for item in reports
        ) if reports else None,
        "reasonable_driving_vehicle_rate": (
            sum(bool(value) for value in reasonable_flags)
            / len(reasonable_flags)
            if reasonable_flags else None
        ),
        "usage": _model_usage(payload),
    }


def _experiment_directories(root: Path) -> list[Path]:
    root = Path(root)
    if any(root.glob("*.json.gz")):
        return [root]
    return [
        path for path in (root / "Basic", root / "MultiLLM")
        if path.is_dir()
    ]


def _collect_root(root: Path) -> dict[str, dict]:
    collected = {}
    for experiment_dir in _experiment_directories(root):
        aggregate = aggregate_run_directory(experiment_dir)
        for row in aggregate.get("rows", []):
            if not row.get("requires_llm"):
                continue
            variant_id = row["variant_id"]
            raw_path = experiment_dir / f"{variant_id}.json.gz"
            payload = load_run(raw_path)
            collected[variant_id] = {
                "experiment_id": row["experiment_id"],
                "variant_id": variant_id,
                "base_scenario_id": row["base_scenario_id"],
                "scores_100": {
                    "cabin": row.get("cabin_layer_score_100"),
                    "single_vehicle": row.get(
                        "single_vehicle_layer_score_100"),
                },
                "system_metrics": {
                    key: row.get(key)
                    for key in _SYSTEM_METRICS
                },
                "trajectory_quality_score_100": (
                    round(100.0 * float(row["trajectory_quality_score"]), 2)
                    if row.get("trajectory_quality_score") is not None
                    else None
                ),
                "reference_identity": {
                    "protocol_hash": payload.get("protocol_hash"),
                    "reference_protocol": payload.get(
                        "reference_protocol"),
                    "source_hash": payload.get("source_hash"),
                    "manifest_hash": payload.get("manifest_hash"),
                    "variant": {
                        "scenario_hash": payload.get(
                            "variant", {}).get("scenario_hash"),
                    },
                },
                **_raw_outcomes(payload),
            }
    return collected


def _expected_variants(manifest_root: Optional[Path]) -> dict[str, dict]:
    expected = {}
    if manifest_root is None:
        return expected
    for path in sorted(manifest_root.glob("*.manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for variant in manifest.get("variants", []):
            if not variant.get("requires_llm"):
                continue
            expected[variant["variant_id"]] = {
                "experiment_id": manifest["experiment_id"],
                "base_scenario_id": variant["base_scenario_id"],
            }
    return expected


def _attach_reference_metrics(
        items: dict[str, dict], references: dict[str, dict]):
    for variant_id, item in items.items():
        reference = references.get(variant_id)
        if reference is None or not reference.get("infrastructure_valid"):
            continue
        compatibility = reference_compatibility(
            item["reference_identity"], reference["reference_identity"])
        item["reference_pairing"] = compatibility
        if not compatibility["applicable"]:
            continue
        item["system_delta_vs_sumo"] = {
            key: round(
                float(item["system_metrics"].get(key) or 0.0)
                - float(reference["system_metrics"].get(key) or 0.0), 6)
            for key in _SYSTEM_METRICS
        }


def _summary(items: list[dict]) -> dict:
    usage = [item["usage"] for item in items]
    score_applicable_counts = {
        layer: sum(
            item["scores_100"].get(layer) is not None
            for item in items)
        for layer in ("cabin", "single_vehicle")
    }
    return {
        "scene_count": len(items),
        "infrastructure_valid_count": sum(
            bool(item["infrastructure_valid"]) for item in items),
        "hard_safety_pass_count": sum(
            item["hard_safety_passed"] is True for item in items),
        "mean_task_completed_rate": _mean([
            item["task_completed_rate"] for item in items]),
        "mean_reasonable_driving_vehicle_rate": _mean([
            item["reasonable_driving_vehicle_rate"] for item in items]),
        "mean_scores_100": {
            layer: _mean([
                item["scores_100"].get(layer) for item in items
            ])
            for layer in ("cabin", "single_vehicle")
        },
        "score_applicable_counts": score_applicable_counts,
        "model_calls": sum(item["call_count"] for item in usage),
        "prompt_tokens": sum(item["prompt_tokens"] for item in usage),
        "completion_tokens": sum(
            item["completion_tokens"] for item in usage),
    }


def compare_roots(
    left_root: Path, right_root: Path, *,
    left_label: str, right_label: str,
    manifest_root: Optional[Path] = None,
    reference_root: Optional[Path] = None,
) -> dict:
    left = _collect_root(left_root)
    right = _collect_root(right_root)
    references = _collect_root(reference_root) if reference_root else {}
    _attach_reference_metrics(left, references)
    _attach_reference_metrics(right, references)
    expected = _expected_variants(manifest_root)
    all_ids = sorted(set(expected) | set(left) | set(right))
    rows = []
    for variant_id in all_ids:
        metadata = (expected.get(variant_id) or left.get(variant_id)
                    or right.get(variant_id))
        rows.append({
            "variant_id": variant_id,
            "experiment_id": metadata["experiment_id"],
            "base_scenario_id": metadata["base_scenario_id"],
            left_label: left.get(variant_id),
            right_label: right.get(variant_id),
        })
    present_pairs = [
        row for row in rows
        if row[left_label] is not None and row[right_label] is not None
    ]
    paired = [
        row for row in present_pairs
        if row[left_label]["infrastructure_valid"]
        and row[right_label]["infrastructure_valid"]
    ]
    layer_wins = {}
    for layer in ("cabin", "single_vehicle"):
        counts = {left_label: 0, right_label: 0, "tie": 0,
                  "comparable": 0}
        for row in paired:
            left_score = row[left_label]["scores_100"].get(layer)
            right_score = row[right_label]["scores_100"].get(layer)
            if left_score is None or right_score is None:
                continue
            counts["comparable"] += 1
            if left_score > right_score:
                counts[left_label] += 1
            elif right_score > left_score:
                counts[right_label] += 1
            else:
                counts["tie"] += 1
        layer_wins[layer] = counts
    completion = {}
    for label in (left_label, right_label):
        values = [row[label] for row in rows]
        completion[label] = {
            "valid": sum(
                item is not None and item["infrastructure_valid"]
                for item in values),
            "invalid_infrastructure": sum(
                item is not None and not item["infrastructure_valid"]
                for item in values),
            "missing": sum(item is None for item in values),
        }
    return {
        "schema": "vehiclearena-dual-model-comparison-v3",
        "labels": [left_label, right_label],
        "left_root": str(left_root),
        "right_root": str(right_root),
        "manifest_root": str(manifest_root) if manifest_root else None,
        "reference_root": str(reference_root) if reference_root else None,
        "reference_scene_count": len(references),
        "expected_scene_count": len(expected) if expected else len(rows),
        "present_pair_count": len(present_pairs),
        "paired_scene_count": len(paired),
        "missing": {
            left_label: [row["variant_id"] for row in rows
                         if row[left_label] is None],
            right_label: [row["variant_id"] for row in rows
                          if row[right_label] is None],
        },
        "summary": {
            left_label: _summary([row[left_label] for row in paired]),
            right_label: _summary([row[right_label] for row in paired]),
        },
        "layer_wins": layer_wins,
        "completion": completion,
        "scenes": rows,
    }


def render_markdown(report: dict) -> str:
    left_label, right_label = report["labels"]
    left = report["summary"][left_label]
    right = report["summary"][right_label]
    lines = [
        "# VehicleArena 双模型 LLM 场景对比",
        "",
        f"计划场景：{report['expected_scene_count']}；有效完整配对："
        f"{report['paired_scene_count']}；两侧文件均存在："
        f"{report['present_pair_count']}。车内分和驾驶分均为 "
        "0–100，N/A 不参与平均；任务到达率、NPC 影响和 Token 单独统计。",
        ("SUMO 配对基准仅用于计算系统指标差值。" if report["reference_root"]
         else "未提供 SUMO 配对基准，不计算系统指标差值。"),
        "",
        "## 完整性",
        "",
        f"| 状态 | {left_label} | {right_label} |",
        "|---|---:|---:|",
        (f"| 有效 | {report['completion'][left_label]['valid']} | "
         f"{report['completion'][right_label]['valid']} |"),
        ("| 基础设施无效 | "
         f"{report['completion'][left_label]['invalid_infrastructure']} | "
         f"{report['completion'][right_label]['invalid_infrastructure']} |"),
        (f"| 尚未生成 | {report['completion'][left_label]['missing']} | "
         f"{report['completion'][right_label]['missing']} |"),
        "",
        "基础设施无效项不进入任何均分或胜负统计。",
        "",
        "## 汇总",
        "",
        f"| 指标 | {left_label} | {right_label} |",
        "|---|---:|---:|",
        ("| 基础设施有效场景 | "
         f"{left['infrastructure_valid_count']}/{left['scene_count']} | "
         f"{right['infrastructure_valid_count']}/{right['scene_count']} |"),
        ("| 硬安全通过场景 | "
         f"{left['hard_safety_pass_count']}/{left['scene_count']} | "
         f"{right['hard_safety_pass_count']}/{right['scene_count']} |"),
        ("| 平均任务完成率 | "
         f"{_rate_text(left['mean_task_completed_rate'])} | "
         f"{_rate_text(right['mean_task_completed_rate'])} |"),
        ("| 平均车内分 | "
         f"{_score_text(left['mean_scores_100']['cabin'])} | "
         f"{_score_text(right['mean_scores_100']['cabin'])} |"),
        ("| 平均驾驶分 | "
         f"{_score_text(left['mean_scores_100']['single_vehicle'])} | "
         f"{_score_text(right['mean_scores_100']['single_vehicle'])} |"),
        f"| 模型调用 | {left['model_calls']} | {right['model_calls']} |",
        f"| 输入 Token | {left['prompt_tokens']:,} | {right['prompt_tokens']:,} |",
        (f"| 输出 Token | {left['completion_tokens']:,} | "
         f"{right['completion_tokens']:,} |"),
        "",
        "",
        "## 分层胜负",
        "",
        f"| 层级 | {left_label} 胜 | {right_label} 胜 | 平局 | 可比场景 |",
        "|---|---:|---:|---:|---:|",
    ]
    for layer, label in (("cabin", "车内"),
                         ("single_vehicle", "驾驶")):
        wins = report["layer_wins"][layer]
        lines.append(
            f"| {label} | {wins[left_label]} | {wins[right_label]} | "
            f"{wins['tie']} | {wins['comparable']} |")
    lines.extend([
        "",
        "## 逐场景",
        "",
        (f"| 实验 | 场景 | {left_label} 车内/驾驶 | "
         f"{right_label} 车内/驾驶 | 安全 | 任务完成率 |"),
        "|---|---|---:|---:|---|---|",
    ])
    for row in report["scenes"]:
        left_item = row[left_label]
        right_item = row[right_label]

        def pair_scores(item: Optional[dict]) -> str:
            if item is None:
                return "缺失"
            if not item["infrastructure_valid"]:
                return "无效（基础设施）"
            scores = item["scores_100"]
            return "/".join(_score_text(scores.get(key)) for key in (
                "cabin", "single_vehicle"))

        def pair_text(key: str, *, rate: bool = False) -> str:
            values = []
            for item in (left_item, right_item):
                if item is None or item.get(key) is None:
                    values.append("N/A")
                elif not item["infrastructure_valid"]:
                    values.append("无效")
                elif isinstance(item[key], bool):
                    values.append("是" if item[key] else "否")
                elif rate:
                    values.append(_rate_text(float(item[key])))
                else:
                    values.append(f"{float(item[key]):.2f}")
            return "/".join(values)

        lines.append(
            f"| {row['experiment_id']} | `{row['variant_id']}` | "
            f"{pair_scores(left_item)} | {pair_scores(right_item)} | "
            f"{pair_text('hard_safety_passed')} | "
            f"{pair_text('task_completed_rate', rate=True)} |")
    lines.extend([
        "",
        f"表中成对值顺序始终为 `{left_label}/{right_label}`。",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-root", type=Path, required=True)
    parser.add_argument("--right-root", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument(
        "--reference-root", type=Path,
        help="Matching --sumo-reference root used for system metric deltas.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_roots(
        args.left_root, args.right_root,
        left_label=args.left_label, right_label=args.right_label,
        manifest_root=args.manifest_root,
        reference_root=args.reference_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "comparison.md").write_text(
        render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "paired_scene_count": report["paired_scene_count"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
