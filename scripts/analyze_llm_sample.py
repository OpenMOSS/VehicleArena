#!/usr/bin/env python3
"""Read a selected real-LLM batch: physical outcomes, tools, PA/Judge and overlap.

Never equate a completed experiment with a successful driving task. API trace
overlap is measured within each isolated scene, not across scene processes.
"""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path

from run_llm_suite import inspect_run, write_json, read_model_trace


def percentile(values, q):
    values = sorted(values)
    return round(values[min(len(values) - 1, math.ceil(len(values) * q) - 1)], 3) if values else None


def trace_overlap(path):
    starts, intervals = {}, []
    for row in read_model_trace(path):
        if row.get("event") == "model_call_start":
            starts[row["call_id"]] = row
        elif row.get("event") == "model_call" and row.get("call_id") in starts:
            begin = starts.pop(row["call_id"])
            intervals.append((begin["started_at_epoch_s"], row["ended_at_epoch_s"]))
    points = sorted([(a, 1) for a, b in intervals] + [(b, -1) for a, b in intervals])
    active = peak = 0
    busy = overlap = total = 0.0
    previous = points[0][0] if points else 0
    for moment, delta in points:
        duration = moment - previous
        busy += duration if active else 0
        overlap += duration if active > 1 else 0
        total += active * duration
        active += delta
        peak = max(peak, active)
        previous = moment
    return {"trace": str(path), "finished_requests": len(intervals),
            "unfinished_requests": len(starts), "peak_in_scene_requests": peak,
            "network_busy_s": round(busy, 3), "network_overlap_s": round(overlap, 3),
            "summed_request_s": round(total, 3),
            "request_sum_over_union": round(total / busy, 3) if busy else None}


def analyze_scene(path, log_path=None):
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    calls = [row for rows in data.get("model_call_log", {}).values() for row in rows]
    tools = [dict(row, entity_id=vid) for vid, rows in data.get("tool_call_log", {}).items() for row in rows]
    failed_tools = [row for row in tools if isinstance(row.get("result"), dict)
                    and (row["result"].get("success") is False or row["result"].get("error"))]
    vehicles = []
    for vid, result in data["result"]["vehicles"].items():
        meta = data.get("agent_metadata", {}).get(vid, {})
        if meta.get("agent_type") != "llm":
            continue
        driving = result.get("driving_evaluation") or {}
        vehicles.append({"entity_id": vid, "model": meta.get("model"),
            "driver_prompt": meta.get("driver_prompt", ""),
            "is_evaluated": result.get("is_evaluated"), "arrived": result.get("arrived"),
            "hard_safety_passed": driving.get("hard_safety_passed"),
            "reasonable_driving_pass": driving.get("reasonable_driving_pass"),
            "metrics": driving.get("metrics", {}),
            "hard_violations": driving.get("hard_violations", []),
            "passenger_evaluation": result.get("passenger_evaluation")})
    passenger_records, checks, triggers, role_errors = [], [], [], []
    for vid, provenance in data.get("agent_provenance", {}).items():
        pa = provenance.get("personal_agent") or {}
        judge = provenance.get("passenger_judge") or {}
        passenger_records.extend(dict(row, entity_id=vid) for row in pa.get("request_records", []))
        checks.extend(dict(row, entity_id=vid) for row in judge.get("check_history", []))
        triggers.extend(dict(row, entity_id=vid) for row in pa.get("trigger_log", []))
        for role, state in [("personal_agent", pa), ("passenger_judge", judge)]:
            role_errors.extend(dict(row, entity_id=vid, role=role) for row in state.get("errors", []))
    request_counts = collections.Counter(row.get("status", "unknown") for row in passenger_records)
    by_wake = collections.defaultdict(list)
    for row in tools:
        by_wake[(row["entity_id"], row.get("time_s"))].append(row)
    finish_only = sum(all(row.get("function") == "finish" for row in rows) for rows in by_wake.values())
    trace = data.get("trajectories", {}).get("vehicles", [])
    row = {
        "scenario": data["result"]["scenario_id"], "study": data["experiment_id"],
        "file": str(path), "source_hash": data.get("source_hash"),
        "manifest_hash": data.get("manifest_hash"), "run": data["run"],
        "valid_run": inspect_run(path).get("status") == "completed",
        "simulation_end_s": max((item["time_s"] for item in trace), default=None),
        "all_vehicles_arrived": data["result"].get("all_vehicles_arrived"),
        "llm_vehicles": vehicles, "collisions": data.get("collision_log", []),
        "infrastructure_errors": data.get("agent_infrastructure_errors", []),
        "callback_errors": data.get("agent_callback_errors", []), "role_errors": role_errors,
        "models_called": sorted({call.get("model", "unknown") for call in calls}),
        "model_calls": len(calls),
        "calls_by_role": dict(collections.Counter(call.get("role", "unknown") for call in calls)),
        "failed_calls": sum(call.get("ok") is False for call in calls),
        "truncated_calls": sum(call.get("finish_reason") == "length" for call in calls),
        "prompt_tokens": sum(call.get("prompt_tokens", 0) for call in calls),
        "completion_tokens": sum(call.get("completion_tokens", 0) for call in calls),
        "latency_p50_s": percentile([call["latency_s"] for call in calls if "latency_s" in call], .5),
        "latency_p95_s": percentile([call["latency_s"] for call in calls if "latency_s" in call], .95),
        "latency_max_s": max((call.get("latency_s", 0) for call in calls), default=0),
        "tool_calls": len(tools), "failed_tools": failed_tools,
        "failed_tool_counts": dict(collections.Counter(row.get("function") for row in failed_tools)),
        "tool_wakes": len(by_wake), "finish_only_wakes": finish_only,
        "pa_triggers": triggers, "passenger_requests": passenger_records,
        "passenger_outcomes": dict(request_counts), "judge_checks": checks,
        "judge_bound_violations": [row for row in passenger_records if row.get("checks", 0) > 3],
        "request_lifetime_s": [round(row["closed_at_s"] - row["created_at_s"], 6)
                               for row in passenger_records if "closed_at_s" in row],
    }
    if log_path and log_path.exists():
        row["concurrency"] = trace_overlap(log_path)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    config = json.loads((args.output / "configuration.json").read_text())
    selected = set(config.get("selected_variants", []))
    status = json.loads((args.output / "suite-status.json").read_text())
    failures = []
    for key, row in status.get("variants", {}).items():
        if row.get("status") in {"failed", "controller_error", "invalid_infrastructure"}:
            detail_path = args.output / row["study"] / f"status-{key}.json"
            detail = json.loads(detail_path.read_text()).get("variants", {}).get(key, {}) if detail_path.exists() else {}
            failure = {"scenario": key, **row, **detail}
            if row.get("log"):
                failure["concurrency"] = trace_overlap(Path(row["log"]))
            failures.append(failure)
    scenes, previous = [], []
    for study in ("Basic", "MultiLLM"):
        for path in sorted((args.output / study).glob("*.json.gz")):
            key = path.name.removesuffix(".json.gz")
            if selected and key not in selected:
                continue
            log = status.get("variants", {}).get(key, {}).get("log")
            if not log:
                logs = sorted((args.output / "logs").glob(f"{key}-*.log"))
                log = str(logs[-1]) if logs else None
            scenes.append(analyze_scene(path, Path(log) if log else None))
            if args.previous:
                prior = args.previous / study / path.name
                if prior.exists() and inspect_run(prior).get("status") == "completed":
                    previous.append(analyze_scene(prior))
    tool_errors = collections.Counter()
    role_calls = collections.Counter()
    requests = collections.Counter()
    for scene in scenes:
        tool_errors.update(scene["failed_tool_counts"])
        role_calls.update(scene["calls_by_role"])
        requests.update(scene["passenger_outcomes"])
    summary = {
        "expected_scenes": len(selected), "trajectories": len(scenes),
        "failed_runs_without_valid_trajectory": len(failures),
        "valid_completed_runs": sum(scene["valid_run"] for scene in scenes),
        "all_vehicles_arrived_scenes": sum(bool(scene["all_vehicles_arrived"]) for scene in scenes),
        "all_llm_vehicles_arrived_scenes": sum(bool(s["llm_vehicles"]) and all(v["arrived"] for v in s["llm_vehicles"]) for s in scenes),
        "llm_vehicles": sum(len(scene["llm_vehicles"]) for scene in scenes),
        "arrived_llm_vehicles": sum(bool(v["arrived"]) for s in scenes for v in s["llm_vehicles"]),
        "hard_safety_passed_llm_vehicles": sum(v["hard_safety_passed"] is True for s in scenes for v in s["llm_vehicles"]),
        "reasonable_driving_passed_llm_vehicles": sum(v["reasonable_driving_pass"] is True for s in scenes for v in s["llm_vehicles"]),
        "collision_scenes": sum(bool(scene["collisions"]) for scene in scenes),
        "collision_events": sum(len(scene["collisions"]) for scene in scenes),
        "model_calls_by_role": dict(role_calls), "passenger_outcomes": dict(requests),
        "tool_failures_by_function": dict(tool_errors),
        **{key: sum(scene[key] for scene in scenes) for key in (
            "model_calls", "failed_calls", "truncated_calls", "prompt_tokens", "completion_tokens", "tool_calls")},
        "role_error_count": sum(len(scene["role_errors"]) for scene in scenes),
    }
    all_attempt_calls = [row for path in (args.output / "logs").glob("*.log")
                         for row in read_model_trace(path) if row.get("event") == "model_call"]
    summary["all_attempts_traced_model_calls"] = len(all_attempt_calls)
    summary["all_attempts_traced_failed_model_calls"] = sum(row.get("ok") is False for row in all_attempt_calls)
    summary["all_attempts_total_tokens"] = sum(row.get("tokens") or 0 for row in all_attempt_calls)
    report = {"summary": summary, "scenes": scenes, "failed_runs": failures, "previous_same_scenes": previous}
    write_json(args.output / "analysis.json", report)
    lines = ["# 10 场 Qwen3.8-27B 验证：量化结果", "",
        "`completed` 只表示实验有效结束，不代表到达或安全通过。样本为针对性选择，不能外推完整目录。", "",
        "| 场景 | 有效结束 | LLM 到达 | 碰撞 | 墙钟分钟 | 调用数 | 工具失败 | 场景内请求峰值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for scene in scenes:
        vehicles = scene["llm_vehicles"]
        lines.append(f"| {scene['scenario']} | {'是' if scene['valid_run'] else '否'} | "
                     f"{sum(bool(v['arrived']) for v in vehicles)}/{len(vehicles)} | {len(scene['collisions'])} | "
                     f"{scene['run']['wall_time_s']/60:.1f} | {scene['model_calls']} | {len(scene['failed_tools'])} | "
                     f"{scene.get('concurrency',{}).get('peak_in_scene_requests', '—')} |")
    if failures:
        lines.extend(["", "## 中断的运行（不计入有效轨迹统计）", ""])
        for failure in failures:
            lines.append(f"- {failure['scenario']}: `{failure.get('error', 'unknown')}`；日志：{failure.get('log', '')}")
    lines.extend(["", "## 汇总", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
        "## 解释边界", "",
        "- 并发按同一场景日志中的真实请求起止时间计算；请求时长之和/时间并集不是完整场景加速比。",
        "- 旧批次同场景数据保存在 analysis.json 的 previous_same_scenes；源码、PA 调度、服务延迟及采样都不同，不能据此给出因果提升。",
        "- Judge 仍使用用户指定的 0.1 秒间隔、最多 3 次。持续驾驶请求在 0.3 秒内难以充分验收，需要结合原始证据解释。",
        "- 有效轨迹指标不包含中断的尝试；all_attempts_* 另行统计所有场景尝试。两者均不包含独立接口预检。", ""])
    (args.output / "METRICS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
