#!/usr/bin/env python3
"""Run one shared Basic and MultiLLM paired-reference suite.

Basic references replace every LLM vehicle with SUMO.  MultiLLM references
replace only the focal vehicle with SUMO; fixed peer LLMs stay on their
configured model so only focal behaviour differs.  Pairing is decided by the
persisted protocol hash; source and manifest hashes remain provenance and may
differ across collaborators.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vehiclearena")]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def active_workers(output):
    """Find detached reference workers so a restarted controller can adopt them."""
    found = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = [os.fsdecode(arg) for arg in
                    (directory / "cmdline").read_bytes().split(b"\0") if arg]
            if not argv or not Path(argv[0]).name.startswith("python"):
                continue
            if not any(arg.endswith("scripts/run_reference_suite.py")
                       for arg in argv):
                continue
            if not all(flag in argv for flag in
                       ("--worker", "--study", "--output")):
                continue
            actual_output = Path(argv[argv.index("--output") + 1])
            if not actual_output.is_absolute():
                actual_output = Path(os.readlink(directory / "cwd")) / actual_output
            if actual_output.resolve() != output:
                continue
            key = (argv[argv.index("--study") + 1],
                   argv[argv.index("--worker") + 1])
            found[key] = int(directory.name)
        except (OSError, ValueError, IndexError):
            continue
    return found


def worker(args):
    from evaluation.experiments.batch_runner import run_manifest

    source_config = json.loads(
        ((args.source_output or args.output) / "configuration.json").read_text())
    # A paired reference must retain the treatment's fixed-peer behavioural
    # configuration. Endpoints may differ because they are not protocol data.
    peer_config = dict(source_config.get("peer") or {})
    peer_config.setdefault("temperature", source_config.get(
        "temperature", 0.7))
    peer_config.setdefault("max_tokens", source_config.get(
        "max_output_tokens", 4096))
    peer_config.setdefault("context_window_tokens", args.context_window)
    peer_config["model"] = (
        args.peer_model or peer_config.get("model") or args.model)
    peer_config["api_base"] = (
        args.peer_base_url or peer_config.get("api_base") or args.base_url)
    if args.peer_thinking:
        peer_config.update({
            "thinking_mode": "enabled",
            "chat_template_enable_thinking": True,
        })
    if not peer_config.get("reasoning_effort"):
        peer_config["reasoning_effort"] = "xhigh"

    # Calibration must exercise the same in-cabin agents as treatment runs.
    # Copy the frozen role configuration instead of silently inheriting the
    # driver client's smaller defaults.  Old source configurations fall back
    # to the current fixed Qwen role contract.
    personal_config = dict(
        source_config.get("personal_agent") or peer_config)
    personal_config.setdefault("model", peer_config["model"])
    personal_config.setdefault("api_base", peer_config["api_base"])
    personal_config.setdefault("temperature", 0.7)
    personal_config.setdefault("context_window_tokens", args.context_window)
    personal_config.setdefault("max_tokens", 32768)
    personal_config.setdefault("thinking_mode", "enabled")
    if not personal_config.get("reasoning_effort"):
        personal_config["reasoning_effort"] = "xhigh"
    personal_config.setdefault("chat_template_enable_thinking", True)

    source_judge_config = source_config.get("passenger_judge")
    judge_config = dict(source_judge_config or peer_config)
    judge_config.setdefault("model", peer_config["model"])
    judge_config.setdefault("api_base", peer_config["api_base"])
    judge_config["temperature"] = float(
        judge_config.get("temperature", 0.0)
        if source_judge_config else 0.0)
    judge_config.setdefault("context_window_tokens", args.context_window)
    judge_config.setdefault("max_tokens", 32768)
    judge_config.setdefault("thinking_mode", "enabled")
    if not judge_config.get("reasoning_effort"):
        judge_config["reasoning_effort"] = "xhigh"
    judge_config.setdefault("chat_template_enable_thinking", True)

    personal_runtime = {
        **personal_config,
        **dict(source_config.get("personal_agent_runtime") or {}),
        "enabled": True,
    }
    judge_runtime = {
        **judge_config,
        **dict(source_config.get("passenger_judge_runtime") or {}),
        "enabled": True,
    }
    result = run_manifest(
        args.output / "manifests" / f"{args.study}.manifest.json",
        args.output / args.study,
        allow_llm=True, resume=False, variant_ids=[args.worker],
        status_filename=f"status-{args.worker}.json",
        llm_runtime_config=peer_config,
        fixed_peer_models=[peer_config["model"]],
        personal_agent_runtime_config=personal_runtime,
        passenger_judge_runtime_config=judge_runtime,
        sumo_reference=True,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["variants"][args.worker]["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-output", type=Path,
        help="Optional treatment suite whose manifests/config are reused")
    parser.add_argument(
        "--prepare", action="store_true",
        help="Prepare fresh Basic and MultiLLM manifests from --catalog")
    parser.add_argument(
        "--catalog", type=Path,
        default=ROOT / "vehiclearena/evaluation/experiments/scenarios")
    parser.add_argument("--calibration-registry", type=Path)
    parser.add_argument(
        "--allow-provisional-time-windows", action="store_true",
        help="Retain catalog time limits without requiring a matching fresh calibration")
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--peer-base-url",
                        help="Peer-vehicle model endpoint (default: --base-url)")
    parser.add_argument("--peer-model",
                        help="Fixed-peer driver model (default: --model)")
    parser.add_argument("--peer-thinking", action="store_true",
                        help=("Force thinking on for reference peers; by "
                              "default inherit the treatment peer config"))
    parser.add_argument("--context-window", type=int, default=1000000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--timeout", type=float, default=28800,
        help="Per-scene timeout in seconds; 0 disables the controller timeout")
    parser.add_argument("--direct-api", action="store_true")
    parser.add_argument(
        "--adopt-active", action="store_true",
        help="Wait for existing detached workers instead of rerunning them")
    parser.add_argument("--worker")
    parser.add_argument("--study", choices=("Basic", "MultiLLM"))
    args = parser.parse_args()
    if args.worker:
        if not args.study:
            parser.error("--study is required with --worker")
        return worker(args)
    args.output = args.output.resolve()
    if args.source_output:
        args.source_output = args.source_output.resolve()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required")

    if args.direct_api:
        hosts = {
            urlparse(url).hostname
            for url in (args.base_url, args.peer_base_url)
            if urlparse(url).hostname
        }
        exclusions = os.environ.get("NO_PROXY", os.environ.get(
            "no_proxy", ""))
        exclusions = ",".join(filter(None, [exclusions, *sorted(hosts)]))
        os.environ["NO_PROXY"] = exclusions
        os.environ["no_proxy"] = exclusions

    if args.prepare:
        from run_llm_suite import prepare

        only = []
        if args.selection_file:
            selection = json.loads(args.selection_file.read_text())
            if not isinstance(selection, list):
                parser.error("--selection-file must contain a JSON list")
            only = [
                item if isinstance(item, str) else item["scene_id"]
                for item in selection]
        prepare(argparse.Namespace(
            output=args.output,
            catalog=args.catalog.resolve(),
            calibration_registry=(
                args.calibration_registry.resolve()
                if args.calibration_registry else None),
            allow_provisional_time_windows=args.allow_provisional_time_windows,
            only=only,
            model=args.model,
            base_url=args.base_url,
            peer_model=args.peer_model or args.model,
            peer_base_url=args.peer_base_url or args.base_url,
            peer_base_urls=[args.peer_base_url or args.base_url],
            aux_base_url=args.peer_base_url or args.base_url,
            aux_base_urls=[args.peer_base_url or args.base_url],
            endpoint_pool_offset=0,
            personal_agent_model=args.peer_model or args.model,
            passenger_judge_model=args.peer_model or args.model,
            context_window=args.context_window,
            max_output_tokens=32768,
            temperature=0.7,
            peer_temperature=0.7,
            peer_reasoning_effort="xhigh",
            personal_agent_temperature=0.7,
            passenger_judge_temperature=0.0,
            thinking="enabled",
            peer_thinking="enabled" if args.peer_thinking else "disabled",
            aux_thinking="enabled",
            aux_reasoning_effort="xhigh",
            direct_api=args.direct_api,
            workers=args.workers,
            study_ratio=None,
        ))
        args.source_output = args.output
    elif not args.source_output:
        parser.error("either --prepare or --source-output is required")

    manifests = args.output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    jobs = []
    for study in ("Basic", "MultiLLM"):
        source_manifest = (
            args.source_output / "manifests" / f"{study}.manifest.json")
        if not source_manifest.is_file():
            continue
        target = manifests / source_manifest.name
        if (target.exists()
                and target.read_bytes() != source_manifest.read_bytes()):
            source_payload = json.loads(source_manifest.read_text())
            target_payload = json.loads(target.read_text())
            source_scenarios = {
                item["variant_id"]: item.get("scenario_hash")
                for item in source_payload.get("variants", [])}
            target_scenarios = {
                item["variant_id"]: item.get("scenario_hash")
                for item in target_payload.get("variants", [])}
            if source_scenarios != target_scenarios:
                parser.error(
                    f"Existing {study} reference manifest contains "
                    "different frozen scenarios")
        # Use the treatment copy for new workers. Existing references retain
        # provenance and are accepted only if protocol_hash agrees.
        if source_manifest.resolve() != target.resolve():
            shutil.copyfile(source_manifest, target)
        manifest = json.loads(target.read_text())
        jobs.extend((study, v["variant_id"]) for v in manifest["variants"])
    if not jobs:
        parser.error("no Basic or MultiLLM manifests were found")
    adopted = active_workers(args.output)
    if adopted and not args.adopt_active:
        parser.error("This reference suite has active workers; use --adopt-active")
    jobs.sort(key=lambda job: (job not in adopted, job[0], job[1]))

    def execute(job):
        study, variant = job
        existing_pid = adopted.get(job)
        if existing_pid is not None:
            process_args = Path(f"/proc/{existing_pid}/cmdline")
            while process_args.exists():
                try:
                    if variant.encode() not in process_args.read_bytes().split(b"\0"):
                        break
                except OSError:
                    break
                time.sleep(2)
        log_dir = args.output / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        trajectory = args.output / study / f"{variant}.json.gz"
        if trajectory.exists():
            return job, {"status": "already_present", "study": study}
        for attempt in range(args.attempts):
            tag = time.time_ns()
            log_path = log_dir / f"{variant}-{tag}.log"
            if trajectory.exists():
                archive = args.output / "attempts" / str(tag)
                archive.mkdir(parents=True, exist_ok=True)
                trajectory.rename(archive / trajectory.name)
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--output", str(args.output),
                       "--source-output", str(args.source_output),
                       "--base-url", args.base_url, "--model", args.model,
                       "--context-window", str(args.context_window),
                       "--worker", variant, "--study", study]
            if args.peer_base_url:
                command.extend(["--peer-base-url", args.peer_base_url])
            if args.peer_model:
                command.extend(["--peer-model", args.peer_model])
            if args.peer_thinking:
                command.append("--peer-thinking")
            if args.direct_api:
                command.append("--direct-api")
            started = time.monotonic()
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=ROOT, start_new_session=True)
                try:
                    code = process.wait(timeout=(
                        None if args.timeout <= 0 else args.timeout))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 15)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, 9)
                        process.wait()
                    code = -1
            status = "completed" if (code == 0 and trajectory.exists()) \
                else "failed"
            result = {"status": status, "returncode": code,
                      "study": study,
                      "log": str(log_path),
                      "elapsed_s": round(time.monotonic() - started, 2)}
            write_json(log_path.with_suffix(".result.json"), result)
            if status == "completed":
                return job, result
        return job, result

    counts = collections.Counter()
    checkpoint_path = args.output / "reference-status.json"

    def checkpoint():
        write_json(checkpoint_path, {
            "updated_at_epoch_s": time.time(), "total": len(jobs),
            "counts": dict(counts), "variants": dict(counts_lists)})
        print(json.dumps({"counts": dict(counts)}), flush=True)

    counts_lists = {}
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers) as pool:
        futures = {pool.submit(execute, job): job for job in jobs}
        while futures:
            done, _ = concurrent.futures.wait(
                futures, timeout=30,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                study, variant = futures.pop(future)
                try:
                    _, result = future.result()
                except Exception as exc:
                    result = {"status": "controller_error", "study": study,
                              "error": str(exc)}
                counts[result["status"]] += 1
                counts_lists[variant] = result
                print(json.dumps(
                    {"variant": variant, **result}, ensure_ascii=False),
                    flush=True)
            checkpoint()
    checkpoint()
    return 0 if counts["completed"] + counts.get("already_present", 0) \
        == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
