#!/usr/bin/env python3
"""Run all catalog scenes in isolated processes with auditable checkpoints.

Credentials are read from environment variables only. ``OPENAI_API_KEY`` (or
``FOCAL_OPENAI_API_KEY``) authenticates the focal model; optional
``PEER_OPENAI_API_KEY`` and ``AUX_OPENAI_API_KEY`` can authenticate the fixed
peer and PA/Judge roles independently. Each scene has its own status and
attempt log; a failed or corrupt trajectory is never treated as completed.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import faulthandler
import gzip
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vehiclearena")]

from evaluation.experiments.manifest import load_manifest, source_fingerprint
from evaluation.experiments.pipelines import prepare_scene_manifests


DRIVER_ROLES = {
    "cautious": "You are a cautious driver. Prefer generous following gaps, early observation, and gradual speed changes. Yield when right of way or visibility is uncertain; proceed when the situation is safely resolved. Complete your own route and passenger tasks.",
    "efficient": "You are an efficient driver. Make timely progress along your own route, choose legal useful gaps, and avoid unnecessary stops. Observe speed limits and right of way, and never trade safety for speed. Complete passenger tasks when safe.",
    "cooperative": "You are a cooperative driver. Make your actions predictable, signal intentions, and accommodate safe merges and vulnerable road users. Resolve interactions using only your own observations and available tools; do not assume access to other agents' plans. Complete your own route and passenger tasks.",
    "comfortable": "You are a comfort-focused driver. Prefer smooth acceleration, progressive braking and stable lane choice. Balance passenger comfort with timely legal progress, following distances and road safety. Complete your own route and passenger tasks.",
}


def parse_study_ratio(value):
    """Parse a Basic:MultiLLM submission ratio such as ``2:1``."""
    if value is None:
        return None
    try:
        basic, multillm = (int(item) for item in str(value).split(":", 1))
    except (TypeError, ValueError):
        raise ValueError(
            "study ratio must have the form BASIC:MULTILLM, e.g. 2:1")
    if basic <= 0 or multillm <= 0:
        raise ValueError("study ratio components must be positive")
    return {"Basic": basic, "MultiLLM": multillm}


def choose_ratio_study(queues, active_counts, weights):
    """Choose the next study while keeping active scene jobs near a ratio.

    The scheduler fills free worker slots lazily.  It therefore controls the
    ratio of *submitted/running* scene processes, instead of submitting the
    whole queue at once and accidentally sending a Basic-only burst to the
    remote model service.
    """
    available = [
        study for study in ("Basic", "MultiLLM")
        if queues.get(study) and queues[study]
    ]
    if not available:
        return None
    return min(
        available,
        key=lambda study: (
            active_counts.get(study, 0) / weights[study],
            0 if study == "Basic" else 1,
        ),
    )


def drain_requested(args):
    """Whether the controller should stop submitting new scene workers."""
    marker = getattr(args, "drain_file", None)
    return bool(marker and Path(marker).exists())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_model_trace(path):
    """Recover complete JSON records even when concurrent stdout lines join."""
    decoder = json.JSONDecoder()
    for line in Path(path).read_text(errors="replace").splitlines():
        for match in re.finditer(r'\{"event": "model_call(?:_start)?"', line):
            try:
                row, _ = decoder.raw_decode(line[match.start():])
            except ValueError:
                continue
            yield row


def compatible_hashes(manifest):
    return [manifest.manifest_hash] + manifest.metadata.get(
        "single_system_compatible_previous_hashes", [])


def inspect_run(path, manifest_hash=None):
    if not path.exists():
        return {"status": "missing"}
    try:
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
        hashes = ([manifest_hash] if isinstance(manifest_hash, str)
                  else list(manifest_hash or []))
        actual_hash = data.get("manifest_hash")
        if hashes and actual_hash not in hashes:
            return {"status": "stale_manifest"}
        if hashes and actual_hash != hashes[0]:
            # The sole compatible source change merges loaded skills into the
            # first system message. Reuse old successes only when every actual
            # request already used exactly one system message at index zero.
            snapshots = data.get("agent_context_snapshots", {})
            if not snapshots or any(
                not row.get("messages")
                or row["messages"][0].get("role") != "system"
                or sum(m.get("role") == "system" for m in row["messages"]) != 1
                for rows in snapshots.values() for row in rows
            ):
                return {"status": "stale_manifest"}
        errors = list(data.get("agent_infrastructure_errors", []))
        role_errors = []
        for entity_id, provenance in data.get("agent_provenance", {}).items():
            for role in ("personal_agent", "passenger_judge"):
                for error in (provenance.get(role) or {}).get("errors", []):
                    item = {"entity_id": entity_id, "role": role, **error}
                    role_errors.append(item)
                    if "Error:" in str(error.get("error", "")):
                        errors.append(item)
        status = data["run"]["status"]
        if errors:
            status = "invalid_infrastructure"
        calls = [call for rows in data.get("model_call_log", {}).values() for call in rows]
        return {
            "status": status, "errors": errors, "role_errors": role_errors,
            "manifest_hash": actual_hash,
            "wall_time_s": data["run"]["wall_time_s"],
            "all_vehicles_arrived": data["result"].get("all_vehicles_arrived"),
            "collision_count": len(data.get("collision_log", [])),
            "model_calls": len(calls),
            "failed_model_calls": sum(call.get("ok") is False for call in calls),
            "prompt_tokens": sum(call.get("prompt_tokens", 0) for call in calls),
            "completion_tokens": sum(call.get("completion_tokens", 0) for call in calls),
        }
    except Exception as exc:
        return {"status": "unreadable", "error": str(exc)}


def active_workers(output):
    """Discover this suite's surviving isolated workers after a controller restart."""
    found = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = [os.fsdecode(arg) for arg in
                    (directory / "cmdline").read_bytes().split(b"\0") if arg]
            if not argv or not Path(argv[0]).name.startswith("python"):
                continue
            if not any(arg.endswith("scripts/run_llm_suite.py") for arg in argv):
                continue
            if "--worker" not in argv or "--output" not in argv:
                continue
            actual_output = Path(argv[argv.index("--output") + 1])
            if not actual_output.is_absolute():
                actual_output = Path(os.readlink(directory / "cwd")) / actual_output
            if actual_output.resolve() != output:
                continue
            key = (argv[argv.index("--study") + 1], argv[argv.index("--worker") + 1])
            found[key] = {"pid": int(directory.name),
                          "log": os.readlink(directory / "fd/1")}
        except (OSError, ValueError, IndexError):
            continue
    return found


def disable_focal_in_cabin_agents(variant):
    """Disable PA and Passenger Judge only for a variant's scored vehicle.

    MultiLLM scenes identify the scored vehicle explicitly.  Basic scenes do
    not need a separate experiment block, so their single evaluated LLM
    vehicle is used as the focal vehicle.  The switches live in the scenario
    agent_config, which means background LLM peers retain the normal PA/Judge
    callbacks when the manifest is executed.
    """
    scenario = variant.scenario
    vehicles = scenario.get("vehicles", [])
    experiment_scene = scenario.get("experiment_scene", {}) or {}
    focal_id = experiment_scene.get("focal_vehicle_id")
    if focal_id is None:
        focal_candidates = [
            vehicle for vehicle in vehicles
            if vehicle.get("is_evaluated")
            and vehicle.get("agent_config", {}).get("type") == "llm"
        ]
        if len(focal_candidates) != 1:
            raise ValueError(
                f"Cannot identify one focal LLM vehicle for {variant.variant_id}: "
                f"{[vehicle.get('vehicle_id') for vehicle in focal_candidates]!r}")
        focal_id = focal_candidates[0].get("vehicle_id")
    for vehicle in vehicles:
        if vehicle.get("vehicle_id") != focal_id:
            continue
        agent_config = dict(vehicle.get("agent_config") or {})
        if agent_config.get("type") != "llm":
            return None
        agent_config["personal_agent_enabled"] = False
        agent_config["passenger_judge_enabled"] = False
        vehicle["agent_config"] = agent_config
        return focal_id
    raise ValueError(
        f"Focal vehicle {focal_id!r} is missing from {variant.variant_id}")


def prepare(args):
    # Keep programmatic callers with the original compact argument set
    # compatible while freezing the current role defaults in the output.
    role_defaults = {
        "max_output_tokens": 32768,
        "temperature": 0.7,
        "thinking": "enabled",
        "peer_model": args.model,
        "peer_base_url": args.base_url,
        "peer_temperature": 0.7,
        "peer_thinking": "enabled",
        "peer_reasoning_effort": "xhigh",
        "personal_agent_model": args.model,
        "personal_agent_temperature": 0.7,
        "passenger_judge_model": args.model,
        "passenger_judge_temperature": 0.0,
        "aux_thinking": "enabled",
        "aux_reasoning_effort": "xhigh",
        "endpoint_pool_offset": 0,
        "study_ratio": None,
    }
    for name, value in role_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    for name, value in {
        "peer_base_urls": [args.peer_base_url],
        "aux_base_url": args.peer_base_url,
    }.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    if not hasattr(args, "aux_base_urls"):
        args.aux_base_urls = [args.aux_base_url]
    ablation_enabled = bool(getattr(args, "disable_focal_pa_judge", False))
    allow_provisional_time_windows = bool(
        getattr(args, "allow_provisional_time_windows", False))
    catalog = getattr(args, "catalog", None) or ROOT / "vehiclearena/evaluation/experiments/scenarios"
    if not allow_provisional_time_windows:
        from prepare_native_npc_catalog import check_native_calibration
        check_native_calibration(
            catalog, getattr(args, "calibration_registry", None), args.only)
    paths = prepare_scene_manifests(
        args.output / "manifests", catalog)
    endpoint_assignments = {}
    all_catalog_variants = []
    for path in paths.values():
        all_catalog_variants.extend(load_manifest(path).variants)
    for index, variant in enumerate(sorted(
            all_catalog_variants, key=lambda item: item.variant_id)):
        endpoint_assignments[variant.variant_id] = {
            "peer": args.peer_base_urls[
                (args.endpoint_pool_offset + index) % len(args.peer_base_urls)],
            "aux": args.aux_base_urls[
                (args.endpoint_pool_offset + index) % len(args.aux_base_urls)],
        }
    variants = []
    assignment = {}
    if "MultiLLM" in paths:
        manifest = load_manifest(paths["MultiLLM"])
        for variant in manifest.variants:
            overrides = {
                key: dict(value)
                for key, value in variant.agent_overrides.items()}
            ids = [v["vehicle_id"] for v in variant.scenario["vehicles"]
                   if v.get("agent_config", {}).get("type") == "llm"]
            peer_ids = set(variant.scenario.get(
                "experiment_scene", {}).get("fixed_peer_vehicle_ids", []))
            assignment[variant.variant_id] = {}
            for index, entity_id in enumerate(ids):
                name, prompt = list(DRIVER_ROLES.items())[
                    index % len(DRIVER_ROLES)]
                overrides.setdefault(entity_id, {})[
                    "driver_prompt"] = prompt
                assignment[variant.variant_id][entity_id] = name
                if entity_id in peer_ids:
                    overrides[entity_id].update({
                        "model": args.peer_model,
                        "api_base": args.peer_base_url,
                        "temperature": args.peer_temperature,
                        "context_window_tokens": args.context_window,
                        "max_tokens": args.max_output_tokens,
                        "thinking_mode": args.peer_thinking,
                        "reasoning_effort": args.peer_reasoning_effort,
                        "chat_template_enable_thinking": (
                            args.peer_thinking == "enabled"),
                    })
            variants.append(dataclasses.replace(
                variant, agent_overrides=overrides))
        manifest.variants = variants
        manifest.metadata["driver_roles"] = DRIVER_ROLES
        manifest.metadata["driver_role_assignment"] = assignment
        manifest.write(paths["MultiLLM"])
    selected = set(args.only)
    available = {variant.variant_id for path in paths.values()
                 for variant in load_manifest(path).variants}
    if selected - available:
        raise ValueError(f"Unknown variants: {sorted(selected - available)}")
    for path in paths.values():
        study_manifest = load_manifest(path)
        chosen = [variant for variant in study_manifest.variants
                  if not selected or variant.variant_id in selected]
        # Manifests cannot be empty. Keep the unused study's catalog intact;
        # execution and reporting still filter by configuration.selected_variants.
        if chosen:
            study_manifest.variants = chosen
        for variant in study_manifest.variants:
            variant.scenario["max_parallel_model_calls"] = 4
        disabled_focals = []
        if ablation_enabled:
            for variant in study_manifest.variants:
                focal_id = disable_focal_in_cabin_agents(variant)
                if focal_id is not None:
                    disabled_focals.append({
                        "variant_id": variant.variant_id,
                        "focal_vehicle_id": focal_id,
                    })
        study_manifest.metadata["selected_variants"] = [
            variant.variant_id for variant in chosen]
        if ablation_enabled:
            study_manifest.metadata["focal_pa_judge_ablation"] = {
                "disabled": True,
                "variants": disabled_focals,
            }
        study_manifest.write(path)
    # OpenAI-compatible focal endpoints expose their reasoning budget through
    # ``reasoning_effort``. Keep these model-specific defaults on the focal
    # role only; fixed Qwen peer/PA/Judge defaults stay independent.
    focal_reasoning_effort = getattr(args, "reasoning_effort", None)
    if focal_reasoning_effort is None:
        focal_reasoning_effort = {
            "gpt-5.6-sol": "high",
            "kimi-k3": "max",
        }.get(str(args.model).lower())
    write_json(args.output / "configuration.json", {
        "model": args.model, "base_url": args.base_url,
        "context_window_tokens": args.context_window,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": focal_reasoning_effort,
        "chat_template_enable_thinking": args.thinking == "enabled",
        "focal": {
            "model": args.model, "api_base": args.base_url,
            "temperature": args.temperature,
            "context_window_tokens": args.context_window,
            "max_tokens": args.max_output_tokens,
            "thinking_mode": args.thinking,
            "reasoning_effort": focal_reasoning_effort,
            "chat_template_enable_thinking": args.thinking == "enabled",
        },
        "peer": {
            "model": args.peer_model, "api_base": args.peer_base_url,
            "temperature": args.peer_temperature,
            "context_window_tokens": args.context_window,
            "max_tokens": args.max_output_tokens,
            "thinking_mode": args.peer_thinking,
            "reasoning_effort": args.peer_reasoning_effort,
            "chat_template_enable_thinking": (
                args.peer_thinking == "enabled"),
        },
        "personal_agent": {
            "model": args.personal_agent_model,
            "api_base": args.aux_base_url,
            "temperature": args.personal_agent_temperature,
            "context_window_tokens": args.context_window,
            "max_tokens": args.max_output_tokens,
            "thinking_mode": args.aux_thinking,
            "reasoning_effort": args.aux_reasoning_effort,
            "chat_template_enable_thinking": args.aux_thinking == "enabled",
        },
        "passenger_judge": {
            "model": args.passenger_judge_model,
            "api_base": args.aux_base_url,
            "temperature": args.passenger_judge_temperature,
            "context_window_tokens": args.context_window,
            "max_tokens": args.max_output_tokens,
            "thinking_mode": args.aux_thinking,
            "reasoning_effort": args.aux_reasoning_effort,
            "chat_template_enable_thinking": args.aux_thinking == "enabled",
        },
        "personal_agent_enabled": True,
        "passenger_judge_model": args.passenger_judge_model,
        "personal_agent_runtime": {"trigger_mode": "event_random", "seed": 0},
        "passenger_judge_runtime": {
            "check_offsets_s": [0.1, 1.0, 3.0], "max_checks": 3,
            "acceptance_timeout_s": 3.0},
        "direct_api": args.direct_api,
        "scene_workers": args.workers, "max_parallel_model_calls": 4,
        "submission_study_ratio": args.study_ratio,
        "catalog": str(Path(catalog).resolve()),
        "calibration_registry": str(args.calibration_registry.resolve()) if getattr(args, "calibration_registry", None) else None,
        "time_window_policy": (
            "retained_provisional_not_revalidated"
            if allow_provisional_time_windows else "fresh_calibration_required"),
        "selected_variants": sorted(selected or available),
        "driver_roles": DRIVER_ROLES, "driver_role_assignment": assignment,
        "focal_pa_judge_ablation": {
            "disabled": ablation_enabled,
            "scope": "focal_vehicle_only",
        },
        "endpoint_pool_offset": args.endpoint_pool_offset,
        "peer_base_urls": list(args.peer_base_urls),
        "aux_base_urls": list(args.aux_base_urls),
        "endpoint_assignments": endpoint_assignments,
    })


def refresh_single_system_source(args):
    """Archive old manifests without relabelling their existing trajectories."""
    current_source = source_fingerprint(ROOT / "vehiclearena")
    for study in ("Basic", "MultiLLM"):
        path = args.output / "manifests" / f"{study}.manifest.json"
        manifest = load_manifest(path)
        if manifest.source_hash == current_source:
            continue
        old_hash = manifest.manifest_hash
        manifest.write(path.parent / "revisions" / manifest.source_hash / path.name)
        manifest.metadata.setdefault("single_system_compatible_previous_hashes", []).append(old_hash)
        manifest.metadata["revision_reason"] = (
            "Merge loaded skill bodies into the first system message for Qwen; "
            "reuse prior successes only if all saved requests already had one initial system message.")
        manifest.source_hash = current_source
        manifest.write(path)


def worker(args):
    # Periodic all-thread dumps have crashed inside CPython frame inspection
    # while callbacks were running. Progress comes from model_call logs and
    # controller checkpoints; the parent still enforces the worker timeout.
    faulthandler.enable(all_threads=False)
    from evaluation.multi_agent_runner import AgentClient
    from simulation.model_concurrency import model_io
    for method_name in ("chat", "chat_with_tools"):
        decorated = getattr(AgentClient, method_name)
        original = getattr(decorated, "__wrapped__", decorated)
        def traced(self, *a, _original=original, _name=method_name, **kw):
            started = time.monotonic()
            call_id = uuid.uuid4().hex
            print(json.dumps({"event": "model_call_start", "call_id": call_id,
                "client_id": id(self), "started_at_epoch_s": time.time(),
                "method": _name, "model": self.model}), flush=True)
            try:
                return _original(self, *a, **kw)
            finally:
                meta = self.last_call_metadata
                print(json.dumps({"event": "model_call", "method": _name,
                    "call_id": call_id, "client_id": id(self),
                    "ended_at_epoch_s": time.time(),
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "ok": meta.get("ok"), "tokens": meta.get("total_tokens"),
                    "finish_reason": meta.get("finish_reason")}), flush=True)
        setattr(AgentClient, method_name, model_io(traced))
    from evaluation.experiments.batch_runner import run_manifest
    frozen_config = json.loads((args.output / "configuration.json").read_text())
    focal_config = dict(frozen_config.get("focal", {
        "model": args.model, "api_base": args.base_url,
        "context_window_tokens": args.context_window,
        "max_tokens": frozen_config.get("max_output_tokens", 4096),
        "reasoning_effort": frozen_config.get("reasoning_effort"),
        "chat_template_enable_thinking": frozen_config.get(
            "chat_template_enable_thinking", False),
    }))
    peer_config = dict(frozen_config.get("peer", focal_config))
    personal_agent_config = dict(
        frozen_config.get("personal_agent", peer_config))
    passenger_judge_config = dict(frozen_config.get(
        "passenger_judge", peer_config))

    # Scaling runs may freeze a pool of equivalent Qwen endpoints.  Select
    # one endpoint per scene so Peer, PA, and Judge traffic for that scene is
    # colocated while the pool is balanced across the whole catalog.
    endpoint_assignment = (frozen_config.get("endpoint_assignments", {})
                           .get(args.worker, {}))
    if endpoint_assignment.get("peer"):
        peer_config["api_base"] = endpoint_assignment["peer"]
    if endpoint_assignment.get("aux"):
        personal_agent_config["api_base"] = endpoint_assignment["aux"]
        passenger_judge_config["api_base"] = endpoint_assignment["aux"]

    # Keep credentials out of configuration.json and trajectory provenance,
    # while allowing a GPT focal run to use Qwen3.8 for PA/Judge.  The batch
    # runner redacts api_key from persisted public runtime metadata.
    focal_api_key = (os.environ.get("FOCAL_OPENAI_API_KEY")
                     or os.environ.get("OPENAI_API_KEY"))
    peer_api_key = (os.environ.get("PEER_OPENAI_API_KEY")
                    or focal_api_key)
    aux_api_key = (os.environ.get("AUX_OPENAI_API_KEY")
                   or peer_api_key)
    if focal_api_key:
        focal_config["api_key"] = focal_api_key
    if peer_api_key:
        peer_config["api_key"] = peer_api_key
    if aux_api_key:
        personal_agent_config["api_key"] = aux_api_key
        passenger_judge_config["api_key"] = aux_api_key
    manifest_path = args.output / "manifests" / f"{args.study}.manifest.json"
    result = run_manifest(
        manifest_path, args.output / args.study,
        allow_llm=True, resume=False, variant_ids=[args.worker],
        status_filename=f"status-{args.worker}.json",
        llm_runtime_config=focal_config,
        fixed_peer_models=[peer_config["model"]],
        fixed_peer_runtime_config={
            "api_key": peer_config.get("api_key"),
            "api_base": peer_config.get("api_base")},
        personal_agent_runtime_config={
            "enabled": True, **personal_agent_config,
            **frozen_config.get(
                "personal_agent_runtime", {"trigger_mode": "legacy"})},
        passenger_judge_runtime_config={
            **passenger_judge_config,
            **frozen_config.get(
                "passenger_judge_runtime", {"window_s": 1.0})},
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["variants"][args.worker]["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--peer-base-url")
    parser.add_argument(
        "--peer-base-urls", nargs="+",
        help="Equivalent peer endpoints; assigned round-robin per scene")
    parser.add_argument("--peer-model", default="Qwen3.8-27B")
    parser.add_argument("--aux-base-url")
    parser.add_argument(
        "--aux-base-urls", nargs="+",
        help="Equivalent PA/Judge endpoints; assigned round-robin per scene")
    parser.add_argument("--personal-agent-model", default="Qwen3.8-27B")
    parser.add_argument("--passenger-judge-model", default="Qwen3.8-27B")
    parser.add_argument("--context-window", type=int, default=1000000)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--peer-temperature", type=float, default=0.7)
    parser.add_argument("--personal-agent-temperature", type=float, default=0.7)
    parser.add_argument("--passenger-judge-temperature", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("max", "xhigh", "high", "medium", "low", "minimal", "none"),
        default=None,
        help=(
            "Reasoning effort for the focal model. gpt-5.6-sol defaults "
            "to high and Kimi-K3 to max when this is omitted."),
    )
    parser.add_argument(
        "--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument(
        "--peer-thinking", choices=("enabled", "disabled"),
        default="enabled")
    parser.add_argument(
        "--peer-reasoning-effort",
        choices=("max", "xhigh", "high", "medium", "low", "minimal", "none"),
        default="xhigh")
    parser.add_argument(
        "--aux-thinking", choices=("enabled", "disabled"),
        default="enabled")
    parser.add_argument(
        "--aux-reasoning-effort",
        choices=("max", "xhigh", "high", "medium", "low", "minimal", "none"),
        default="xhigh")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--study-ratio",
        help=(
            "Keep active scene submissions near BASIC:MULTILLM, for example "
            "2:1. Without this option, jobs use the legacy queue order."),
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--disable-focal-pa-judge", action="store_true",
        help=(
            "Ablation mode: disable both PA and Passenger Judge only for the "
            "scored focal vehicle; background LLM agents keep both enabled."),
    )
    parser.add_argument("--catalog", type=Path,
                        help="Optional isolated scene catalog, including seeded NPC variants")
    parser.add_argument("--calibration-registry", type=Path,
                        help="Matching fresh baselines required for seeded NPC catalogs")
    parser.add_argument(
        "--allow-provisional-time-windows", action="store_true",
        help=("Use the catalog's retained time limits without requiring a "
              "fresh physical calibration; recorded in configuration.json."))
    parser.add_argument("--refresh-single-system-source", action="store_true",
                        help="Archive manifests and record the audited system-message fix revision")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--selection-file", type=Path,
        help="JSON list of scene IDs or objects containing scene_id.")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--timeout", type=float, default=14400,
        help=(
            "Per-scene controller timeout in seconds; 0 disables the "
            "controller timeout (default: 14400)."),
    )
    parser.add_argument("--direct-api", action="store_true",
                        help="Bypass environment proxies only for the model API host")
    parser.add_argument("--adopt-active", action="store_true",
                        help="Wait for existing isolated workers instead of duplicating them")
    parser.add_argument(
        "--drain-file", type=Path,
        help=(
            "Stop submitting new scenes once this marker exists; active "
            "workers are allowed to finish for a safe handoff."),
    )
    parser.add_argument("--worker")
    parser.add_argument("--study", choices=["Basic", "MultiLLM"])
    parser.add_argument(
        "--endpoint-pool-offset", type=int, default=0,
        help="Starting index for round-robin endpoint assignment")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.timeout < 0:
        parser.error("--timeout must be non-negative; use 0 for unlimited")
    if args.study_ratio:
        try:
            study_ratio = parse_study_ratio(args.study_ratio)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        study_ratio = None
    args.output = args.output.resolve()
    args.peer_base_url = args.peer_base_url or args.base_url
    args.aux_base_url = args.aux_base_url or args.peer_base_url
    args.peer_base_urls = list(args.peer_base_urls or [args.peer_base_url])
    args.aux_base_urls = list(args.aux_base_urls or [args.aux_base_url])
    if any(not str(url).strip() for url in [*args.peer_base_urls,
                                             *args.aux_base_urls]):
        parser.error("endpoint pools must not contain empty URLs")
    args.peer_base_url = args.peer_base_urls[0]
    args.aux_base_url = args.aux_base_urls[0]
    if args.selection_file:
        selection = json.loads(args.selection_file.read_text())
        if not isinstance(selection, list):
            parser.error("--selection-file must contain a JSON list")
        try:
            selected = [
                item if isinstance(item, str) else item["scene_id"]
                for item in selection]
        except (KeyError, TypeError):
            parser.error(
                "--selection-file entries must be strings or contain scene_id")
        args.only = list(dict.fromkeys([*args.only, *selected]))
    if args.direct_api:
        hosts = []
        for api_base in (
                args.base_url, *args.peer_base_urls, *args.aux_base_urls):
            host = urlparse(api_base).hostname
            if not host:
                parser.error("API base URLs must contain a host")
            if host not in hosts:
                hosts.append(host)
        exclusions = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
        exclusions = ",".join(filter(None, [exclusions, *hosts]))
        os.environ["NO_PROXY"] = exclusions
        os.environ["no_proxy"] = exclusions
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required")
    if args.worker:
        return worker(args)
    if args.prepare:
        prepare(args)
    if args.refresh_single_system_source:
        refresh_single_system_source(args)
    jobs = []
    records = {}
    for study in ["Basic", "MultiLLM"]:
        manifest_path = args.output / "manifests" / f"{study}.manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_manifest(manifest_path)
        for variant in manifest.variants:
            key = variant.variant_id
            if args.only and key not in args.only:
                continue
            output = args.output / study / f"{key}.json.gz"
            hashes = compatible_hashes(manifest)
            records[key] = {"study": study, **inspect_run(output, hashes)}
            if records[key]["status"] != "completed":
                jobs.append((study, key, hashes))
    unknown = set(args.only) - records.keys()
    if unknown:
        parser.error(f"Unknown variants: {sorted(unknown)}")
    adopted = active_workers(args.output)
    if adopted and not args.adopt_active:
        parser.error("This suite has active workers; use its existing controller or --adopt-active")
    # Exercise both studies early instead of deferring every multi-car case
    # until all Basic scenes finish. Include the close-junction regression first.
    jobs.sort(key=lambda job: (
        (job[0], job[1]) not in adopted,
        job[1] != "multi_009_two_lane_merge",
        int(job[1].split("_")[1]), job[0]))

    def execute(job):
        study, key, manifest_hash = job
        existing = adopted.get((study, key))
        if existing:
            process_args = Path(f"/proc/{existing['pid']}/cmdline")
            while process_args.exists():
                try:
                    if key.encode() not in process_args.read_bytes().split(b"\0"):
                        break
                except OSError:
                    break
                time.sleep(5)
        if drain_requested(args):
            return key, {
                "study": study,
                "status": "drained",
                "reason": "drain_requested",
            }
        cached = inspect_run(args.output / study / f"{key}.json.gz", manifest_hash)
        if cached["status"] == "completed":
            return key, {"study": study, **cached, "adopted": bool(existing)}
        log_dir = args.output / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(args.attempts):
            tag = time.time_ns()
            log_path = log_dir / f"{key}-{tag}.log"
            trajectory = args.output / study / f"{key}.json.gz"
            if trajectory.exists():
                archive = args.output / "attempts" / str(tag) / study
                archive.mkdir(parents=True, exist_ok=True)
                trajectory.rename(archive / trajectory.name)
                observations = args.output / study / "observations" / key
                if observations.exists():
                    (archive / "observations").mkdir()
                    observations.rename(archive / "observations" / key)
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--output", str(args.output), "--base-url", args.base_url,
                       "--model", args.model, "--context-window", str(args.context_window),
                       "--worker", key, "--study", study]
            if args.direct_api:
                command.append("--direct-api")
            started = time.monotonic()
            with log_path.open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           cwd=ROOT, start_new_session=True)
                try:
                    if args.timeout == 0:
                        # ``--timeout 0`` explicitly disables the controller's
                        # per-scene wall-clock limit. The simulation and
                        # provider may still finish normally or fail through
                        # their own mechanisms.
                        code = process.wait()
                    else:
                        code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = -1
            result = inspect_run(args.output / study / f"{key}.json.gz", manifest_hash)
            result.update({"study": study, "returncode": code, "log": str(log_path),
                           "attempts_this_launch": attempt + 1,
                           "elapsed_s": round(time.monotonic() - started, 2)})
            if code and result["status"] == "missing":
                result["status"] = "failed"
            write_json(log_path.with_suffix(".result.json"), result)
            if result["status"] == "completed":
                break
        return key, result

    def checkpoint():
        counts = dict(collections.Counter(row["status"] for row in records.values()))
        write_json(args.output / "suite-status.json", {
            "updated_at_epoch_s": time.time(), "total": len(records),
            "submission_study_ratio": args.study_ratio,
            "counts": counts, "variants": records,
        })
        print(json.dumps({"counts": counts, "total": len(records)}), flush=True)

    checkpoint()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        if study_ratio is None:
            futures = {pool.submit(execute, job): job for job in jobs}
            queues = None
            active_counts = None
        else:
            queues = {
                study: collections.deque(
                    job for job in jobs if job[0] == study)
                for study in ("Basic", "MultiLLM")
            }
            active_counts = {"Basic": 0, "MultiLLM": 0}
            futures = {}

            def submit_ratio_job():
                if drain_requested(args):
                    return False
                study = choose_ratio_study(
                    queues, active_counts, study_ratio)
                if study is None:
                    return False
                job = queues[study].popleft()
                futures[pool.submit(execute, job)] = job
                active_counts[study] += 1
                return True

            while len(futures) < args.workers and submit_ratio_job():
                pass
        while futures:
            done, _ = concurrent.futures.wait(futures, timeout=30,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                if active_counts is not None:
                    active_counts[job[0]] -= 1
                try:
                    key, result = future.result()
                except Exception as exc:
                    key, result = job[1], {"study": job[0], "status": "controller_error", "error": str(exc)}
                records[key] = result
                print(json.dumps({"variant": key, **result}, ensure_ascii=False), flush=True)
                if active_counts is not None:
                    submit_ratio_job()
            checkpoint()
    return 0 if all(row["status"] == "completed" for row in records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
