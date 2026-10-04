"""Command-line interface for the Basic and Multi-LLM experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from evaluation.experiments.batch_runner import (
    aggregate_run_directory, run_manifest,
)
from evaluation.multi_agent_runner import (
    DEFAULT_PASSENGER_JUDGE_MAX_OUTPUT_TOKENS,
)
from evaluation.experiments.pipelines import prepare_scene_manifests
from evaluation.experiments.scene_catalog import generate_scene_catalog
from evaluation.experiments.scene_validator import validate_scene_catalog
from evaluation.experiments.map_coverage import (
    generate_map_coverage, validate_map_coverage,
)
from evaluation.experiments.map_visual_audit import render_map_visual_audit
from evaluation.experiments.integrity_checks import (
    audit_simulation_identity_special_cases,
)
from evaluation.experiments.time_window_calibration import (
    DEFAULT_REGISTRY_PATH, calibrate_time_windows,
    calibrate_time_windows_checkpointed,
)
from evaluation.experiments.deadlock_regression import (
    DEFAULT_EXPERIMENTS as DEADLOCK_EXPERIMENTS,
    run_deadlock_regression,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vehiclearena-experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_catalog = commands.add_parser("prepare-scene-manifests")
    prepare_catalog.add_argument("--catalog", type=Path, required=True)
    prepare_catalog.add_argument("--output", type=Path, required=True)
    prepare_catalog.add_argument("--quick", action="store_true")

    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--allow-llm", action="store_true")
    run.add_argument(
        "--sumo-reference", action="store_true",
        help=(
            "Produce the paired physical reference: replace every LLM "
            "role in Basic, or only the focal role in MultiLLM while "
            "retaining --peer-model policies."))
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--variant", action="append", default=[])
    run.add_argument(
        "--status-filename", default="status.json",
        help="Status file name for a sharded concurrent run.")
    run.add_argument("--llm-base-url")
    run.add_argument("--llm-model")
    run.add_argument(
        "--peer-model", action="append", default=[],
        help=(
            "Fixed peer model for MultiLLM; repeat for heterogeneous peers. "
            "Only --llm-model changes the focal vehicle."))
    run.add_argument("--llm-context-window-tokens", type=int)
    run.add_argument("--llm-max-output-tokens", type=int)
    run.add_argument("--llm-temperature", type=float)
    run.add_argument(
        "--llm-thinking", choices=("default", "enabled", "disabled"),
        default="default")
    run.add_argument(
        "--llm-reasoning-effort",
        choices=("max", "xhigh", "high", "medium", "low", "minimal", "none"))
    run.add_argument(
        "--llm-chat-template-thinking",
        choices=("default", "enabled", "disabled"), default="default")
    run.add_argument("--llm-todo-max-ttl-s", type=float)
    run.add_argument(
        "--with-personal-agent", action="store_true",
        help="Enable event/random passenger wakes (legacy mode is also available).")
    run.add_argument("--personal-agent-model")
    run.add_argument("--passenger-judge-model")
    run.add_argument(
        "--passenger-judge-max-output-tokens", type=int,
        default=DEFAULT_PASSENGER_JUDGE_MAX_OUTPUT_TOKENS,
        help="Maximum completion tokens for the independent passenger Judge.")
    run.add_argument(
        "--passenger-judge-window-s", type=float, default=None,
        help="Legacy fixed interval; omit for checks at +0.1, +1, +3 s (legacy mode: 1 s).")
    run.add_argument("--passenger-judge-check-offsets-s", type=float, nargs="+",
                     help="Increasing simulation-time offsets from each request, e.g. 0.1 1 3.")
    run.add_argument("--passenger-judge-acceptance-timeout-s", type=float,
                     help="Acceptance budget, not semantic request expiry.")
    run.add_argument("--personal-agent-persona", default="")
    run.add_argument("--personal-agent-long-term-goal", default="")
    run.add_argument("--personal-agent-trigger-mode", choices=["event_random", "legacy"], default="event_random")
    run.add_argument("--personal-agent-seed", type=int, default=0)
    run.add_argument("--personal-agent-random-min-s", type=float, default=15.0)
    run.add_argument("--personal-agent-random-max-s", type=float, default=45.0)
    run.add_argument("--personal-agent-cooldown-s", type=float, default=5.0)
    run.add_argument("--personal-agent-stopped-after-s", type=float, default=15.0)
    run.add_argument("--passenger-judge-max-checks", type=int, default=3)
    run.add_argument("--passenger-request-ttl-s", type=float,
                     help="Deprecated alias for acceptance timeout.")
    run.add_argument(
        "--web3d-stream-url",
        help=(
            "Publish synchronized SUMO frames to a running Web3D server, "
            "for example http://127.0.0.1:8765/api/live/frame."))
    run.add_argument("--web3d-session-id", default="live")
    run.add_argument("--web3d-focus-entity", default="")
    run.add_argument("--web3d-junction-id")
    run.add_argument("--web3d-radius-m", type=float, default=220.0)
    run.add_argument(
        "--web3d-realtime", action="store_true",
        help="Pace the opt-in visualized episode to simulation wall time.")

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--output", type=Path, required=True)
    prepare_scenes = commands.add_parser("prepare-scenes")
    prepare_scenes.add_argument("--output", type=Path, required=True)
    prepare_scenes.add_argument("--clean", action="store_true")
    validate_scenes = commands.add_parser("validate-scenes")
    validate_scenes.add_argument("--catalog", type=Path, required=True)
    validate_scenes.add_argument("--boot", action="store_true")
    validate_scenes.add_argument("--smoke-seconds", type=float)
    prepare_maps = commands.add_parser("prepare-map-coverage")
    prepare_maps.add_argument("--output", type=Path, required=True)
    prepare_maps.add_argument("--networks", default="")
    validate_maps = commands.add_parser("validate-map-coverage")
    validate_maps.add_argument("--catalog", type=Path, required=True)
    validate_maps.add_argument("--report", type=Path)
    validate_maps.add_argument("--workers", type=int, default=1)
    render_maps = commands.add_parser("render-map-visual-audit")
    render_maps.add_argument("--catalog", type=Path, required=True)
    render_maps.add_argument("--output", type=Path, required=True)
    render_maps.add_argument("--networks", default="")
    render_maps.add_argument("--workers", type=int, default=1)
    render_maps.add_argument("--duration", type=float, default=1.0)
    render_maps.add_argument("--frame-interval", type=float, default=0.1)
    render_maps.add_argument("--width", type=int, default=640)
    render_maps.add_argument("--height", type=int, default=640)
    calibrate_time = commands.add_parser("calibrate-time-windows")
    calibrate_time.add_argument("--catalog", type=Path, required=True)
    calibrate_time.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    calibrate_time.add_argument("--report", type=Path)
    calibrate_time.add_argument("--max-duration", type=float, default=1200.0)
    calibrate_time.add_argument("--workers", type=int, default=1)
    calibrate_time.add_argument("--scene", action="append", default=[])
    calibrate_time.add_argument("--multi-peer-base-url")
    calibrate_time.add_argument("--multi-peer-model")
    calibrate_time.add_argument(
        "--multi-peer-context-window-tokens", type=int, default=1000000)
    calibrate_time.add_argument(
        "--multi-peer-max-output-tokens", type=int, default=32768)
    calibrate_time.add_argument(
        "--multi-peer-temperature", type=float, default=0.7)
    calibrate_time.add_argument(
        "--multi-peer-thinking", choices=("default", "enabled", "disabled"),
        default="enabled")
    calibrate_time.add_argument(
        "--multi-peer-reasoning-effort",
        choices=("max", "xhigh", "high", "medium", "low", "minimal", "none"),
        default="xhigh")
    calibrate_time.add_argument(
        "--multi-peer-chat-template-thinking",
        choices=("default", "enabled", "disabled"), default="enabled")
    calibrate_time.add_argument(
        "--direct-api", action="store_true",
        help="Bypass environment proxies for the MultiLLM peer endpoint")
    calibrate_time.add_argument(
        "--bulk", action="store_true",
        help=(
            "Run all selected scenes as one parallel batch. The default "
            "checkpoints each scene atomically and can resume."))
    calibrate_time.add_argument("--no-resume", action="store_true")
    deadlock = commands.add_parser("audit-traffic-deadlocks")
    deadlock.add_argument("--catalog", type=Path, required=True)
    deadlock.add_argument("--output", type=Path, required=True)
    deadlock.add_argument(
        "--experiments", default=",".join(DEADLOCK_EXPERIMENTS))
    deadlock.add_argument("--tags", default="")
    deadlock.add_argument("--scene", action="append", default=[])
    deadlock.add_argument("--window", type=float, default=20.0)
    deadlock.add_argument("--workers", type=int, default=1)
    integrity = commands.add_parser("audit-runtime-integrity")
    integrity.add_argument(
        "--source", type=Path,
        default=Path(__file__).resolve().parents[2] / "simulation")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-scene-manifests":
        result = {key: str(value) for key, value in prepare_scene_manifests(
            args.output, args.catalog, quick=args.quick).items()}
    elif args.command == "run":
        if (args.allow_llm and not args.sumo_reference
                and args.llm_context_window_tokens is None):
            parser.error(
                "--llm-context-window-tokens is required with --allow-llm")
        llm_runtime_config = {
            "api_base": args.llm_base_url,
            "model": args.llm_model,
            "context_window_tokens": args.llm_context_window_tokens,
            "max_tokens": args.llm_max_output_tokens,
            "temperature": args.llm_temperature,
            "thinking_mode": args.llm_thinking,
            "reasoning_effort": args.llm_reasoning_effort,
            "chat_template_enable_thinking": {
                "enabled": True, "disabled": False,
            }.get(args.llm_chat_template_thinking),
            "todo_max_ttl_s": args.llm_todo_max_ttl_s,
        }
        result = run_manifest(
            args.manifest, args.output, allow_llm=args.allow_llm,
            resume=not args.no_resume, variant_ids=args.variant,
            status_filename=args.status_filename,
            llm_runtime_config=llm_runtime_config,
            fixed_peer_models=args.peer_model,
            personal_agent_runtime_config={
                "enabled": args.with_personal_agent,
                "model": args.personal_agent_model,
                "persona": args.personal_agent_persona,
                "long_term_goal": args.personal_agent_long_term_goal,
                "trigger_mode": args.personal_agent_trigger_mode,
                "seed": args.personal_agent_seed,
                "random_min_s": args.personal_agent_random_min_s,
                "random_max_s": args.personal_agent_random_max_s,
                "cooldown_s": args.personal_agent_cooldown_s,
                "stopped_after_s": args.personal_agent_stopped_after_s,
            },
            passenger_judge_runtime_config={
                "model": args.passenger_judge_model,
                "max_tokens": args.passenger_judge_max_output_tokens,
                "window_s": args.passenger_judge_window_s,
                "max_checks": args.passenger_judge_max_checks,
                "request_ttl_s": args.passenger_request_ttl_s,
                "check_offsets_s": args.passenger_judge_check_offsets_s,
                "acceptance_timeout_s": args.passenger_judge_acceptance_timeout_s,
            },
            web3d_runtime_config=(
                {
                    "stream_url": args.web3d_stream_url,
                    "session_id": args.web3d_session_id,
                    "focus_entity_id": args.web3d_focus_entity,
                    "junction_id": args.web3d_junction_id,
                    "radius_m": args.web3d_radius_m,
                    "realtime": args.web3d_realtime,
                }
                if args.web3d_stream_url else None),
            sumo_reference=args.sumo_reference)
    elif args.command == "aggregate":
        result = aggregate_run_directory(args.output)
    elif args.command == "prepare-scenes":
        result = generate_scene_catalog(args.output, clean=args.clean)
    elif args.command == "validate-scenes":
        result = validate_scene_catalog(
            args.catalog, boot=args.boot,
            smoke_duration_s=args.smoke_seconds)
    elif args.command == "prepare-map-coverage":
        networks = [item.strip() for item in args.networks.split(",")
                    if item.strip()]
        result = generate_map_coverage(
            args.output, network_ids=networks or None)
    elif args.command == "validate-map-coverage":
        result = validate_map_coverage(
            args.catalog, report_dir=args.report,
            workers=max(1, args.workers))
    elif args.command == "render-map-visual-audit":
        networks = [item.strip() for item in args.networks.split(",")
                    if item.strip()]
        result = render_map_visual_audit(
            args.catalog, args.output,
            workers=max(1, args.workers), network_ids=networks or None,
            duration_s=args.duration,
            frame_interval_s=args.frame_interval,
            width=args.width, height=args.height)
    elif args.command == "calibrate-time-windows":
        if bool(args.multi_peer_model) != bool(args.multi_peer_base_url):
            parser.error(
                "--multi-peer-model and --multi-peer-base-url must be used "
                "together")
        multi_peer_runtime_config = None
        if args.multi_peer_model:
            if not os.environ.get("OPENAI_API_KEY"):
                parser.error(
                    "OPENAI_API_KEY is required for MultiLLM peer calibration")
            if args.direct_api:
                host = urlparse(args.multi_peer_base_url).hostname
                if not host:
                    parser.error("--multi-peer-base-url must contain a host")
                exclusions = os.environ.get(
                    "NO_PROXY", os.environ.get("no_proxy", ""))
                exclusions = ",".join(filter(None, [exclusions, host]))
                os.environ["NO_PROXY"] = exclusions
                os.environ["no_proxy"] = exclusions
            multi_peer_runtime_config = {
                "api_base": args.multi_peer_base_url,
                "model": args.multi_peer_model,
                "context_window_tokens": (
                    args.multi_peer_context_window_tokens),
                "max_tokens": args.multi_peer_max_output_tokens,
                "temperature": args.multi_peer_temperature,
                "thinking_mode": args.multi_peer_thinking,
                "reasoning_effort": args.multi_peer_reasoning_effort,
                "chat_template_enable_thinking": {
                    "enabled": True, "disabled": False,
                }.get(args.multi_peer_chat_template_thinking),
            }
        if args.bulk:
            result = calibrate_time_windows(
                args.catalog, registry_path=args.registry,
                report_path=args.report,
                max_duration_s=args.max_duration,
                workers=max(1, args.workers), quick=False,
                scene_ids=args.scene,
                multi_peer_runtime_config=multi_peer_runtime_config)
        else:
            result = calibrate_time_windows_checkpointed(
                args.catalog, registry_path=args.registry,
                report_path=args.report,
                max_duration_s=args.max_duration,
                workers=max(1, args.workers),
                resume=not args.no_resume,
                scene_ids=args.scene,
                multi_peer_runtime_config=multi_peer_runtime_config)
    elif args.command == "audit-traffic-deadlocks":
        report = run_deadlock_regression(
            args.catalog, output_path=args.output,
            experiments=[item.strip()
                         for item in args.experiments.split(",")
                         if item.strip()],
            tags=[item.strip() for item in args.tags.split(",")
                  if item.strip()],
            scene_ids=args.scene, window_s=args.window,
            workers=max(1, args.workers))
        result = {"output": str(args.output), "summary": report["summary"]}
    else:
        result = audit_simulation_identity_special_cases(args.source)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
