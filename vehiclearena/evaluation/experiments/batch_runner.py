"""Resumable manifest runner for VehicleArena experiments."""

from __future__ import annotations

import copy
import gc
import gzip
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, Optional

from evaluation.experiments.manifest import (
    ExperimentManifest, ExperimentVariant, load_manifest, sha256_data,
    sha256_file, source_fingerprint,
)
from evaluation.experiments.reference_protocol import (
    REFERENCE_PROTOCOL_REVISION, build_reference_protocol, protocol_hash,
)
from evaluation.experiments.scene_catalog import CATALOG_SCHEMA
from evaluation.experiments.scene_validator import SceneValidator
from evaluation.experiments.pedestrian_evaluator import evaluate_pedestrians
from evaluation.experiments.calibration_evaluator import (
    evaluate_capability_calibration, evaluate_chassis_calibration,
    evaluate_traffic_calibration,
)
from evaluation.experiments.system_evaluator import evaluate_system
from evaluation.layer_scoring import (
    cabin_layer_score_100, single_vehicle_layer_score_100,
)
from evaluation.experiments.telemetry import (
    ExperimentTrackedEngine, callback_tool_logs,
)
from evaluation.multi_agent_runner import (
    DEFAULT_PASSENGER_JUDGE_MAX_OUTPUT_TOKENS,
    apply_resolved_agent_authorities, build_callbacks, resolve_agent_specs,
)
from evaluation.personal_agent import aggregate_passenger_judgements
from simulation.multi_sim_engine import MultiScenario


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(
        value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _write_gzip_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, default=str,
                  separators=(",", ":"))
    os.replace(temporary, path)


def _public_runtime_config(value: Optional[dict]) -> dict:
    """Remove credentials recursively before persisting run metadata."""
    def redact(item):
        if isinstance(item, dict):
            return {key: redact(child) for key, child in item.items()
                    if str(key).lower() not in ("api_key", "authorization")}
        if isinstance(item, (list, tuple)):
            return [redact(child) for child in item]
        return copy.deepcopy(item)
    return redact(value or {})


class _FailedVariant(RuntimeError):
    def __init__(self, original_error, diagnostic_path):
        super().__init__(original_error)
        self.original_error = original_error
        self.diagnostic_path = str(diagnostic_path)


def load_run(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _runtime_summary(model_calls: dict, wake_runtime: dict) -> dict:
    result = {}
    for entity_id in sorted(set(model_calls) | set(wake_runtime)):
        calls = list(model_calls.get(entity_id, []))
        wakes = list(wake_runtime.get(entity_id, []))
        latencies = sorted(float(item.get("latency_s", 0.0)) for item in calls)
        p95_index = max(0, int(0.95 * len(latencies) + 0.999999) - 1)
        result[entity_id] = {
            "wake_count": len(wakes),
            "model_call_count": len(calls),
            "model_calls_per_wake": round(
                len(calls) / len(wakes), 4) if wakes else None,
            "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in calls),
            "completion_tokens": sum(
                int(item.get("completion_tokens", 0)) for item in calls),
            "model_latency_total_s": round(sum(latencies), 6),
            "model_latency_mean_s": round(
                sum(latencies) / len(latencies), 6) if latencies else None,
            "model_latency_p95_s": round(
                latencies[p95_index], 6) if latencies else None,
            "wake_wall_time_total_s": round(sum(
                float(item.get("wall_time_s", 0.0)) for item in wakes), 6),
            "response_truncation_count": sum(
                int(item.get("response_truncations", 0)) for item in wakes),
        }
    return result


def _task_passenger_summary(
    result: dict, *, focal_id: str, focal_only: bool,
) -> tuple[dict, dict]:
    """Return task-level and all-agent PA summaries without mixing scopes."""
    all_agents = result.get("evaluation", {}).get(
        "passenger_interaction", {}) or {}
    if not focal_only:
        return all_agents, all_agents
    focal = (all_agents.get("vehicles", {}) or {}).get(focal_id)
    # A missing focal report must not silently inherit successful peer work.
    return (focal or {}), all_agents


def _task_cabin_summary(
    result: dict, *, focal_id: str, focal_only: bool,
) -> tuple[Optional[float], Optional[float]]:
    """Return task-level cabin score and completion rate.

    MultiLLM stores a cabin result for every evaluated LLM vehicle.  The
    focal vehicle is the task subject; peer cabin results must remain
    available under ``result.vehicles`` but cannot affect the task metric.
    """
    overall = (result.get("evaluation", {}).get("cabin", {}) or {})
    if not focal_only:
        return overall.get("score"), overall.get("task_completion_rate")
    focal = (result.get("vehicles", {}).get(focal_id, {}) or {})
    cabin = focal.get("cabin_evaluation")
    if isinstance(cabin, dict):
        return cabin.get("score"), cabin.get("task_completion_rate")
    # Do not fall back to the top-level field here: in a MultiLLM artifact it
    # may be the peer-contaminated average this helper is designed to avoid.
    return None, None


class ExperimentBatchRunner:
    def __init__(
        self, manifest: ExperimentManifest, output_dir: Path,
        *, allow_llm: bool = False, resume: bool = True,
        status_filename: str = "status.json",
        llm_runtime_config: Optional[dict] = None,
        fixed_peer_models: Optional[list[str]] = None,
        fixed_peer_runtime_config: Optional[dict] = None,
        personal_agent_runtime_config: Optional[dict] = None,
        passenger_judge_runtime_config: Optional[dict] = None,
        web3d_runtime_config: Optional[dict] = None,
        sumo_reference: bool = False,
    ):
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.allow_llm = bool(allow_llm)
        self.resume = bool(resume)
        self.status_filename = Path(status_filename).name
        self.llm_runtime_config = {
            key: value for key, value in (llm_runtime_config or {}).items()
            if value is not None
        }
        self.public_llm_runtime_config = {
            key: value for key, value in self.llm_runtime_config.items()
            if key not in {"api_key"}
        }
        self.fixed_peer_models = [
            str(model) for model in (fixed_peer_models or []) if str(model)]
        # Per-peer credentials must remain runtime-only.  In particular, a
        # GPT focal run uses a different key from its fixed Qwen peers; do
        # not rely on the process-wide OPENAI_API_KEY fallback.
        self.fixed_peer_runtime_config = {
            key: copy.deepcopy(value)
            for key, value in (fixed_peer_runtime_config or {}).items()
            if value is not None}
        self.sumo_reference = bool(sumo_reference)
        self.reference_policy = (
            "focal_sumo_fixed_peers"
            if self.sumo_reference
            and self.manifest.experiment_id == "MultiLLM"
            else "all_sumo"
            if self.sumo_reference else None
        )
        if (self.reference_policy == "focal_sumo_fixed_peers"
                and not self.fixed_peer_models):
            raise ValueError(
                "MultiLLM paired references require at least one fixed "
                "peer model; only the focal vehicle is replaced by SUMO")
        self.personal_agent_runtime_config = copy.deepcopy(
            personal_agent_runtime_config or {})
        self.passenger_judge_runtime_config = {
            key: copy.deepcopy(value)
            for key, value in (passenger_judge_runtime_config or {}).items()
            if value is not None}
        self.passenger_judge_runtime_config.setdefault(
            "max_tokens", DEFAULT_PASSENGER_JUDGE_MAX_OUTPUT_TOKENS)
        if self.personal_agent_runtime_config.get("trigger_mode") == "legacy":
            self.passenger_judge_runtime_config.setdefault("window_s", 1.0)
        else:
            from evaluation.judge_schedule import resolve_judge_schedule
            offsets, timeout = resolve_judge_schedule(**{
                key: value for key, value in self.passenger_judge_runtime_config.items()
                if key in {"window_s", "max_checks", "check_offsets_s",
                           "acceptance_timeout_s", "request_ttl_s"}})
            self.passenger_judge_runtime_config.pop("window_s", None)
            self.passenger_judge_runtime_config.pop("request_ttl_s", None)
            self.passenger_judge_runtime_config.update(
                check_offsets_s=offsets, acceptance_timeout_s=timeout)
        self.web3d_runtime_config = copy.deepcopy(
            web3d_runtime_config or {})
        self.passenger_judge_window_s = float(
            (self.passenger_judge_runtime_config or {}).get(
                "window_s", self.passenger_judge_runtime_config.get(
                    "check_offsets_s", [1.0])[0]))
        if self.passenger_judge_window_s <= 0.0:
            raise ValueError("passenger judge window must be positive")
        self.public_personal_agent_runtime_config = _public_runtime_config(
            personal_agent_runtime_config)
        self.public_passenger_judge_runtime_config = _public_runtime_config(
            self.passenger_judge_runtime_config)
        self.personal_agent_enabled = bool(
            self.personal_agent_runtime_config.get("enabled", False))
        self.callback_personal_agent_config = copy.deepcopy(
            self.personal_agent_runtime_config)
        self.runtime_integrity_warnings = []

    def _validate_runtime_integrity(
        self, variant_ids: Optional[Iterable[str]] = None,
    ) -> None:
        """Validate runnable assets; retain broad hashes as provenance only."""
        source_root = Path(__file__).resolve().parents[2]
        errors = []
        warnings = []
        if self.manifest.source_hash:
            current_source_hash = source_fingerprint(source_root)
            if current_source_hash != self.manifest.source_hash:
                warnings.append({
                    "type": "source_hash_mismatch",
                    "message": (
                        "runtime source differs from the manifest; "
                        "reference compatibility is decided by protocol_hash"),
                    "manifest_source_hash": self.manifest.source_hash,
                    "runtime_source_hash": current_source_hash,
                })

        network_dir = source_root / "simulation" / "road_networks"
        for filename, expected_hash in sorted(
                self.manifest.metadata.get("map_hashes", {}).items()):
            path = network_dir / filename
            if not path.is_file():
                errors.append(f"manifest map is missing: {filename}")
            elif sha256_file(path) != expected_hash:
                warnings.append({
                    "type": "manifest_map_hash_mismatch",
                    "message": (
                        f"manifest map provenance differs for {filename}; "
                        "the actual map hash is included in protocol_hash"),
                    "filename": filename,
                    "manifest_map_hash": expected_hash,
                    "runtime_map_hash": sha256_file(path),
                })

        catalog_hash = self.manifest.metadata.get("catalog_hash")
        catalog_dir_value = self.manifest.metadata.get("catalog_dir")
        if catalog_hash:
            if not catalog_dir_value:
                errors.append("catalog_hash exists without catalog_dir")
            else:
                catalog_dir = Path(str(catalog_dir_value))
                if not catalog_dir.is_absolute():
                    catalog_dir = source_root.parent / catalog_dir
                catalog_file = catalog_dir / "catalog.json"
                if not catalog_file.is_file():
                    warnings.append({
                        "type": "catalog_missing",
                        "message": (
                            "the source catalog is unavailable; the frozen "
                            "scenario embedded in the manifest remains usable"),
                        "catalog_file": str(catalog_file),
                    })
                elif sha256_file(catalog_file) != catalog_hash:
                    warnings.append({
                        "type": "catalog_hash_mismatch",
                        "message": (
                            "catalog provenance differs; reference "
                            "compatibility is decided by protocol_hash"),
                        "manifest_catalog_hash": catalog_hash,
                        "runtime_catalog_hash": sha256_file(catalog_file),
                    })

        if errors:
            raise RuntimeError(
                "experiment integrity check failed: " + "; ".join(errors))
        self.runtime_integrity_warnings = warnings

        selected = set(variant_ids or [])
        validator = SceneValidator(network_dir)
        for variant in self.manifest.variants:
            if selected and variant.variant_id not in selected:
                continue
            experiment_scene = variant.scenario.get(
                "experiment_scene", {})
            assertions = experiment_scene.get("setup_assertions")
            if catalog_hash and not assertions:
                raise RuntimeError(
                    f"{variant.variant_id}: catalog scenario has no frozen "
                    "setup_assertions")
            if assertions:
                validator.validate(variant.scenario, {
                    "schema": experiment_scene.get(
                        "schema", CATALOG_SCHEMA),
                    "scene_id": variant.scenario.get("scenario_id"),
                    "setup_assertions": assertions,
                })

    def run(self, variant_ids: Optional[Iterable[str]] = None) -> dict:
        selected = set(variant_ids or [])
        self._validate_runtime_integrity(selected)
        status = {
            "experiment_id": self.manifest.experiment_id,
            "manifest_hash": self.manifest.manifest_hash,
            "reference_protocol_revision": REFERENCE_PROTOCOL_REVISION,
            "runtime_integrity_warnings": copy.deepcopy(
                self.runtime_integrity_warnings),
            "started_at_epoch_s": time.time(),
            "allow_llm": self.allow_llm,
            "sumo_reference": self.sumo_reference,
            "reference_policy": self.reference_policy,
            "llm_runtime_config": copy.deepcopy(
                self.public_llm_runtime_config),
            "fixed_peer_models": list(self.fixed_peer_models),
            "personal_agent_runtime_config": copy.deepcopy(
                self.public_personal_agent_runtime_config),
            "passenger_judge_runtime_config": copy.deepcopy(
                self.public_passenger_judge_runtime_config),
            "variants": {},
        }
        status_path = self.output_dir / self.status_filename
        for variant in self.manifest.variants:
            if selected and variant.variant_id not in selected:
                continue
            output_path = self.output_dir / f"{variant.variant_id}.json.gz"
            if (variant.requires_llm and not self.allow_llm
                    and not self.sumo_reference):
                status["variants"][variant.variant_id] = {
                    "status": "skipped_llm", "output": None}
            elif self.resume and output_path.exists():
                status["variants"][variant.variant_id] = {
                    "status": "resumed", "output": str(output_path)}
            else:
                payload = None
                try:
                    payload = self._run_variant(variant)
                    _write_gzip_json_atomic(output_path, payload)
                    status["variants"][variant.variant_id] = {
                        "status": payload["run"]["status"],
                        "output": str(output_path),
                        "elapsed_s": payload["run"]["wall_time_s"],
                    }
                except Exception as exc:
                    status["variants"][variant.variant_id] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                    if isinstance(exc, _FailedVariant):
                        status["variants"][variant.variant_id].update({
                            "error": exc.original_error,
                            "diagnostic_output": exc.diagnostic_path,
                        })
                finally:
                    # A run owns a mutable map runtime, callbacks and complete
                    # trajectories.  Large manifests must release cyclic
                    # references between variants instead of relying on an
                    # eventual interpreter-wide collection.
                    payload = None
                    gc.collect()
            _write_json_atomic(status_path, status)
        status["finished_at_epoch_s"] = time.time()
        counts = {}
        for item in status["variants"].values():
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        status["counts"] = counts
        _write_json_atomic(status_path, status)
        return status

    def _run_variant(self, variant: ExperimentVariant) -> dict:
        scenario_payload = copy.deepcopy(variant.scenario)
        scenario = MultiScenario.from_dict(scenario_payload)
        agent_overrides = copy.deepcopy(variant.agent_overrides)
        experiment_scene = variant.scenario.get("experiment_scene", {})
        peer_ids = [
            str(item) for item in experiment_scene.get(
                "fixed_peer_vehicle_ids", [])]
        if self.fixed_peer_runtime_config:
            for peer_id in peer_ids:
                agent_overrides.setdefault(peer_id, {}).update(
                    self.fixed_peer_runtime_config)
        if self.sumo_reference:
            focal_id = str(experiment_scene.get(
                "focal_vehicle_id", ""))
            if self.reference_policy == "focal_sumo_fixed_peers":
                if not focal_id:
                    raise ValueError(
                        "MultiLLM paired reference has no focal_vehicle_id")
                agent_overrides.setdefault(focal_id, {}).update({
                    "type": "sumo",
                })
                for index, peer_id in enumerate(peer_ids):
                    agent_overrides.setdefault(peer_id, {}).update({
                        "type": "llm",
                        "model": self.fixed_peer_models[
                            index % len(self.fixed_peer_models)],
                    })
                for vehicle in scenario.vehicles:
                    if (vehicle.agent_type == "llm"
                            and vehicle.vehicle_id != focal_id
                            and vehicle.vehicle_id not in peer_ids):
                        agent_overrides.setdefault(
                            vehicle.vehicle_id, {}).update({"type": "sumo"})
            else:
                for vehicle in scenario.vehicles:
                    if vehicle.agent_type == "llm":
                        agent_overrides.setdefault(
                            vehicle.vehicle_id, {}).update({"type": "sumo"})
            for pedestrian in scenario.pedestrians:
                if pedestrian.agent_type == "llm":
                    agent_overrides.setdefault(pedestrian.ped_id, {}).update({
                        "type": "sumo",
                    })
        elif self.manifest.experiment_id == "MultiLLM":
            if not self.fixed_peer_models:
                raise ValueError(
                    "MultiLLM runs require at least one fixed peer model")
            for index, peer_id in enumerate(peer_ids):
                agent_overrides.setdefault(peer_id, {}).update({
                    "type": "llm",
                    "model": self.fixed_peer_models[
                        index % len(self.fixed_peer_models)],
                })
        specs = resolve_agent_specs(
            scenario, agent_overrides,
            llm_runtime_config=self.llm_runtime_config)
        apply_resolved_agent_authorities(scenario, specs)
        callbacks, vehicle_meta = build_callbacks(
            specs,
            personal_agent_runtime_config=(
                self.callback_personal_agent_config),
            passenger_judge_runtime_config=(
                self.passenger_judge_runtime_config),
        )
        visual_reference_root = Path("observations") / variant.variant_id
        for entity_id, callback in callbacks.items():
            state = getattr(callback, "_state", None)
            if not isinstance(state, dict) or specs[entity_id].agent_type != "llm":
                continue
            entity_root = visual_reference_root / entity_id
            state["visual_observation_output_dir"] = str(
                self.output_dir / entity_root)
            state["visual_observation_reference_root"] = str(entity_root)
        engine = ExperimentTrackedEngine(scenario)
        web3d_publisher = None
        if self.web3d_runtime_config.get("stream_url"):
            from visualization.web3d_live import Web3DFramePublisher
            web3d_publisher = Web3DFramePublisher(
                **self.web3d_runtime_config)
            engine.add_world_observer(web3d_publisher)
        started = time.perf_counter()
        try:
            result = engine.run(callbacks)
        except Exception as exc:
            original_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            try:
                physics_evidence = engine.failure_snapshot()
            except Exception as snapshot_error:
                # Broken telemetry must not mask the original physics error
                # or prevent callback evidence from being retained.
                physics_evidence = {"snapshot_error": str(snapshot_error)}
            # Separate, uniquely named artifacts cannot be mistaken for a
            # completed run by resume/export; retries retain prior evidence.
            diagnostic_path = (self.output_dir / "failures"
                               / f"{variant.variant_id}.{time.time_ns()}.json.gz")
            evidence_keys = (
                "prompt_profile_id", "instruction", "effective_config",
                "initial_tools", "tools", "tool_call_log", "model_call_log",
                "wake_runtime_log", "visual_observation_log", "all_messages",
                "context_snapshot_log", "infrastructure_errors",
                "personal_requests", "passenger_evaluations",
                "personal_agent", "passenger_judge", "todo_state", "todo_audit_log",
            )
            evidence = {
                "schema": "vehiclearena-failed-experiment-v1",
                "experiment_id": self.manifest.experiment_id,
                "manifest_hash": self.manifest.manifest_hash,
                "source_hash": self.manifest.source_hash,
                "variant": variant.to_dict(),
                "run": {"status": "failed", "partial": True,
                        "evaluated": False, "error": original_error,
                        "traceback": failure_traceback,
                        "wall_time_s": round(time.perf_counter() - started, 6),
                        "llm_runtime_config": self.public_llm_runtime_config,
                        "personal_agent_runtime_config": self.public_personal_agent_runtime_config,
                        "passenger_judge_runtime_config": self.public_passenger_judge_runtime_config},
                "runtime_integrity_warnings": copy.deepcopy(
                    self.runtime_integrity_warnings),
                **physics_evidence,
                "agent_metadata": vehicle_meta,
                "callbacks": {entity_id: {
                    key: getattr(callback, "_state", {}).get(key)
                    for key in evidence_keys}
                    for entity_id, callback in callbacks.items()},
            }
            try:
                _write_gzip_json_atomic(diagnostic_path, _public_runtime_config(evidence))
            except Exception as save_error:
                exc.add_note(f"Failed to save partial trajectory: {save_error}")
                raise exc
            raise _FailedVariant(original_error, diagnostic_path) from exc
        elapsed = time.perf_counter() - started
        for entity_id, callback in callbacks.items():
            state = getattr(callback, "_state", {})
            evaluations = state.get("passenger_evaluations")
            if (evaluations is not None
                    and entity_id in result.vehicle_results):
                result.vehicle_results[entity_id].passenger_evaluation = (
                    aggregate_passenger_judgements(
                        evaluations,
                        request_count=len(
                            state.get("personal_requests", []))))
        vehicle_trajectory = getattr(result, "_vehicle_trajectory", [])
        pedestrian_trajectory = getattr(
            result, "_pedestrian_trajectory", [])
        evaluated_vehicle_ids = [
            config.vehicle_id for config in scenario.vehicles
            if config.is_evaluated]
        agent_provenance = {}
        conversations = {}
        context_snapshots = {}
        model_call_log = {}
        wake_runtime_log = {}
        visual_observations = {}
        callback_infrastructure_errors = []
        for entity_id, callback in callbacks.items():
            state = getattr(callback, "_state", {})
            callback_infrastructure_errors.extend(
                state.get("infrastructure_errors", []))
            agent_provenance[entity_id] = {
                "prompt_profile_id": state.get("prompt_profile_id"),
                "instruction": state.get("instruction"),
                "instruction_hash": (
                    sha256_data(state["instruction"])
                    if state.get("instruction") else None),
                "initial_tool_schema_hash": (
                    sha256_data(state["initial_tools"])
                    if state.get("initial_tools") else None),
                "final_tool_schema_hash": (
                    sha256_data(state["tools"])
                    if state.get("tools") else None),
                "model": state.get("model", ""),
                "effective_config": state.get("effective_config", {}),
                "initial_tool_schema": state.get("initial_tools"),
                "final_tool_schema": state.get("tools"),
                "loaded_skills": state.get("loaded_skills", []),
                "protocol_events": state.get("protocol_events", []),
                "context_data_registry": state.get(
                    "context_data_registry", {}),
                "context_budget_log": state.get(
                    "context_budget_log", []),
                "capability_epoch": state.get("capability_epoch", 0),
                "todo_state": state.get("todo_state", {}),
                "todo_audit_log": state.get("todo_audit_log", []),
                "heartbeat_interval_s": state.get(
                    "heartbeat_interval_s"),
                "heartbeat_interval_revision": state.get(
                    "heartbeat_interval_revision", 0),
                "heartbeat_interval_audit_log": state.get(
                    "heartbeat_interval_audit_log", []),
                "scheduled_wake_audit_log": state.get(
                    "scheduled_wake_audit_log", []),
                "personal_agent": copy.deepcopy(
                    state.get("personal_agent")),
                "passenger_judge": copy.deepcopy(
                    state.get("passenger_judge")),
                "passenger_evaluations": copy.deepcopy(
                    state.get("passenger_evaluations", [])),
            }
            conversations[entity_id] = state.get("all_messages", [])
            context_snapshots[entity_id] = state.get(
                "context_snapshot_log", [])
            role_calls = [
                {"role": "vehicle_agent", **copy.deepcopy(item)}
                for item in state.get("model_call_log", [])
            ]
            for role_key in ("personal_agent", "passenger_judge"):
                role_calls.extend(copy.deepcopy(
                    (state.get(role_key) or {}).get(
                        "model_call_log", [])))
            model_call_log[entity_id] = role_calls
            wake_runtime_log[entity_id] = state.get("wake_runtime_log", [])
            visual_observations[entity_id] = state.get(
                "visual_observation_log", [])
        engine_errors = getattr(result, "_agent_callback_errors", [])
        infrastructure_engine_errors = [
            item for item in engine_errors
            if item.get("phase") in ("callback", "command_commit")]
        run_status = (
            "invalid_infrastructure"
            if callback_infrastructure_errors or infrastructure_engine_errors
            else "completed")
        variant_payload = variant.to_dict()
        result_payload = result.to_dict()
        network_id = str(scenario_payload.get("road_network_id", ""))
        network_path = (Path(__file__).resolve().parents[2]
                        / "simulation" / "road_networks"
                        / f"{network_id}_lane_level.json")
        reference_protocol = build_reference_protocol(
            experiment_id=self.manifest.experiment_id,
            variant=variant_payload,
            network_path=network_path,
            physics_engine=result_payload.get("physics_engine", {}),
            resolved_specs=specs,
        )
        return {
            "schema": "vehiclearena-experiment-run-v0.6",
            "experiment_id": self.manifest.experiment_id,
            "manifest_hash": self.manifest.manifest_hash,
            "source_hash": self.manifest.source_hash,
            "protocol_hash": protocol_hash(reference_protocol),
            "reference_protocol": reference_protocol,
            "variant": variant_payload,
            "run": {
                "status": run_status,
                "wall_time_s": round(elapsed, 6),
                "python": sys.version,
                "platform": platform.platform(),
                "trajectory_step_s": scenario.physics_step_s,
                "llm_runtime_config": copy.deepcopy(
                    self.public_llm_runtime_config),
                "fixed_peer_models": list(self.fixed_peer_models),
                "sumo_reference": self.sumo_reference,
                "reference_policy": self.reference_policy,
                "personal_agent_runtime_config": copy.deepcopy(
                    self.public_personal_agent_runtime_config),
                "passenger_judge_runtime_config": copy.deepcopy(
                    self.public_passenger_judge_runtime_config),
                "runtime_integrity_warnings": copy.deepcopy(
                    self.runtime_integrity_warnings),
                "trajectory_hash": sha256_data({
                    "vehicles": vehicle_trajectory,
                    "pedestrians": pedestrian_trajectory,
                }),
                "web3d": (
                    {
                        "session_id": web3d_publisher.session_id,
                        "published_frame_count": (
                            web3d_publisher.published_count),
                        "dropped_frame_count": web3d_publisher.dropped_count,
                        "failed_frame_count": web3d_publisher.failed_count,
                        "last_error": web3d_publisher.last_error or None,
                        "observer_errors": copy.deepcopy(
                            engine.world_observer_errors),
                    }
                    if web3d_publisher is not None else None),
            },
            "result": result_payload,
            "system_evaluation": evaluate_system(
                result, vehicle_trajectory, scenario.total_time_s,
                scenario=scenario_payload),
            "pedestrian_evaluation": evaluate_pedestrians(
                result, pedestrian_trajectory, vehicle_trajectory),
            "traffic_calibration": evaluate_traffic_calibration(
                vehicle_trajectory, getattr(result, "_collision_log", [])),
            "chassis_calibration": evaluate_chassis_calibration(
                vehicle_trajectory, evaluated_vehicle_ids),
            "capability_calibration": evaluate_capability_calibration(result),
            "agent_metadata": vehicle_meta,
            "agent_provenance": agent_provenance,
            "agent_callback_errors": engine_errors,
            "agent_infrastructure_errors": (
                callback_infrastructure_errors
                + infrastructure_engine_errors),
            "tool_call_log": callback_tool_logs(callbacks),
            "model_call_log": model_call_log,
            "wake_runtime_log": wake_runtime_log,
            "model_runtime_summary": _runtime_summary(
                model_call_log, wake_runtime_log),
            "visual_observations": visual_observations,
            "agent_conversations": conversations,
            "agent_context_snapshots": context_snapshots,
            "event_audit": getattr(result, "_event_audit", {
                "summary": {}, "agent_visible_events": [],
                "evaluator_only_events": [], "delivery_batches": [],
            }),
            "physics_events": getattr(result, "_physics_events", []),
            "collision_log": getattr(result, "_collision_log", []),
            "trajectories": {
                "vehicles": vehicle_trajectory,
                "pedestrians": pedestrian_trajectory,
            },
        }


def run_manifest(
    manifest_path: Path, output_dir: Path, *, allow_llm: bool = False,
    resume: bool = True, variant_ids: Optional[Iterable[str]] = None,
    status_filename: str = "status.json",
    llm_runtime_config: Optional[dict] = None,
    fixed_peer_models: Optional[list[str]] = None,
    fixed_peer_runtime_config: Optional[dict] = None,
    personal_agent_runtime_config: Optional[dict] = None,
    passenger_judge_runtime_config: Optional[dict] = None,
    web3d_runtime_config: Optional[dict] = None,
    sumo_reference: bool = False,
) -> dict:
    return ExperimentBatchRunner(
        load_manifest(manifest_path), output_dir,
        allow_llm=allow_llm, resume=resume,
        status_filename=status_filename,
        llm_runtime_config=llm_runtime_config,
        fixed_peer_models=fixed_peer_models,
        fixed_peer_runtime_config=fixed_peer_runtime_config,
        personal_agent_runtime_config=personal_agent_runtime_config,
        passenger_judge_runtime_config=passenger_judge_runtime_config,
        web3d_runtime_config=web3d_runtime_config,
        sumo_reference=sumo_reference,
    ).run(variant_ids)


def aggregate_run_directory(output_dir: Path) -> dict:
    """Create a flat, analysis-ready table without discarding raw runs."""
    rows = []
    for path in sorted(Path(output_dir).glob("*.json.gz")):
        payload = load_run(path)
        system = payload["system_evaluation"]
        result = payload["result"]
        factors = payload["variant"]["factors"]
        sumo_reference = bool(payload.get("run", {}).get(
            "sumo_reference", False))
        authority = (
            "sumo" if sumo_reference else
            "llm" if payload["variant"].get("requires_llm", False)
            else "sumo")
        traffic = payload.get(
            "traffic_calibration", {}).get("authorities", {}).get(
                authority, {})
        pedestrian = payload.get(
            "pedestrian_evaluation", {}).get(
                "authority_summary", {}).get("sumo", {})
        focal_pedestrian = payload.get(
            "pedestrian_evaluation", {}).get("pedestrians", {}).get(
                "pedestrian", {})
        chassis_vehicles = payload.get(
            "chassis_calibration", {}).get("vehicles", {})
        chassis = next(iter(chassis_vehicles.values()), {})
        capability_vehicles = payload.get(
            "capability_calibration", {}).get("vehicles", {})
        capability = next(iter(capability_vehicles.values()), {})
        scenario = payload.get("variant", {}).get("scenario", {}) or {}
        experiment_scene = scenario.get("experiment_scene", {}) or {}
        evaluated_ids = [
            vehicle_id for vehicle_id, vehicle
            in result.get("vehicles", {}).items()
            if vehicle.get("is_evaluated")]
        evaluated_reports_by_id = {
            vehicle_id: vehicle.get("driving_evaluation")
            for vehicle_id, vehicle in result.get("vehicles", {}).items()
            if vehicle.get("is_evaluated")
            and vehicle.get("driving_evaluation")
        }
        focal_id = str(
            experiment_scene.get("focal_vehicle_id")
            or factors.get("focal_vehicle_id")
            or system.get("focal_vehicle_id")
            or ""
        )
        focal_vehicle = next(
            (vehicle for vehicle in scenario.get("vehicles", [])
             if str(vehicle.get("vehicle_id")) == focal_id),
            None,
        )
        # A MultiLLM task has several evaluated entities, but only the
        # declared focal vehicle is the subject of the task-level score,
        # including when a SUMO reference replaces the focal LLM.
        focal_is_declared_evaluated_subject = (
            focal_vehicle is not None
            and focal_vehicle.get("is_evaluated", True)
            and (sumo_reference or str((focal_vehicle.get(
                "agent_config", {}) or {}).get("type", "")).lower() == "llm"))
        focal_has_driving_report = (
            focal_is_declared_evaluated_subject
            and focal_id in evaluated_reports_by_id)
        passenger, all_agent_passenger = _task_passenger_summary(
            result, focal_id=focal_id,
            focal_only=focal_is_declared_evaluated_subject)
        driving_score_reports = (
            [evaluated_reports_by_id[focal_id]]
            if focal_has_driving_report else list(evaluated_reports_by_id.values())
        )
        evaluated_driving = list(evaluated_reports_by_id.values())
        evaluated_configs = [
            vehicle for vehicle in scenario.get("vehicles", [])
            if vehicle.get("vehicle_id") in evaluated_ids]
        first_config = next(iter(evaluated_configs), {})
        agent_config = first_config.get("agent_config", {}) or {}
        if sumo_reference:
            policy_family = str(payload.get("run", {}).get(
                "reference_policy") or "all_sumo")
        elif agent_config.get("type") == "llm":
            first_id = next(iter(evaluated_ids), "")
            policy_family = (
                payload.get("agent_provenance", {}).get(
                    first_id, {}).get("model")
                or agent_config.get("model") or "llm_unspecified")
        else:
            policy_family = "sumo_native"
        rational_episode_count = sum(
            int(item.get("metrics", {}).get(
                "rational_episode_count", 0))
            for item in evaluated_driving)
        rational_episode_passed = sum(
            int(item.get("metrics", {}).get(
                "rational_episode_passed", 0))
            for item in evaluated_driving)
        single_vehicle_scores = [
            single_vehicle_layer_score_100(item)
            for item in driving_score_reports
        ]
        single_vehicle_scores = [
            score for score in single_vehicle_scores if score is not None
        ]
        cabin_score, cabin_task_completion_rate = _task_cabin_summary(
            result, focal_id=focal_id,
            focal_only=focal_is_declared_evaluated_subject)
        reasonable_flags = [
            item.get("reasonable_driving_pass")
            for item in driving_score_reports
            if item.get("reasonable_driving_pass") is not None
        ]
        # Derive the arrival denominator from the frozen focal-vehicle
        # declaration, rather than from whichever result happened to be
        # serialised. A missing focal result counts as an arrival failure.
        # ``llm_vehicle_ids`` describes every LLM-controlled entity in the
        # scenario.  For the task-level arrival metric, MultiLLM evaluates
        # only the focal vehicle; fixed-peer LLMs are interaction actors and
        # must not make the task fail merely because they miss their route.
        llm_vehicle_ids = [
            str(vehicle.get("vehicle_id"))
            for vehicle in scenario.get("vehicles", [])
            if vehicle.get("is_evaluated", True)
            and str((vehicle.get("agent_config", {}) or {}).get(
                "type", "")).lower() == "llm"
            and vehicle.get("vehicle_id")
        ]
        llm_arrival_vehicle_ids = (
            [focal_id]
            if focal_vehicle is not None
            and focal_vehicle.get("is_evaluated", True)
            and str((focal_vehicle.get("agent_config", {}) or {}).get(
                "type", "")).lower() == "llm"
            else list(llm_vehicle_ids)
            if len(llm_vehicle_ids) == 1
            else []
        )
        llm_arrival_success = None
        llm_arrival_failure_reasons = []
        if llm_arrival_vehicle_ids and not sumo_reference:
            deadline_s = float(system.get(
                "frozen_duration_s", scenario.get("total_time_s", 0.0)) or 0.0)
            for vehicle_id in llm_arrival_vehicle_ids:
                vehicle_result = result.get("vehicles", {}).get(
                    vehicle_id, {})
                arrived = bool(vehicle_result.get("arrived", False))
                route_failed = bool(vehicle_result.get("route_failed", False))
                arrival_time_s = vehicle_result.get("arrival_time_s")
                within_deadline = (
                    arrived and not route_failed
                    and isinstance(arrival_time_s, (int, float))
                    and float(arrival_time_s) <= deadline_s + 1e-9)
                if not within_deadline:
                    llm_arrival_failure_reasons.append(
                        f"{vehicle_id}:timeout_or_route_failure")
            collision_log = payload.get("collision_log", []) or []
            llm_collision = any(
                event.get("entity_a") in llm_arrival_vehicle_ids
                or event.get("entity_b") in llm_arrival_vehicle_ids
                for event in collision_log)
            # Keep the per-vehicle evaluator as a fallback for older artifacts
            # that predate a persisted collision log.
            llm_collision = llm_collision or any(
                int((result.get("vehicles", {}).get(vehicle_id, {})
                     .get("driving_evaluation", {}) or {})
                    .get("metrics", {}).get("collision_count", 0) or 0) > 0
                for vehicle_id in llm_arrival_vehicle_ids)
            if llm_collision:
                llm_arrival_failure_reasons.append("collision_involving_llm")
            llm_arrival_success = not llm_arrival_failure_reasons
        environment_scores = [
            report.get("driving_process", {}).get("environment_accuracy")
            for report in evaluated_reports_by_id.values()
            if isinstance(report.get("driving_process"), dict)
            and report.get("driving_process", {}).get(
                "environment_accuracy") is not None
        ]
        rows.append({
            "experiment_id": payload["experiment_id"],
            "variant_id": payload["variant"]["variant_id"],
            "base_scenario_id": payload["variant"]["base_scenario_id"],
            "requires_llm": bool(payload["variant"].get(
                "requires_llm", False)),
            "run_valid": payload.get("run", {}).get("status") == "completed",
            "policy_family": policy_family,
            "task_completed": (
                all(item.get("task_completed", False)
                    for item in evaluated_driving)
                if evaluated_driving else None),
            "llm_vehicle_ids": llm_vehicle_ids,
            "llm_arrival_vehicle_ids": llm_arrival_vehicle_ids,
            "llm_arrival_success": llm_arrival_success,
            "llm_arrival_rate": (
                1.0 if llm_arrival_success else 0.0
                if llm_arrival_success is not None else None),
            "llm_arrival_failure_reasons": llm_arrival_failure_reasons,
            "declared_difficulty": scenario.get("difficulty", "controlled"),
            **factors,
            "overall_cabin_score": cabin_score,
            "cabin_layer_score_100": cabin_layer_score_100(cabin_score),
            "request_satisfaction_score_100": passenger.get(
                "overall_score_100"),
            "request_satisfaction_metric_revision": passenger.get(
                "metric_revision"),
            "request_satisfaction_scored_count": passenger.get(
                "scored_count", 0),
            "request_satisfaction_request_count": passenger.get(
                "request_count", 0),
            "request_satisfaction_evaluable_rate": passenger.get(
                "evaluable_rate"),
            "request_satisfaction_excluded_rate": passenger.get(
                "excluded_rate"),
            "request_satisfaction_score_coverage_rate": passenger.get(
                "score_coverage_rate"),
            "request_satisfaction_coverage_adjusted_score_100": passenger.get(
                "coverage_adjusted_score_100"),
            "request_satisfaction_completion_counts": passenger.get(
                "completion_counts", {}),
            "request_satisfaction_na_count": passenger.get("na_count", 0),
            "request_satisfaction_grade_counts": passenger.get(
                "grade_counts", {}),
            "all_agent_request_satisfaction_score_100": (
                all_agent_passenger.get("overall_score_100")),
            "all_agent_request_satisfaction_scored_count": (
                all_agent_passenger.get("scored_count", 0)),
            "all_agent_request_satisfaction_request_count": (
                all_agent_passenger.get("request_count", 0)),
            "all_agent_request_satisfaction_score_coverage_rate": (
                all_agent_passenger.get("score_coverage_rate")),
            "all_agent_request_satisfaction_coverage_adjusted_score_100": (
                all_agent_passenger.get("coverage_adjusted_score_100")),
            "all_agent_request_satisfaction_na_count": (
                all_agent_passenger.get("na_count", 0)),
            "all_agent_request_satisfaction_grade_counts": (
                all_agent_passenger.get("grade_counts", {})),
            "cabin_task_completion_rate": cabin_task_completion_rate,
            "trajectory_quality_score": result["evaluation"]["driving"][
                "trajectory_quality_score"],
            "single_vehicle_layer_score_100": (
                round(sum(single_vehicle_scores)
                      / len(single_vehicle_scores), 2)
                if single_vehicle_scores else None),
            "single_vehicle_score_applicable_count": len(
                single_vehicle_scores),
            "single_vehicle_score_evaluated_count": len(
                driving_score_reports),
            "single_vehicle_score_coverage": (
                len(single_vehicle_scores) / len(driving_score_reports)
                if driving_score_reports else None),
            "environment_accuracy": (
                round(sum(environment_scores) / len(environment_scores), 4)
                if environment_scores and not sumo_reference else None),
            "environment_acc": (
                round(sum(environment_scores) / len(environment_scores), 4)
                if environment_scores and not sumo_reference else None),
            "hard_safety_passed": result["evaluation"]["driving"][
                "hard_safety_passed"],
            "reasonable_driving_vehicle_rate": (
                sum(bool(value) for value in reasonable_flags)
                / len(reasonable_flags)
                if reasonable_flags else None),
            "rational_episode_count": rational_episode_count,
            "rational_episode_rate": (
                rational_episode_passed / rational_episode_count
                if rational_episode_count else None),
            "arrival_rate": system["arrival_rate"],
            "throughput_per_min": system["throughput_per_min"],
            "collision_count": system["collision_count"],
            "secondary_collision_count": system[
                "secondary_collision_count"],
            "wait_mean_s": system["stationary_wait_mean_s"],
            "wait_p90_s": system["stationary_wait_p90_s"],
            "wait_gini": system["stationary_wait_gini"],
            "unexplained_wait_mean_s": system[
                "unexplained_wait_mean_s"],
            "max_queue_length": system["max_queue_length"],
            "candidate_deadlock_count": len(system["candidate_deadlocks"]),
            "confirmed_deadlock_count": len(system.get(
                "confirmed_deadlocks", [])),
            "queue_wait_mean_s": system.get("queue_wait_mean_s"),
            "required_trip_vehicle_ids": system.get(
                "required_trip_vehicle_ids"),
            "map_trip_vehicle_ids": system.get("map_trip_vehicle_ids"),
            "focal_vehicle_id": system.get("focal_vehicle_id"),
            "all_trip_vehicles_arrived": system.get(
                "all_trip_vehicles_arrived"),
            "trip_vehicle_metrics": system.get("trip_vehicle_metrics"),
            "scenario_hash": payload.get("variant", {}).get(
                "scenario_hash"),
            "protocol_hash": payload.get("protocol_hash"),
            "reference_protocol_revision": payload.get(
                "reference_protocol", {}).get("revision"),
            "source_hash": payload.get("source_hash"),
            "manifest_hash": payload.get("manifest_hash"),
            "reference_policy": payload.get("run", {}).get(
                "reference_policy"),
            "fixed_peer_models": payload.get("run", {}).get(
                "fixed_peer_models", []),
            "wall_time_s": payload["run"]["wall_time_s"],
            "traffic_mean_speed_kmh": traffic.get("mean_speed_kmh"),
            "traffic_stopped_ratio": traffic.get("stopped_ratio"),
            "traffic_collision_involved_vehicle_count": traffic.get(
                "collision_involved_vehicle_count"),
            "pedestrian_mean_waiting_time_s": pedestrian.get(
                "mean_waiting_time_s"),
            "pedestrian_mean_crosswalk_time_s": pedestrian.get(
                "mean_crosswalk_time_s"),
            "pedestrian_mean_movement_speed_mps": pedestrian.get(
                "mean_movement_speed_mps"),
            "pedestrian_mean_first_crossing_entry_s": pedestrian.get(
                "mean_first_crossing_entry_s"),
            "focal_pedestrian_waiting_time_s": focal_pedestrian.get(
                "waiting_time_s"),
            "focal_pedestrian_crosswalk_time_s": focal_pedestrian.get(
                "crosswalk_time_s"),
            "focal_pedestrian_movement_speed_mps": focal_pedestrian.get(
                "mean_movement_speed_mps"),
            "focal_pedestrian_min_vehicle_distance_m": focal_pedestrian.get(
                "min_vehicle_center_distance_m"),
            "chassis_peak_speed_kmh": chassis.get("peak_speed_kmh"),
            "chassis_max_braking_mps2": chassis.get("max_braking_mps2"),
            "chassis_first_braking_time_s": chassis.get(
                "first_braking_time_s"),
            "chassis_first_braking_distance_m": chassis.get(
                "first_braking_distance_m"),
            "chassis_lane_change_time_s": chassis.get(
                "lane_change_time_s"),
            "capability_probe_count": capability.get(
                "capability_probe_count"),
            "capability_cabin_applicable": capability.get(
                "cabin_applicable"),
        })
    report = {
        "schema": "vehiclearena-experiment-aggregate-v1",
        "run_count": len(rows),
        "rows": rows,
    }
    llm_rows = [
        row for row in rows if row.get("llm_arrival_success") is not None
    ]
    report["llm_task_count"] = len(llm_rows)
    report["llm_arrival_success_count"] = sum(
        bool(row.get("llm_arrival_success")) for row in llm_rows)
    report["llm_arrival_rate"] = (
        report["llm_arrival_success_count"] / len(llm_rows)
        if llm_rows else None)
    environment_rows = [
        row["environment_accuracy"] for row in rows
        if row.get("environment_accuracy") is not None
    ]
    report["environment_accuracy_mean"] = (
        sum(environment_rows) / len(environment_rows)
        if environment_rows else None)
    _write_json_atomic(Path(output_dir) / "aggregate.json", report)
    return report
