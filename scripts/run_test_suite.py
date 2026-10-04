#!/usr/bin/env python3
"""Run the 112-task calibrated test split with a 2:1 active-study schedule.

The test split is frozen in
``vehiclearena/evaluation/experiments/scenarios/qwen38-full-200-npc-causal-20260913-test-scenes.json``
and contains 80 Basic plus the MultiLLM scenes with completed time-window
calibration (currently 32).  Unlike submitting all Basic
jobs first, this wrapper asks ``run_llm_suite.py`` to keep the active scene
workers close to Basic:MultiLLM = 2:1.  ``--workers`` is the total number of
scene workers across both studies.

Credentials are still read from ``OPENAI_API_KEY`` by the underlying runner.
Model presets select model names and reasoning settings only; callers provide
their own ``--base-url``. All Personal Agent/Judge options remain available
through the underlying runner.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_llm_suite.py"
DEFAULT_SELECTION = (
    ROOT / "vehiclearena/evaluation/experiments/scenarios"
    / "qwen38-full-200-npc-causal-20260913-test-scenes.json"
)
DEFAULT_REGISTRY = ROOT / "vehiclearena/evaluation/experiments/time_window_calibration.json"
MODEL_PRESETS = {
    "qwen3.8-27b": {
        "model": "Qwen3.8-27B",
    },
    "kimi-k3": {
        "model": "Kimi-K3",
        "reasoning_effort": "max",
    },
    "qwen3-vl-8b-instruct": {
        "model": "Qwen3-VL-8B-Instruct",
    },
    "qwen3-vl-32b-instruct": {
        "model": "Qwen3-VL-32B-Instruct",
    },
    "gpt-5.6-sol": {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
    },
}
TEST_BASIC_COUNT = 80
TEST_MULTILLM_COUNT = 32
TEST_SCENE_COUNT = TEST_BASIC_COUNT + TEST_MULTILLM_COUNT


def _selection_path(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--selection-file", type=Path,
                        default=DEFAULT_SELECTION)
    args, _ = parser.parse_known_args(argv)
    return args.selection_file.resolve()


def _model_preset(argv: list[str]) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS))
    args, _ = parser.parse_known_args(argv)
    return args.model_preset


def _load_scene_ids(path: Path) -> list[str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read selection file {path}: {exc}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON selection file {path}: {exc}")
    if not isinstance(raw, list):
        raise SystemExit("the selection file must contain a JSON list")
    try:
        ids = [item if isinstance(item, str) else item["scene_id"]
               for item in raw]
    except (KeyError, TypeError):
        raise SystemExit(
            "selection entries must be scene IDs or objects containing scene_id")
    if len(ids) != len(set(ids)):
        raise SystemExit("selection file contains duplicate scene IDs")
    counts = Counter(
        "Basic" if scene_id.startswith("basic_")
        else "MultiLLM" if scene_id.startswith("multi_")
        else "unknown"
        for scene_id in ids
    )
    expected = {"Basic": TEST_BASIC_COUNT, "MultiLLM": TEST_MULTILLM_COUNT}
    if (len(ids) != TEST_SCENE_COUNT
            or {key: counts[key] for key in expected} != expected):
        raise SystemExit(
            f"the calibrated test split must contain exactly {TEST_BASIC_COUNT} "
            f"Basic and {TEST_MULTILLM_COUNT} MultiLLM scenes; "
            f"got {dict(counts)}")
    if counts["unknown"]:
        raise SystemExit(
            f"selection file contains {counts['unknown']} unknown study IDs")
    return ids


def main() -> int:
    argv = sys.argv[1:]
    selection = _selection_path(argv)
    _load_scene_ids(selection)
    preset_name = _model_preset(argv)
    # This entry point owns the scheduling policy.  Do not let a forwarded
    # --study-ratio silently override the required 2:1 ratio.
    forwarded = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == "--study-ratio":
            skip_next = True
            continue
        if item.startswith("--study-ratio="):
            continue
        if item == "--model-preset":
            skip_next = True
            continue
        if item.startswith("--model-preset="):
            continue
        forwarded.append(item)
    has_registry = any(
        item == "--calibration-registry"
        or item.startswith("--calibration-registry=")
        for item in argv
    )
    has_provisional_override = "--allow-provisional-time-windows" in argv
    has_custom_catalog = any(
        item == "--catalog" or item.startswith("--catalog=")
        for item in argv
    )
    calibration_args = []
    if not has_registry and not has_provisional_override and not has_custom_catalog:
        calibration_args = ["--calibration-registry", str(DEFAULT_REGISTRY)]
    model_args = []
    if preset_name:
        preset = MODEL_PRESETS[preset_name]
        model_args = ["--model", preset["model"]]
        has_reasoning_effort = any(
            item == "--reasoning-effort"
            or item.startswith("--reasoning-effort=")
            for item in argv
        )
        if preset.get("reasoning_effort") and not has_reasoning_effort:
            model_args.extend(["--reasoning-effort", preset["reasoning_effort"]])
    command = [
        sys.executable,
        str(RUNNER),
        "--selection-file", str(selection),
        "--study-ratio", "2:1",
        "--prepare",
        *model_args,
        *calibration_args,
        *forwarded,
    ]
    # Replace this wrapper with the controller so signals and exit status are
    # delivered directly to the existing suite implementation.
    os.execv(sys.executable, command)
    return 0  # pragma: no cover - os.execv never returns


if __name__ == "__main__":
    raise SystemExit(main())
