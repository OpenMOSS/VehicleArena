"""Versioned, auditable calibration parameters for experiment authoring."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PARAMETER_SCHEMA = "vehiclearena-experiment-parameters-v2"
PARAMETER_PATH = Path(__file__).with_name("experiment_parameters.json")


def _validate(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != PARAMETER_SCHEMA:
        raise ValueError(
            f"unsupported experiment parameter schema: {payload.get('schema')!r}")
    map_coverage = payload.get("map_coverage")
    if not isinstance(map_coverage, Mapping):
        raise ValueError("experiment parameter sections must be objects")
    angle = float(map_coverage["minimum_connector_conflict_angle_deg"])
    if not 0.0 < angle <= 90.0:
        raise ValueError(
            "map_coverage.minimum_connector_conflict_angle_deg must be in (0, 90]")


def load_experiment_parameters() -> dict:
    payload = json.loads(PARAMETER_PATH.read_text(encoding="utf-8"))
    _validate(payload)
    return copy.deepcopy(payload)


def parameter_fingerprint() -> str:
    canonical = json.dumps(
        load_experiment_parameters(), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
