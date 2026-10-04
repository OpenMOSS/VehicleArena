"""Stable 0--100 conversions for VehicleArena's cabin and driving layers.

The layer scores are deliberately independent and are reported alongside
task arrival and NPC traffic-impact metrics.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _bounded(value: Any, lower: float = 0.0,
             upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def cabin_layer_score_100(cabin_score: Optional[float]) -> Optional[float]:
    """Return the YAML cabin field score on a 0--100 scale.

    ``None`` means that the scenario contained no applicable cabin rule.
    """
    if cabin_score is None:
        return None
    return round(100.0 * _bounded(cabin_score), 2)


def single_vehicle_layer_score_100(
        driving_report: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Return the current hard-gated driving-process score.

    The process score is persisted either inside ``driving_process`` or at
    the report root. Reports without that field are not scored; rational
    episode coverage is retained as diagnostic evidence only.
    """
    if not driving_report:
        return None
    process_report = driving_report.get("driving_process")
    if isinstance(process_report, Mapping):
        process_score = process_report.get("driving_process_score_100")
        if process_score is not None:
            return round(_bounded(process_score, 0.0, 100.0), 2)
    if driving_report.get("driving_process_score_100") is not None:
        return round(_bounded(
            driving_report["driving_process_score_100"], 0.0, 100.0), 2)
    return None
