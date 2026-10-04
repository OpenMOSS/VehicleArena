"""
Snapshot-based verification for VehicleWorld evaluation.

Verification is snapshot-based and does not execute checks.json expressions.
fully automatic: execute ground-truth actions on a reference VehicleWorld,
snapshot the state, then compare against the agent's VehicleWorld after it
has acted.

The accuracy denominator is the number of fields that the ground-truth
actions actually *change* (not all fields in the module).  This avoids
inflating the denominator with hundreds of untouched default values.

Usage in eval pipelines
-----------------------
    from check_utils import build_expected_snapshot, verify_by_snapshot

    # Build expected state (once per task, can be cached)
    expected_snap, changed_fields = build_expected_snapshot(
        inits_code, ground_truth_code, modules=["fogLight", "wiper"]
    )

    # After agent acts on vw_agent ...
    result = verify_by_snapshot(changed_fields, vw_agent, modules=["fogLight", "wiper"])
"""

import sys
sys.path.append('../')

from vehiclearena import VehicleWorld
from utils import execute
from evaluation.snapshot_utils import (
    snapshot_all,
    snapshot_modules,
    deep_diff,
    get_by_path,
)


_FULL_INIT_PREFIX = (
    "from module import *\n"
    "from module.vehicle_settings import VehicleSettings\n"
    "from module.weather import Weather\n"
    "from module.trafficlight import TrafficLight\n"
    "from module.road import Road\n"
    "from module.speedlimit import SpeedLimit\n"
    "from module.daynight import DayNight\n"
    "from module.map_module import MapModule\n"
)


def _match_announcements(expected: list, actual: list) -> bool:
    """Check that each expected broadcast announcement exists in actual.

    For each expected announcement, checks that at least one actual
    announcement has all the same key-value pairs (subset match).
    The agent may have extra broadcasts — that's fine.

    This handles the common case where the agent broadcasts correctly
    but also adds additional helpful broadcasts.
    """
    def _is_subset(expected_ann, actual_ann):
        """True if every key in expected_ann matches actual_ann.

        Special cases:
        - safety_refusal: reason only needs to be non-empty (free-text).
        - warning: message is auto-generated from warning_type (skip to
          avoid double-checking); priority is subjective (skip).
        """
        for k, v in expected_ann.items():
            actual_v = actual_ann.get(k)
            # For safety_refusal, reason just needs to be a non-empty string
            if (k == "reason"
                    and expected_ann.get("category") == "safety_refusal"):
                if not actual_v or not str(actual_v).strip():
                    return False
                continue
            # For warnings, message is derived from warning_type (redundant)
            # and priority is subjective — skip both.
            if (k in ("message", "priority")
                    and expected_ann.get("category") == "warning"):
                continue
            if actual_v != v:
                return False
        return True

    # Track which actual announcements are already matched
    used = set()
    for exp in expected:
        found = False
        for i, act in enumerate(actual):
            if i not in used and _is_subset(exp, act):
                used.add(i)
                found = True
                break
        if not found:
            return False
    return True


def build_expected_snapshot(inits_code: str, ground_truth_code: str,
                            modules: list = None):
    """Create two VehicleWorlds (init-only vs init+ground-truth), diff them.

    Args:
        inits_code:        Initialisation code.
        ground_truth_code: Ground-truth API calls.
        modules:           If given, only snapshot these modules.

    Returns:
        (expected_snap, changed_fields)
        - expected_snap:  Full snapshot dict after ground-truth execution.
        - changed_fields: ``deep_diff(snap_init, snap_gt)`` -- only the leaf
          paths that the ground-truth actions actually change.  Each entry is
          ``{path: {"expected": <init_val>, "actual": <gt_val>}}``.
    """
    full_init = _FULL_INIT_PREFIX + inits_code

    # 1) Init-only VehicleWorld
    vw_init = VehicleWorld()
    execute(full_init, local_vars={'vw': vw_init}, global_vars=None)

    # 2) Init + ground-truth VehicleWorld
    vw_gt = VehicleWorld()
    local_gt = {'vw': vw_gt}
    execute(full_init, local_vars=local_gt, global_vars=None)
    execute(ground_truth_code, local_vars=local_gt, global_vars=None)

    # 3) Snapshot both
    if modules:
        snap_init = snapshot_modules(vw_init, modules)
        snap_gt = snapshot_modules(vw_gt, modules)
    else:
        snap_init = snapshot_all(vw_init)
        snap_gt = snapshot_all(vw_gt)

    # 4) Changed fields = what ground truth actually modifies
    changed_fields = deep_diff(snap_init, snap_gt)

    return snap_gt, changed_fields


def verify_by_snapshot(changed_fields: dict, vw_actual,
                       modules: list = None) -> dict:
    """Verify agent's VehicleWorld against ground-truth changed fields only.

    Accuracy formula::

        accuracy = correct / total

    where ``total`` = number of fields the ground-truth actually changed,
    and ``correct`` = how many of those the agent also got right.

    Args:
        changed_fields: Output of ``deep_diff(snap_init, snap_gt)`` from
                        ``build_expected_snapshot``.  Keys are dot-separated
                        leaf paths; values are ``{"expected": ..., "actual": ...}``
                        where "actual" is the ground-truth target value.
        vw_actual:      VehicleWorld instance after agent execution.
        modules:        If given, only snapshot these modules (for efficiency).

    Returns:
        dict with: accuracy, total_fields, correct_fields, diff_fields,
        per-field details list, and per-module summary.
    """
    # Build agent snapshot
    if modules:
        actual_snap = snapshot_modules(vw_actual, modules)
    else:
        actual_snap = snapshot_all(vw_actual)

    total = len(changed_fields)
    if total == 0:
        return {
            "accuracy": 1.0,
            "total_fields": 0,
            "correct_fields": 0,
            "diff_fields": 0,
            "details": [],
            "module_summary": {},
        }

    correct = 0
    details = []
    # Per-module aggregation
    mod_stats = {}  # mod_name -> {"total": int, "correct": int}

    for path, change in changed_fields.items():
        gt_val = change["actual"]  # ground-truth target value
        agent_val = get_by_path(actual_snap, path)

        # Special handling for broadcast announcements:
        # Check that each expected announcement has a matching one in agent's list
        # by category + priority (message text is free-form, so skip exact match).
        if path == "broadcast.announcements" and isinstance(gt_val, list) and isinstance(agent_val, list):
            is_correct = _match_announcements(gt_val, agent_val)
        else:
            is_correct = (agent_val == gt_val)

        if is_correct:
            correct += 1

        details.append({
            "field": path,
            "passed": is_correct,
            "expected": gt_val,
            "actual": agent_val,
        })

        # Module-level grouping (first path component)
        mod_name = path.split(".")[0]
        if mod_name not in mod_stats:
            mod_stats[mod_name] = {"total": 0, "correct": 0}
        mod_stats[mod_name]["total"] += 1
        if is_correct:
            mod_stats[mod_name]["correct"] += 1

    # Build module summary
    module_summary = {}
    for mod_name, stats in mod_stats.items():
        module_summary[mod_name] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["total"],
        }

    return {
        "accuracy": correct / total,
        "total_fields": total,
        "correct_fields": correct,
        "diff_fields": total - correct,
        "details": details,
        "module_summary": module_summary,
    }
