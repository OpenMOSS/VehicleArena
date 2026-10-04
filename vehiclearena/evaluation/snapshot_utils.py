"""
Snapshot-based verification utilities for VehicleWorld evaluation.

Provides automatic state comparison by recursively serialising any
VehicleWorld module (and its inner-class objects) into plain dicts,
then computing a field-level diff between *expected* and *actual*
snapshots.

Typical usage
-------------
    from snapshot_utils import snapshot_all, deep_diff

    # --- data construction (offline) ---
    vw_gt = VehicleWorld()
    execute_ground_truth_actions(vw_gt)
    expected = snapshot_all(vw_gt)

    # --- evaluation (online) ---
    vw_agent = VehicleWorld()
    agent_runs_its_actions(vw_agent)
    actual = snapshot_all(vw_agent)

    diff = deep_diff(expected, actual)
    passed = len(diff) == 0
"""

import re
from enum import Enum
from typing import Any, Dict, List, Set

# ── Prefixes / keys to skip during serialisation ──────────────────────
#   _event_bus, _constraint_engine  → coupling-engine internals
#   _parent_system                  → back-reference to VehicleWorld
SKIP_PREFIXES: tuple = (
    "_event_bus",
    "_constraint_engine",
    "_parent_system",
    "_simulation",
)

# Fields whose values change across VehicleWorld instantiations
# (timestamps, random IDs, etc.) and should be excluded from comparison.
SKIP_KEYS: Set[str] = {
    "timestamp",
    "last_operation_timestamp",
    "created_at",
    "updated_at",
    "direction_choices",
    "direction_road_map",
    "last_directions",
}

# ── All module names on VehicleWorld (auto-derived from ModuleRegistry) ──
def _get_module_names():
    """Derive module name lists from the ModuleRegistry at import time.

    Falls back to an empty list if the registry has not been populated yet
    (e.g. during unit tests that import snapshot_utils before VehicleWorld).
    """
    try:
        from registry import ModuleRegistry
        reg = ModuleRegistry.instance()
        return reg.internal_module_names(), reg.external_module_names()
    except Exception:
        return [], []

INTERNAL_MODULES: List[str]
EXTERNAL_MODULES: List[str]
INTERNAL_MODULES, EXTERNAL_MODULES = _get_module_names()


# =====================================================================
# Serialisation
# =====================================================================

def _serialize(val: Any) -> Any:
    """Recursively convert a Python value to a JSON-friendly primitive."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, dict):
        return {
            _serialize(k): _serialize(v)
            for k, v in val.items()
            if not (isinstance(k, str) and k.lstrip("_") in SKIP_KEYS)
        }
    if isinstance(val, (list, tuple)):
        return [_serialize(v) for v in val]
    # Inner-class objects (DoorState, WindowState, HeaterSystem, …)
    if hasattr(val, "__dict__"):
        return module_to_dict(val)
    # Fallback: repr
    return str(val)


def module_to_dict(module: Any) -> Dict[str, Any]:
    """Serialise a module (or inner-class object) into a plain dict.

    Iterates over ``vars(module)`` and recursively serialises each value,
    skipping coupling-engine internals, timestamp noise, and callable
    attributes (e.g. driving API intercept closures installed at runtime).
    """
    result: Dict[str, Any] = {}
    for key, val in vars(module).items():
        if any(key.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        clean_key = key.lstrip("_")
        if clean_key in SKIP_KEYS:
            continue
        if callable(val):
            continue
        result[clean_key] = _serialize(val)
    return result


# =====================================================================
# Full-world snapshot
# =====================================================================

def snapshot_module(vw, module_name: str) -> Dict[str, Any]:
    """Snapshot a single module by name."""
    if hasattr(vw, "has_module") and not vw.has_module(module_name):
        raise KeyError(
            f"Module {module_name!r} is not installed on this vehicle")
    if module_name in EXTERNAL_MODULES:
        mod = getattr(vw.externalWorld, module_name)
    else:
        mod = getattr(vw, module_name)
    return module_to_dict(mod)


def snapshot_all(vw) -> Dict[str, Dict[str, Any]]:
    """Snapshot every module on a VehicleWorld instance.

    Returns:
        ``{module_name: {field: value, ...}, ...}``
    """
    snap: Dict[str, Dict[str, Any]] = {}
    names = (
        vw.available_module_names()
        if hasattr(vw, "available_module_names")
        else INTERNAL_MODULES + EXTERNAL_MODULES)
    for name in names:
        snap[name] = snapshot_module(vw, name)
    return snap


def snapshot_modules(vw, module_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Snapshot only the specified modules."""
    snap: Dict[str, Dict[str, Any]] = {}
    for name in module_names:
        if hasattr(vw, "has_module") and not vw.has_module(name):
            continue
        snap[name] = snapshot_module(vw, name)
    return snap


# =====================================================================
# Leaf counting
# =====================================================================

def count_leaves(obj: Any) -> int:
    """Count leaf nodes in a serialised snapshot (nested dicts/lists).

    Every primitive value (str, int, float, bool, None) counts as 1 leaf.
    Empty dicts and empty lists also count as 1 leaf each.

    >>> count_leaves({"a": 1, "b": {"c": 2, "d": 3}})
    3
    >>> count_leaves([1, 2, 3])
    3
    """
    if isinstance(obj, dict):
        if not obj:
            return 1
        return sum(count_leaves(v) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return 1
        return sum(count_leaves(v) for v in obj)
    else:
        return 1


# =====================================================================
# Path-based value lookup
# =====================================================================

def get_by_path(data: dict, path: str) -> Any:
    """Navigate a nested dict/list by a deep_diff-style path.

    Path components are separated by ``'.'``; list indices use ``[i]``.

    Examples::

        get_by_path(snap, "wiper.wipers.front.is_active")
        get_by_path(snap, "conversation.contacts[0].name")

    Returns ``None`` if the path does not exist.
    """
    tokens = re.findall(r'[^.\[\]]+|\[\d+\]', path)
    current = data
    for tok in tokens:
        if current is None:
            return None
        if tok.startswith('[') and tok.endswith(']'):
            idx = int(tok[1:-1])
            current = current[idx] if isinstance(current, list) and idx < len(current) else None
        elif isinstance(current, dict):
            current = current.get(tok)
        else:
            return None
    return current


# =====================================================================
# Deep diff
# =====================================================================

def deep_diff(expected: Any, actual: Any, path: str = "") -> Dict[str, Dict]:
    """Recursively compare two serialised snapshots.

    Returns a dict of differing leaf paths::

        {
            "fogLight.front_light.is_on": {"expected": false, "actual": true},
            "door.doors.driver's seat.is_locked": {"expected": true, "actual": false},
        }

    An empty dict means the two snapshots are identical.
    """
    diffs: Dict[str, Dict] = {}

    if type(expected) != type(actual):
        diffs[path or "(root)"] = {"expected": expected, "actual": actual}
    elif isinstance(expected, dict):
        all_keys = set(list(expected.keys()) + list(actual.keys()))
        for k in all_keys:
            child_path = f"{path}.{k}" if path else str(k)
            diffs.update(
                deep_diff(expected.get(k), actual.get(k), child_path)
            )
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            diffs[path] = {"expected": expected, "actual": actual}
        else:
            for i in range(len(expected)):
                diffs.update(
                    deep_diff(expected[i], actual[i], f"{path}[{i}]")
                )
    elif expected != actual:
        diffs[path] = {"expected": expected, "actual": actual}

    return diffs


# =====================================================================
# High-level verification API
# =====================================================================

def verify_snapshot(vw_expected, vw_actual,
                    modules: List[str] = None) -> Dict[str, Any]:
    """Compare two VehicleWorld instances and return a verification report.

    Args:
        vw_expected: VehicleWorld after ground-truth actions.
        vw_actual:   VehicleWorld after agent actions.
        modules:     If given, only compare these modules; otherwise all.

    Returns:
        dict with *accuracy*, *total*, *passed*, *failed*, *diff*, *details*.
    """
    if modules:
        expected = snapshot_modules(vw_expected, modules)
        actual = snapshot_modules(vw_actual, modules)
    else:
        expected = snapshot_all(vw_expected)
        actual = snapshot_all(vw_actual)

    diff = deep_diff(expected, actual)

    # Count per-module pass/fail
    module_results: Dict[str, bool] = {}
    compared_modules = list(modules or sorted(
        set(expected) | set(actual)))
    for mod_name in compared_modules:
        mod_diff = deep_diff(
            expected.get(mod_name, {}),
            actual.get(mod_name, {}),
        )
        module_results[mod_name] = len(mod_diff) == 0

    passed = sum(1 for v in module_results.values() if v)
    total = len(module_results)

    return {
        "accuracy": passed / total if total > 0 else 1.0,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "diff": diff,
        "module_results": module_results,
    }


def verify_snapshot_from_dict(expected_snap: Dict, vw_actual,
                              modules: List[str] = None) -> Dict[str, Any]:
    """Compare a pre-saved expected snapshot dict against a live VehicleWorld.

    This is the primary entry point for evaluation: the expected snapshot is
    generated offline (during data construction) and stored alongside the task.

    Accuracy is computed at **field level**:
        accuracy = (total_fields - diff_fields) / total_fields

    Args:
        expected_snap: Pre-computed snapshot dict (from ``snapshot_all``).
        vw_actual:     VehicleWorld instance after agent execution.
        modules:       If given, only compare these modules.

    Returns:
        Same format as ``verify_snapshot``.
    """
    if modules:
        actual = snapshot_modules(vw_actual, modules)
        expected = {k: v for k, v in expected_snap.items() if k in modules}
    else:
        actual = snapshot_all(vw_actual)
        expected = expected_snap

    diff = deep_diff(expected, actual)

    module_results: Dict[str, bool] = {}
    total_fields_all = 0
    diff_fields_all = 0

    check_modules = modules or list(expected.keys())
    for mod_name in check_modules:
        exp_mod = expected.get(mod_name, {})
        mod_diff = deep_diff(exp_mod, actual.get(mod_name, {}))
        module_results[mod_name] = len(mod_diff) == 0
        total_fields_all += count_leaves(exp_mod)
        diff_fields_all += len(mod_diff)

    passed = sum(1 for v in module_results.values() if v)
    total = len(module_results)

    return {
        "accuracy": max(0.0, (total_fields_all - diff_fields_all)) / total_fields_all if total_fields_all > 0 else 1.0,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "total_fields": total_fields_all,
        "diff_fields": diff_fields_all,
        "diff": diff,
        "module_results": module_results,
    }
