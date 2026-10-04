"""YAML-derived evaluation for in-cabin behavior.

The rule loader and ground-truth engine turn matching YAML rules into action
lines.  This evaluator executes those actions on an isolated reference
VehicleWorld and compares only the per-event state delta with the agent's
delta.  It deliberately does not inspect physical driving trajectories.
"""

from __future__ import annotations

import ast
import copy
from fnmatch import fnmatch
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation.check_utils import _match_announcements
from evaluation.snapshot_utils import (
    INTERNAL_MODULES,
    deep_diff,
    get_by_path,
    snapshot_modules,
)
from capabilities import action_module_name


_SIMULATION_NAVIGATION_INTERCEPTS = (
    "navigation_set_speed",
    "navigation_emergency_stop",
    "navigation_change_lane",
    "navigation_select_maneuver",
    "navigation_u_turn",
    "navigation_route_plan",
)

def _flatten_leaves(value: Any, path: str = "") -> Dict[str, Any]:
    leaves: Dict[str, Any] = {}
    if isinstance(value, dict):
        if not value:
            leaves[path] = value
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            leaves.update(_flatten_leaves(child, child_path))
    elif isinstance(value, list):
        if not value:
            leaves[path] = value
        for index, child in enumerate(value):
            leaves.update(_flatten_leaves(child, f"{path}[{index}]"))
    else:
        leaves[path] = value
    return leaves


def execute_yaml_action(vw, action: str) -> Any:
    """Execute one strictly shaped ``vw.module.method(literals...)`` action.

    YAML is benchmark data, not arbitrary Python.  Restricting the grammar
    here makes GT evaluation deterministic and prevents imports, assignments,
    comprehensions or private-attribute access from becoming executable.
    """
    tree = ast.parse(action, mode="exec")
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.Expr)
        or not isinstance(tree.body[0].value, ast.Call)
    ):
        raise ValueError("YAML action must contain exactly one function call")
    call = tree.body[0].value

    attributes = []
    current = call.func
    while isinstance(current, ast.Attribute):
        if current.attr.startswith("_"):
            raise ValueError("private attributes are not allowed")
        attributes.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or current.id != "vw":
        raise ValueError("YAML action must start with 'vw'")

    target = vw
    for attribute in reversed(attributes):
        target = getattr(target, attribute)
    if not callable(target):
        raise ValueError("YAML action target is not callable")

    args = [ast.literal_eval(argument) for argument in call.args]
    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("**kwargs expansion is not allowed")
        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    return target(*args, **kwargs)


def _candidate_action_sets(
    ground_truth_lines: Sequence[str],
    acceptable_actions: Optional[Sequence[Any]],
    max_candidates: int = 32,
) -> List[List[str]]:
    """Build primary and ``any_of`` variants.

    An ``AcceptableAction`` replaces its primary action with one alternative.
    Alternatives prefixed by ``*:`` are field-skip declarations and do not
    create an executable candidate.
    """
    candidates: List[List[str]] = [list(ground_truth_lines)]
    for acceptable in acceptable_actions or ():
        alternatives = [
            action for action in acceptable.alternatives
            if isinstance(action, str) and not action.startswith("*:")
        ]
        if not acceptable.primary or not alternatives:
            continue
        expanded: List[List[str]] = list(candidates)
        for current in candidates:
            try:
                replace_index = current.index(acceptable.primary)
            except ValueError:
                continue
            for alternative in alternatives:
                variant = list(current)
                variant[replace_index] = alternative
                expanded.append(variant)
                if len(expanded) >= max_candidates:
                    break
            if len(expanded) >= max_candidates:
                break
        candidates = expanded[:max_candidates]

    unique: List[List[str]] = []
    seen = set()
    for candidate in candidates:
        signature = tuple(candidate)
        if signature not in seen:
            seen.add(signature)
            unique.append(candidate)
    return unique


class CabinYamlEvaluator:
    """Evaluate cabin rule events independently for every vehicle."""

    def __init__(self):
        self._active_negative_violations: Dict[str, set] = {}

    def evaluate(
        self,
        vehicle_id: str,
        time_s: float,
        pre_agent_vw,
        post_agent_vw,
        ground_truth_lines: Sequence[str],
        acceptable_actions: Optional[Sequence[Any]] = None,
        trend_tolerances: Optional[Sequence[Any]] = None,
        negative_checks: Optional[Sequence[Any]] = None,
        global_skip_fields: Optional[Sequence[str]] = None,
    ) -> Optional[dict]:
        """Evaluate one wake event.

        Cabin scoring is YAML-triggered. Empty heartbeats and unrequested
        state changes are not scored; a result is emitted only when a YAML
        expectation applies, its reference action is invalid, or a YAML
        negative invariant becomes newly violated.
        """
        supported_lines = []
        skipped_unavailable = []
        for line in ground_truth_lines:
            module = action_module_name(line)
            if (
                module
                and hasattr(post_agent_vw, "has_module")
                and not post_agent_vw.has_module(module)
            ):
                skipped_unavailable.append({
                    "action": line,
                    "module": module,
                    "reason": "capability_not_available",
                })
            else:
                supported_lines.append(line)
        ground_truth_lines = supported_lines

        pre_snapshot = snapshot_modules(pre_agent_vw, INTERNAL_MODULES)
        actual_snapshot = snapshot_modules(post_agent_vw, INTERNAL_MODULES)
        actual_delta = deep_diff(pre_snapshot, actual_snapshot)

        skip_patterns = list(global_skip_fields or ())
        for acceptable in acceptable_actions or ():
            skip_patterns.extend(
                alternative[2:]
                for alternative in acceptable.alternatives
                if isinstance(alternative, str)
                and alternative.startswith("*:")
            )

        candidates = _candidate_action_sets(
            ground_truth_lines, acceptable_actions)
        evaluated_candidates = []
        execution_errors: List[dict] = []
        for lines in candidates:
            expected_vw, errors = self._build_expected_world(
                pre_agent_vw, lines)
            if errors:
                execution_errors.extend(errors)
                continue
            expected_snapshot = snapshot_modules(
                expected_vw, INTERNAL_MODULES)
            evaluated_candidates.append(
                self._compare_candidate(
                    pre_snapshot=pre_snapshot,
                    actual_snapshot=actual_snapshot,
                    actual_delta=actual_delta,
                    expected_snapshot=expected_snapshot,
                    lines=lines,
                    skip_patterns=skip_patterns,
                    trend_tolerances=trend_tolerances or (),
                )
            )

        if ground_truth_lines and not evaluated_candidates:
            # Invalid reference actions are a benchmark-definition failure,
            # never a silent pass for an agent that happened to do nothing.
            return {
                "tick": time_s,
                "evaluation_type": "cabin_yaml",
                "applicable": True,
                "total_fields": 1,
                "correct_fields": 0,
                "accuracy": 0.0,
                "expected_actions": list(ground_truth_lines),
                "accepted_actions": [],
                "skipped_unavailable_actions": skipped_unavailable,
                "gt_execution_errors": execution_errors,
                "details": [{
                    "field": "(ground_truth_execution)",
                    "passed": False,
                    "expected": "all YAML actions execute successfully",
                    "actual": "ground-truth action failed",
                }],
            }

        best = (
            max(
                evaluated_candidates,
                key=lambda item: (
                    item["accuracy"],
                    item["correct_fields"],
                    -item["total_fields"],
                ),
            )
            if evaluated_candidates else {
                "total_fields": 0,
                "correct_fields": 0,
                "accuracy": 1.0,
                "details": [],
                "accepted_actions": [],
            }
        )

        negative_details, currently_active = self._negative_violations(
            actual_snapshot, negative_checks or ())
        previous_active = self._active_negative_violations.setdefault(
            vehicle_id, set())
        new_negative_ids = currently_active - previous_active
        self._active_negative_violations[vehicle_id] = currently_active
        new_negative_details = [
            detail for detail in negative_details
            if detail["rule_id"] in new_negative_ids
        ]

        total = best["total_fields"] + len(new_negative_details)
        correct = best["correct_fields"]
        details = [*best["details"], *new_negative_details]

        if (
            not ground_truth_lines
            and not acceptable_actions
            and not trend_tolerances
            and not new_negative_details
        ):
            return None

        # A YAML action can be idempotent because the requested state already
        # holds.  That is a real applicable rule event, unlike an empty
        # heartbeat, and counts as one fulfilled obligation.
        if total == 0 and ground_truth_lines:
            total = 1
            correct = 1
            details.append({
                "field": "(already_satisfied)",
                "passed": True,
                "expected": "YAML target state",
                "actual": "already satisfied before this event",
            })

        return {
            "tick": time_s,
            "evaluation_type": "cabin_yaml",
            "applicable": total > 0,
            "total_fields": total,
            "correct_fields": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "expected_actions": list(ground_truth_lines),
            "accepted_actions": best["accepted_actions"],
            "skipped_unavailable_actions": skipped_unavailable,
            "gt_execution_errors": [],
            "ignored_invalid_alternatives": execution_errors,
            "details": details,
        }

    @staticmethod
    def _build_expected_world(pre_agent_vw, lines: Sequence[str]):
        expected_vw = copy.deepcopy(pre_agent_vw)
        from coupling_rules import register_all_rules

        expected_vw._constraint_engine._enabled = True
        expected_vw._constraint_engine.set_vehicle_world(expected_vw)
        register_all_rules(
            expected_vw._event_bus, expected_vw._constraint_engine)

        navigation = expected_vw.navigation
        for name in _SIMULATION_NAVIGATION_INTERCEPTS:
            navigation.__dict__.pop(name, None)

        errors = []
        for line in lines:
            action = line.strip()
            if not action:
                continue
            try:
                output = execute_yaml_action(expected_vw, action)
                if isinstance(output, dict) and output.get("success") is False:
                    raise RuntimeError(
                        output.get("error")
                        or output.get("reason")
                        or output.get("message")
                        or "API returned success=false")
            except Exception as exc:
                errors.append({
                    "action": action,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return expected_vw, errors

    @staticmethod
    def _compare_candidate(
        pre_snapshot: dict,
        actual_snapshot: dict,
        actual_delta: dict,
        expected_snapshot: dict,
        lines: Sequence[str],
        skip_patterns: Sequence[str],
        trend_tolerances: Sequence[Any],
    ) -> dict:
        expected_delta = deep_diff(pre_snapshot, expected_snapshot)
        # A positive YAML rule defines the fields that this checkpoint scores.
        # Other actions in the same wake may belong to the driving domain (for
        # example route planning) or to another independently evaluated cabin
        # request.  Treating every actual delta as a new positive obligation
        # lets unrelated, otherwise valid actions dilute the rule score.
        # Explicit YAML negative checks remain responsible for forbidden
        # state, so positive scoring is intentionally limited to the reference
        # action's target delta.
        changed_paths = set(expected_delta)

        total = 0
        correct = 0
        details = []
        for path in sorted(changed_paths):
            if any(fnmatch(path, pattern) for pattern in skip_patterns):
                continue
            total += 1
            expected_value = (
                expected_delta[path]["actual"]
                if path in expected_delta
                else get_by_path(pre_snapshot, path)
            )
            actual_value = (
                actual_delta[path]["actual"]
                if path in actual_delta
                else get_by_path(pre_snapshot, path)
            )
            if (
                path == "broadcast.announcements"
                and path in expected_delta
                and isinstance(expected_value, list)
                and isinstance(actual_value, list)
            ):
                passed = _match_announcements(
                    expected_value, actual_value)
            else:
                passed = actual_value == expected_value

            if not passed:
                passed = CabinYamlEvaluator._matches_trend(
                    path, actual_value, trend_tolerances)
            if passed:
                correct += 1
            details.append({
                "field": path,
                "passed": passed,
                "expected": expected_value,
                "actual": actual_value,
            })

        return {
            "total_fields": total,
            "correct_fields": correct,
            "accuracy": correct / total if total > 0 else 1.0,
            "details": details,
            "accepted_actions": list(lines),
        }

    @staticmethod
    def _matches_trend(
        path: str, actual_value: Any,
        trend_tolerances: Sequence[Any],
    ) -> bool:
        for tolerance in trend_tolerances:
            if not fnmatch(path, tolerance.field_pattern):
                continue
            baseline = tolerance.baseline_value
            if not (
                isinstance(actual_value, (int, float))
                and isinstance(baseline, (int, float))
            ):
                return False
            if tolerance.direction == "decrease":
                return actual_value < baseline
            if tolerance.direction == "increase":
                return actual_value > baseline
            if tolerance.direction == "any_change":
                return actual_value != baseline
            if tolerance.direction == "at_most":
                return actual_value <= baseline
            return False
        return False

    @staticmethod
    def _negative_violations(
        snapshot: dict, negative_checks: Iterable[Any],
    ) -> Tuple[List[dict], set]:
        leaves = _flatten_leaves(snapshot)
        details = []
        active_ids = set()
        for check in negative_checks:
            rule_id = getattr(
                check, "id", "") or getattr(check, "field_path", "")
            matched_paths = [
                (path, value)
                for path, value in leaves.items()
                if fnmatch(path, check.field_path)
            ]
            for path, value in matched_paths:
                forbidden = check.forbidden_value
                forbidden_values = (
                    forbidden
                    if isinstance(forbidden, (list, tuple, set))
                    else [forbidden]
                )
                if value not in forbidden_values:
                    continue
                active_ids.add(rule_id)
                details.append({
                    "field": path,
                    "passed": False,
                    "expected": f"not {forbidden!r}",
                    "actual": value,
                    "rule_id": rule_id,
                    "severity": getattr(
                        check, "severity", "violation"),
                    "reason": check.reason,
                })
        return details, active_ids
