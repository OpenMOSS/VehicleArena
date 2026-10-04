"""Declarative rule loader for VehicleWorld GT engine.

Loads rules from YAML files in the rules/ directory. All rules share one
authoritative schema:

    - id: ...
      description: ...
      domain: weather|daynight|map_event|user_intent
      trigger:
        state_change: { field, from, to }   # weather/daynight
        user_intent:  { messages, vague }   # passenger
        map_event:    { event_types }
      guard: { <field>: <value> }
      expect:
        actions: [...]
        broadcast: { ... }
        action_template / speed_camera_template / ...
      tolerance:
        skip: [...]
        any_of: [...]
        trend: { ... }
        ceiling: { ... }
      priority: <int>
      message_params: { ... }              # user_intent only

Negative checks (cross-cutting safety invariants) are stored in a separate
``negative_checks:`` section.

Top-level YAML structure:
    rules: [...]                 # all domains incl. user_intent
    global_skip_fields: [...]
    negative_checks: [...]       # optional

Only this schema is accepted; unknown layouts and fields are rejected.
"""

from __future__ import annotations

import os
import glob
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Any

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Unified Rule
# ---------------------------------------------------------------------------

_ALLOWED_DOMAINS = {"weather", "daynight", "map_event", "user_intent"}
_TOP_LEVEL_FIELDS = {"rules", "global_skip_fields", "negative_checks"}
_RULE_FIELDS = {
    "id", "description", "domain", "trigger", "guard", "expect",
    "tolerance", "priority", "message_params", "warning_map", "skip_on",
    "speed_camera_warning", "requires_modules",
}
_TRIGGER_FIELDS = {
    "state_change": {"field", "from", "to"},
    "map_event": {"event_types"},
    "user_intent": {"messages", "vague"},
}
_DOMAIN_TRIGGER = {
    "weather": "state_change",
    "daynight": "state_change",
    "map_event": "map_event",
    "user_intent": "user_intent",
}
_EXPECT_FIELDS = {
    "actions", "broadcast", "action_template", "speed_camera_template",
    "speed_camera_template_no_type",
}
_TOLERANCE_FIELDS = {"skip", "any_of", "trend", "ceiling"}
_TREND_FIELDS = {"field_pattern", "direction", "baseline_field"}
_CEILING_FIELDS = {"field_pattern", "context_var"}
_NEGATIVE_CHECK_FIELDS = {
    "id", "field", "forbidden", "reason", "severity", "condition",
    "passenger_keyword",
}


def _require_mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _reject_unknown(mapping: Dict[str, Any], allowed: set, label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {unknown!r}")


@dataclass
class Rule:
    """One rule in the authoritative declarative schema."""

    id: str = ""
    description: str = ""
    domain: str = ""
    trigger: Dict[str, Any] = field(default_factory=dict)
    guard: Dict[str, Any] = field(default_factory=dict)
    expect: Dict[str, Any] = field(default_factory=dict)
    tolerance: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    message_params: Dict[str, Dict] = field(default_factory=dict)
    warning_map: Dict[str, str] = field(default_factory=dict)
    skip_on: List[str] = field(default_factory=list)
    speed_camera_warning: str = ""
    declared_modules: List[str] = field(default_factory=list)

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        """Parse one rule and reject any unsupported layout."""
        d = _require_mapping(d, "rule")
        _reject_unknown(d, _RULE_FIELDS, "rule")
        domain = d.get("domain")
        if domain not in _ALLOWED_DOMAINS:
            raise ValueError(
                f"Unsupported or missing rule domain {domain!r}; "
                f"expected one of {sorted(_ALLOWED_DOMAINS)!r}")
        rule_id = d.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            raise ValueError("Every rule requires a non-empty string id")

        trigger = _require_mapping(d.get("trigger"), "rule.trigger")
        expected_trigger = _DOMAIN_TRIGGER[domain]
        if set(trigger) != {expected_trigger}:
            raise ValueError(
                f"Rule domain {domain!r} requires exactly the "
                f"{expected_trigger!r} trigger")
        trigger_body = _require_mapping(
            trigger[expected_trigger], f"rule.trigger.{expected_trigger}")
        _reject_unknown(
            trigger_body, _TRIGGER_FIELDS[expected_trigger],
            f"rule.trigger.{expected_trigger}")
        for name in {
            "from", "to", "event_types", "messages", "vague",
        } & set(trigger_body):
            _require_list(
                trigger_body[name], f"rule.trigger.{expected_trigger}.{name}")

        expect = _require_mapping(d.get("expect", {}), "rule.expect")
        _reject_unknown(expect, _EXPECT_FIELDS, "rule.expect")
        if "actions" in expect:
            _require_list(expect["actions"], "rule.expect.actions")
        if "broadcast" in expect:
            _require_mapping(expect["broadcast"], "rule.expect.broadcast")

        tolerance = _require_mapping(
            d.get("tolerance", {}), "rule.tolerance")
        _reject_unknown(tolerance, _TOLERANCE_FIELDS, "rule.tolerance")
        for name in {"skip", "any_of"} & set(tolerance):
            _require_list(tolerance[name], f"rule.tolerance.{name}")
        for name, allowed in (
            ("trend", _TREND_FIELDS), ("ceiling", _CEILING_FIELDS),
        ):
            if name in tolerance:
                body = _require_mapping(
                    tolerance[name], f"rule.tolerance.{name}")
                _reject_unknown(body, allowed, f"rule.tolerance.{name}")

        guard = _require_mapping(d.get("guard", {}), "rule.guard")
        message_params = _require_mapping(
            d.get("message_params", {}), "rule.message_params")
        warning_map = _require_mapping(
            d.get("warning_map", {}), "rule.warning_map")
        for name in ("skip_on", "requires_modules"):
            if name in d:
                _require_list(d[name], f"rule.{name}")
        priority = d.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("rule.priority must be an integer")

        return cls(
            id=rule_id,
            description=d.get("description", ""),
            domain=domain,
            trigger=trigger,
            guard=guard,
            expect=expect,
            tolerance=tolerance,
            priority=priority,
            message_params=message_params,
            warning_map=warning_map,
            skip_on=d.get("skip_on", []) or [],
            speed_camera_warning=d.get("speed_camera_warning", "") or "",
            declared_modules=d.get("requires_modules", []) or [],
        )
    @property
    def state_change(self) -> Dict[str, Any]:
        return self.trigger.get("state_change", {}) or {}

    @property
    def expected_actions(self) -> List[str]:
        return self.expect.get("actions", []) or []

    @property
    def vague_expected_actions(self) -> List[str]:
        return self.expect.get("vague_actions", []) or []

    @property
    def broadcasts(self) -> Dict[str, str]:
        return self.expect.get("broadcast", {}) or {}

    @property
    def ignored_fields(self) -> List[str]:
        return self.tolerance.get("skip", []) or []

    @property
    def accepted_alternatives(self) -> List[str]:
        return self.tolerance.get("any_of", []) or []

    @property
    def trend_spec(self) -> Optional[Dict[str, Any]]:
        return self.tolerance.get("trend")

    @property
    def ceiling_spec(self) -> Optional[Dict[str, Any]]:
        return self.tolerance.get("ceiling")

    @property
    def intent_messages(self) -> List[str]:
        return (self.trigger.get("user_intent") or {}).get("messages", []) or []

    @property
    def vague_intent_messages(self) -> List[str]:
        return (self.trigger.get("user_intent") or {}).get("vague", []) or []

    @property
    def required_modules(self) -> List[str]:
        """Return explicitly declared or action-derived equipment modules."""
        if self.declared_modules:
            return list(dict.fromkeys(self.declared_modules))
        modules = []
        for action in [
            *self.expected_actions,
            *self.vague_expected_actions,
        ]:
            match = re.match(
                r"^\s*vw\.([A-Za-z][A-Za-z0-9_]*)\.", action or "")
            if match and match.group(1) not in modules:
                modules.append(match.group(1))
        return modules


# ---------------------------------------------------------------------------
# Negative checks
# ---------------------------------------------------------------------------

@dataclass
class NegativeCheckDecl:
    """A declarative negative check rule.

    Specifies a world-state condition under which a field must NOT
    have the forbidden value.
    """
    id: str
    field: str
    forbidden: Any = True
    reason: str = ""
    severity: str = "violation"
    condition: Dict[str, Any] = field(default_factory=dict)
    passenger_keyword: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "NegativeCheckDecl":
        d = _require_mapping(d, "negative check")
        _reject_unknown(d, _NEGATIVE_CHECK_FIELDS, "negative check")
        check_id = d.get("id")
        check_field = d.get("field")
        if not isinstance(check_id, str) or not check_id:
            raise ValueError(
                "Every negative check requires a non-empty string id")
        if not isinstance(check_field, str) or not check_field:
            raise ValueError(
                "Every negative check requires a non-empty string field")
        condition = _require_mapping(
            d.get("condition", {}), "negative check.condition")
        keywords = d.get("passenger_keyword", [])
        _require_list(keywords, "negative check.passenger_keyword")
        return cls(
            id=check_id,
            field=check_field,
            forbidden=d.get("forbidden", True),
            reason=d.get("reason", ""),
            severity=d.get("severity", "violation"),
            condition=condition,
            passenger_keyword=keywords,
        )


# ---------------------------------------------------------------------------
# RuleLoader
# ---------------------------------------------------------------------------

class RuleLoader:
    """Load declarative rules from YAML files in rules/ directory."""

    def __init__(self, rules_dir: Optional[Any] = None):
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required for YAML-based cabin evaluation. "
                "Install dependencies from requirements.txt.")
        source = rules_dir or os.path.join(os.path.dirname(__file__))
        if isinstance(source, (str, os.PathLike)):
            source = [source]
        if not isinstance(source, Iterable):
            raise TypeError("rules_dir must be a path or list of paths")
        self.rules_dirs = tuple(
            str(Path(item).expanduser().resolve()) for item in source)
        if not self.rules_dirs:
            raise ValueError("At least one rule directory is required")
        missing = [path for path in self.rules_dirs
                   if not os.path.isdir(path)]
        if missing:
            raise ValueError(f"Rule directories do not exist: {missing!r}")
        self.env_rules: List[Rule] = []
        self.user_intent_rules: List[Rule] = []
        self._user_intent_by_category: Dict[str, Rule] = {}
        self.global_skip_fields: List[str] = []
        self.negative_checks: List[NegativeCheckDecl] = []
        self._load_all()

    def _load_all(self):
        yaml_files = []
        for rules_dir in self.rules_dirs:
            yaml_files.extend(glob.glob(os.path.join(rules_dir, "*.yaml")))
        for yaml_file in sorted(yaml_files):
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data:
                continue
            data = _require_mapping(data, f"YAML document {yaml_file}")
            _reject_unknown(
                data, _TOP_LEVEL_FIELDS, f"YAML document {yaml_file}")

            rules = data.get("rules", [])
            _require_list(rules, f"{yaml_file}: rules")
            for rule_dict in rules:
                rule = Rule.from_dict(rule_dict)
                if rule.domain == "user_intent":
                    self.user_intent_rules.append(rule)
                    self._user_intent_by_category[rule.id] = rule
                else:
                    self.env_rules.append(rule)

            global_skip_fields = data.get("global_skip_fields", [])
            _require_list(
                global_skip_fields, f"{yaml_file}: global_skip_fields")
            for pat in global_skip_fields:
                if pat not in self.global_skip_fields:
                    self.global_skip_fields.append(pat)

            negative_checks = data.get("negative_checks", [])
            _require_list(
                negative_checks, f"{yaml_file}: negative_checks")
            for nc_dict in negative_checks:
                self.negative_checks.append(NegativeCheckDecl.from_dict(nc_dict))

        seen = set()
        for rule in [*self.env_rules, *self.user_intent_rules]:
            key = (rule.domain, rule.id)
            if key in seen:
                raise ValueError(
                    f"Duplicate YAML rule id in domain {rule.domain!r}: "
                    f"{rule.id!r}")
            seen.add(key)

    # ── Generic transition matching ───────────────────────────────────

    @staticmethod
    def _val_in(val, pattern_list) -> bool:
        """Check if *val* matches any entry in *pattern_list*.

        Handles bool/str/number coercion and treats ``True`` as a
        truthy-match wildcard.
        """
        if not pattern_list:
            return True  # no constraint = wildcard
        for p in pattern_list:
            if val == p:
                return True
            if isinstance(val, bool) and isinstance(p, str):
                if str(val).lower() == p.lower():
                    return True
            if isinstance(p, bool) and isinstance(val, str):
                if str(p).lower() == val.lower():
                    return True
            if p is True and val:
                return True
            if p is False and not val:
                return True
        return False

    def match_state_change(self, domain: str,
                         prev_snapshot=None, curr_snapshot=None,
                         prev_value=None, curr_value=None,
                         ) -> List[Rule]:
        """Match one weather/daynight state transition.

        Two matching modes:
        1. **Field-based** (rule declares ``field``): reads the named
           attribute from *prev_snapshot* / *curr_snapshot* and compares
           against the rule's from/to lists.
        2. **Value-based** (no ``field``): compares *prev_value*
           / *curr_value* directly (used by weather / daynight callers).

        Returns matched rules sorted by priority (lower number first).
        """
        matched = []
        for rule in self.env_rules:
            if rule.domain != domain:
                continue
            trigger = rule.state_change
            trigger_field = trigger.get("field", "")
            if trigger_field:
                if prev_snapshot is None or curr_snapshot is None:
                    continue
                prev_val = getattr(prev_snapshot, trigger_field, None)
                curr_val = getattr(curr_snapshot, trigger_field, None)
            else:
                prev_val = prev_value
                curr_val = curr_value

            if prev_val == curr_val:
                continue

            trigger_from = trigger.get("from", []) or []
            trigger_to = trigger.get("to", []) or []
            if trigger_from and not self._val_in(prev_val, trigger_from):
                continue
            if trigger_to and not self._val_in(curr_val, trigger_to):
                continue

            if rule.guard and curr_snapshot:
                cond_ok = True
                for cond_field, cond_val in rule.guard.items():
                    actual = getattr(curr_snapshot, cond_field, None)
                    if actual != cond_val:
                        cond_ok = False
                        break
                if not cond_ok:
                    continue

            matched.append(rule)

        matched.sort(key=lambda r: r.priority)
        return matched

    @staticmethod
    def resolve_actions(rule: Rule,
                        context: Dict[str, Any] = None) -> List[str]:
        """Resolve ``{var}`` template placeholders in rule actions."""
        if not context:
            return list(rule.expected_actions)
        resolved = []
        for action in rule.expected_actions:
            try:
                resolved.append(action.format_map(context))
            except (KeyError, ValueError):
                resolved.append(action)
        return resolved

    # ── Domain wrappers ───────────────────────────────────────────────

    def match_weather(self, prev_cond: str, curr_cond: str) -> List[Rule]:
        return self.match_state_change(
            "weather", prev_value=prev_cond, curr_value=curr_cond)

    def match_daynight(self, prev_period: str, curr_period: str,
                       is_first_tick: bool = False) -> List[Rule]:
        if is_first_tick:
            _DARK = {"dusk", "night", "dawn"}
            if curr_period not in _DARK:
                return []
            matched = []
            for rule in self.env_rules:
                if rule.domain != "daynight":
                    continue
                trigger_to = rule.state_change.get("to", []) or []
                if trigger_to and self._val_in(curr_period, trigger_to):
                    matched.append(rule)
            matched.sort(key=lambda r: r.priority)
            return matched
        return self.match_state_change(
            "daynight", prev_value=prev_period, curr_value=curr_period)

    def match_map_events(self) -> List[Rule]:
        return [r for r in self.env_rules if r.domain == "map_event"]

    # ── User-intent rule access ───────────────────────────────────────

    def get_user_intent_rule(self, category: str) -> Optional[Rule]:
        """Get a user-intent rule by exact or declared prefix category."""
        exact = self._user_intent_by_category.get(category)
        if exact is not None:
            return exact
        for pat, rule in self._user_intent_by_category.items():
            if pat.endswith("*") and category.startswith(pat[:-1]):
                return rule
        return None

    def all_user_intent_rules(self) -> List[Rule]:
        return list(self.user_intent_rules)

    def get_user_intent_categories(self) -> List[str]:
        return list(self._user_intent_by_category.keys())

    def missing_modules_for_user_intent(
        self, category: str, available_modules,
    ) -> List[str]:
        """Return equipment that makes a user-intent rule inapplicable."""
        rule = self.get_user_intent_rule(category)
        if rule is None:
            return []
        return sorted(
            set(rule.required_modules) - set(available_modules or ()))

    # ── Negative checks ───────────────────────────────────────────────

    def match_negative_checks(self, snapshot, passenger_messages: List[str] = None,
                              ) -> List[NegativeCheckDecl]:
        """Return negative checks whose conditions match the current snapshot.

        Checks with passenger_keyword are excluded when any message
        contains one of the keywords.
        """
        msgs_lower = [m.lower() for m in (passenger_messages or [])
                       if isinstance(m, str)]
        matched = []
        for nc in self.negative_checks:
            if nc.passenger_keyword and any(
                kw in ml for kw in nc.passenger_keyword for ml in msgs_lower
            ):
                continue
            if nc.condition and not self._eval_condition(nc.condition, snapshot):
                continue
            matched.append(nc)
        return matched

    @staticmethod
    def _eval_condition(condition: Dict[str, Any], snapshot) -> bool:
        """Evaluate a negative-check condition dict against a snapshot.

        Supports:
          field: { in: [...] }      — value must be in list
          field: { not_in: [...] }  — value must not be in list
          field: { gt/gte/lt/lte: number }
        """
        for field_name, predicate in condition.items():
            val = getattr(snapshot, field_name, None)
            if isinstance(predicate, dict):
                if "in" in predicate:
                    if val not in predicate["in"]:
                        return False
                if "not_in" in predicate:
                    if val in predicate["not_in"]:
                        return False
                if "gt" in predicate:
                    if (not isinstance(val, (int, float))
                            or not val > predicate["gt"]):
                        return False
                if "gte" in predicate:
                    if (not isinstance(val, (int, float))
                            or not val >= predicate["gte"]):
                        return False
                if "lt" in predicate:
                    if (not isinstance(val, (int, float))
                            or not val < predicate["lt"]):
                        return False
                if "lte" in predicate:
                    if (not isinstance(val, (int, float))
                            or not val <= predicate["lte"]):
                        return False
            else:
                if val != predicate:
                    return False
        return True


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_rule_loader: Optional[RuleLoader] = None
_additional_rule_directories: List[str] = []


def register_rule_directory(path: str) -> None:
    """Add an extension rule directory and rebuild the global rule index."""
    global _rule_loader
    resolved = str(Path(path).expanduser().resolve())
    if not os.path.isdir(resolved):
        raise ValueError(f"Rule directory does not exist: {resolved!r}")
    if resolved in _additional_rule_directories:
        raise ValueError(f"Rule directory already registered: {resolved!r}")
    _additional_rule_directories.append(resolved)
    _rule_loader = None


def list_rule_directories() -> List[str]:
    return list(_additional_rule_directories)


def get_rule_loader() -> RuleLoader:
    """Get or create the global RuleLoader singleton."""
    global _rule_loader
    if _rule_loader is None:
        built_in = str(Path(__file__).resolve().parent)
        _rule_loader = RuleLoader(
            [built_in, *_additional_rule_directories])
    return _rule_loader
