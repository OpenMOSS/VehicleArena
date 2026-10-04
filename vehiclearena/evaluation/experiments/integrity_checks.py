"""Static checks that keep experiment identities out of runtime policy code."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable


_IDENTITY_NAMES = {
    "experiment_id", "scenario_id", "variant_id",
    "road_network_id", "network_id", "map_id",
}
_EXPERIMENT_VALUE = re.compile(r"^(?:e\d(?:_|$)|map__)", re.IGNORECASE)


def _referenced_names(node: ast.AST) -> set[str]:
    names = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            names.add(item.id)
        elif isinstance(item, ast.Attribute):
            names.add(item.attr)
    return names


def _string_values(node: ast.AST) -> Iterable[str]:
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            yield item.value


def audit_simulation_identity_special_cases(source_root: Path) -> dict:
    """Find runtime branches coupled to experiment/scenario/map identities.

    Behavior may depend on physical state and explicit configuration, but not
    on labels used by the evaluator.  The authoring and evaluation packages
    are deliberately outside this scan because their job is to select scenes.
    """
    source_root = Path(source_root)
    violations = []
    files = sorted(source_root.rglob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Compare, ast.If, ast.IfExp)):
                continue
            expression = node.test if isinstance(
                node, (ast.If, ast.IfExp)) else node
            names = _referenced_names(expression)
            if not names.intersection(_IDENTITY_NAMES):
                continue
            values = [
                value for value in _string_values(expression)
                if _EXPERIMENT_VALUE.search(value)
            ]
            if values:
                violations.append({
                    "path": str(path.relative_to(source_root)),
                    "line": int(getattr(node, "lineno", 0)),
                    "identity_fields": sorted(
                        names.intersection(_IDENTITY_NAMES)),
                    "matched_values": sorted(set(values)),
                })
    return {
        "schema": "vehiclearena-runtime-integrity-audit-v1",
        "source_root": str(source_root),
        "python_file_count": len(files),
        "violations": violations,
        "passed": not violations,
    }
