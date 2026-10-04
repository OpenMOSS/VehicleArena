"""Explicit, deterministic loading of trusted VehicleArena extensions."""

from __future__ import annotations

import importlib
import re
from dataclasses import asdict, dataclass
from typing import Iterable, List


_IMPORT_PATH_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


@dataclass(frozen=True)
class ExtensionReport:
    import_path: str
    vehicle_modules: tuple = ()
    runtime_entities: tuple = ()
    scenario_layers: tuple = ()
    equipment_profiles: tuple = ()
    chassis_profiles: tuple = ()
    rule_directories: tuple = ()

    def to_dict(self) -> dict:
        return asdict(self)


def _registry_snapshot() -> dict:
    # Importing VehicleWorld ensures all built-in equipment modules have
    # registered before a third-party package is evaluated.
    from vehiclearena import VehicleWorld  # noqa: F401
    from capabilities import (
        list_chassis_profiles, list_equipment_profiles,
    )
    from registry import ModuleRegistry
    from rules.rule_loader import list_rule_directories
    from simulation.runtime_entities import list_runtime_entity_types
    from simulation.scenario_generator import list_scenario_layers

    return {
        "vehicle_modules": set(
            ModuleRegistry.instance().all_modules()),
        "runtime_entities": set(list_runtime_entity_types()),
        "scenario_layers": set(list_scenario_layers()),
        "equipment_profiles": set(list_equipment_profiles()),
        "chassis_profiles": set(list_chassis_profiles()),
        "rule_directories": set(list_rule_directories()),
    }


def load_extension(import_path: str) -> ExtensionReport:
    """Import one trusted package and report exactly what it registered.

    Extension loading is deliberately a run-level operation. Scenario data is
    never allowed to import Python code by itself.
    """
    if (not isinstance(import_path, str)
            or not _IMPORT_PATH_RE.fullmatch(import_path)):
        raise ValueError(f"Invalid extension import path: {import_path!r}")
    before = _registry_snapshot()
    importlib.import_module(import_path)
    after = _registry_snapshot()
    additions = {
        name: tuple(sorted(after[name] - before[name]))
        for name in before
    }
    return ExtensionReport(import_path=import_path, **additions)


def load_extensions(import_paths: Iterable[str]) -> List[ExtensionReport]:
    """Load extensions in the caller-specified order."""
    if isinstance(import_paths, str):
        raise TypeError("extensions must be a list of import paths")
    return [load_extension(path) for path in import_paths]


__all__ = ["ExtensionReport", "load_extension", "load_extensions"]
