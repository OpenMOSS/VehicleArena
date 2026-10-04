"""Resolve immutable per-vehicle equipment and chassis capabilities."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

import yaml


_ACTION_MODULE_RE = re.compile(r"^\s*vw\.([A-Za-z][A-Za-z0-9_]*)\.")
_CHASSIS_FIELDS = {
    "length_m",
    "width_m",
    "max_acceleration_mps2",
    "max_braking_mps2",
    "lane_change_duration_s",
}
_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CUSTOM_EQUIPMENT_PROFILES: Dict[str, dict] = {}
_CUSTOM_CHASSIS_PROFILES: Dict[str, dict] = {}


@dataclass(frozen=True)
class ChassisSpec:
    """Physical properties that affect motion and collision geometry."""

    length_m: float
    width_m: float
    max_acceleration_mps2: float
    max_braking_mps2: float
    lane_change_duration_s: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VehicleCapabilitySet:
    """Resolved capabilities for exactly one vehicle."""

    equipment_profile: str
    chassis_profile: str
    modules: FrozenSet[str]
    required_modules: FrozenSet[str]
    explicitly_enabled: Tuple[str, ...]
    explicitly_disabled: Tuple[str, ...]
    chassis: ChassisSpec

    def has_module(self, module_name: str) -> bool:
        return module_name in self.modules

    def missing_modules(self, module_names: Iterable[str]) -> list:
        return sorted(set(module_names) - set(self.modules))

    def to_dict(self) -> dict:
        return {
            "equipment_profile": self.equipment_profile,
            "chassis_profile": self.chassis_profile,
            "modules": sorted(self.modules),
            "required_modules": sorted(self.required_modules),
            "explicitly_enabled": list(self.explicitly_enabled),
            "explicitly_disabled": list(self.explicitly_disabled),
            "chassis": self.chassis.to_dict(),
        }


def action_module_name(action: str) -> Optional[str]:
    """Return the module targeted by ``vw.<module>.<method>(...)``."""
    match = _ACTION_MODULE_RE.match(action or "")
    return match.group(1) if match else None


@lru_cache(maxsize=1)
def _profiles() -> dict:
    path = Path(__file__).with_name("profiles.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data.get("equipment_profiles"), dict):
        raise ValueError("profiles.yaml needs an equipment_profiles mapping")
    if not isinstance(data.get("chassis_profiles"), dict):
        raise ValueError("profiles.yaml needs a chassis_profiles mapping")
    return data


def _profile_name(name: str) -> str:
    if not isinstance(name, str) or not _PROFILE_NAME_RE.fullmatch(name):
        raise ValueError("Profile name must match [a-z][a-z0-9_]*")
    return name


def _equipment_profiles() -> Dict[str, dict]:
    return {
        **_profiles()["equipment_profiles"],
        **_CUSTOM_EQUIPMENT_PROFILES,
    }


def _chassis_profiles() -> Dict[str, dict]:
    return {
        **_profiles()["chassis_profiles"],
        **_CUSTOM_CHASSIS_PROFILES,
    }


def register_equipment_profile(
    name: str,
    *,
    include: Iterable[str],
    exclude: Iterable[str] = (),
) -> None:
    """Register an extension-owned equipment profile.

    Module names are resolved when a scenario is parsed, allowing an
    extension to declare its profile before or after its module classes.
    """
    name = _profile_name(name)
    if name in _equipment_profiles():
        raise ValueError(f"Equipment profile {name!r} is already registered")
    _CUSTOM_EQUIPMENT_PROFILES[name] = {
        "include": list(_normalise_names(include, "include")),
        "exclude": list(_normalise_names(exclude, "exclude")),
    }


def register_chassis_profile(name: str, **values: float) -> None:
    """Register a complete physical chassis profile for scenario use."""
    name = _profile_name(name)
    if name in _chassis_profiles():
        raise ValueError(f"Chassis profile {name!r} is already registered")
    unknown = set(values) - _CHASSIS_FIELDS
    missing = _CHASSIS_FIELDS - set(values)
    if unknown or missing:
        raise ValueError(
            f"Chassis profile fields mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}")
    numeric = {field: float(values[field]) for field in _CHASSIS_FIELDS}
    if any(value <= 0 for value in numeric.values()):
        raise ValueError("All chassis values must be positive")
    _CUSTOM_CHASSIS_PROFILES[name] = numeric


def list_equipment_profiles() -> list:
    return sorted(_equipment_profiles())


def list_chassis_profiles() -> list:
    return sorted(_chassis_profiles())


def _normalise_names(value, field_name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a list of module names")
    names = tuple(str(item) for item in value)
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} contains duplicate module names")
    return names


def _registered_modules() -> Dict[str, object]:
    from registry import ModuleRegistry

    modules = ModuleRegistry.instance().all_modules()
    if not modules:
        raise RuntimeError(
            "Vehicle module registry is empty; import VehicleWorld first")
    return modules


def _resolve_chassis(
    profile_name: str,
    overrides: Optional[Mapping[str, float]],
) -> ChassisSpec:
    profiles = _chassis_profiles()
    try:
        raw = dict(profiles[profile_name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown chassis profile {profile_name!r}; "
            f"available={sorted(profiles)}") from exc

    overrides = dict(overrides or {})
    unknown = set(overrides) - _CHASSIS_FIELDS
    if unknown:
        raise ValueError(f"Unknown chassis override fields: {sorted(unknown)}")
    raw.update(overrides)
    missing = _CHASSIS_FIELDS - set(raw)
    if missing:
        raise ValueError(
            f"Chassis profile {profile_name!r} misses {sorted(missing)}")
    values = {name: float(raw[name]) for name in _CHASSIS_FIELDS}
    if any(value <= 0 for value in values.values()):
        raise ValueError("All chassis values must be positive")
    return ChassisSpec(**values)


def resolve_vehicle_capabilities(
    equipment_profile: str = "executive",
    chassis_profile: str = "sedan",
    enable_modules: Optional[Iterable[str]] = None,
    disable_modules: Optional[Iterable[str]] = None,
    chassis_overrides: Optional[Mapping[str, float]] = None,
) -> VehicleCapabilitySet:
    """Resolve a profile plus per-vehicle overrides.

    Required world/driver modules cannot be disabled. Unknown profile, module
    or chassis fields fail at scenario load time rather than halfway through a
    simulation.
    """
    registry = _registered_modules()
    all_modules = set(registry)
    profile_defs = _equipment_profiles()
    try:
        profile = profile_defs[equipment_profile] or {}
    except KeyError as exc:
        raise ValueError(
            f"Unknown equipment profile {equipment_profile!r}; "
            f"available={sorted(profile_defs)}") from exc

    include = _normalise_names(profile.get("include", ()), "profile.include")
    exclude = set(_normalise_names(
        profile.get("exclude", ()), "profile.exclude"))
    enabled = _normalise_names(enable_modules, "enable_modules")
    disabled = _normalise_names(disable_modules, "disable_modules")

    overlap = set(enabled) & set(disabled)
    if overlap:
        raise ValueError(
            f"Modules cannot be both enabled and disabled: {sorted(overlap)}")

    requested_names = (
        (set(include) - {"*"}) | exclude | set(enabled) | set(disabled))
    unknown = requested_names - all_modules
    if unknown:
        raise ValueError(f"Unknown vehicle modules: {sorted(unknown)}")

    modules = set(all_modules if "*" in include else include)
    modules.difference_update(exclude)
    modules.update(enabled)
    modules.difference_update(disabled)

    required = set(_profiles().get("required_modules", ()))
    unknown_required = required - all_modules
    if unknown_required:
        raise ValueError(
            f"profiles.yaml has unknown required modules: "
            f"{sorted(unknown_required)}")
    forbidden = required & set(disabled)
    if forbidden:
        raise ValueError(
            f"Required modules cannot be disabled: {sorted(forbidden)}")
    modules.update(required)

    return VehicleCapabilitySet(
        equipment_profile=equipment_profile,
        chassis_profile=chassis_profile,
        modules=frozenset(modules),
        required_modules=frozenset(required),
        explicitly_enabled=enabled,
        explicitly_disabled=disabled,
        chassis=_resolve_chassis(chassis_profile, chassis_overrides),
    )
