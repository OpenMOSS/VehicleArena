"""Extension registry for entities backed by authoritative physics actors.

An entity adapter translates a domain-specific scenario entry (bus, cargo
bike, delivery robot, and so on) into a strict native vehicle or pedestrian
configuration. SUMO materialises its body dimensions and owns all contacts;
VehicleArena consumes the synchronized state for perception and evaluation.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type


_REGISTRY: Dict[str, Type["RuntimeEntityAdapter"]] = {}
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PHYSICAL_KINDS = {"vehicle", "pedestrian"}


class RuntimeEntityAdapter(ABC):
    """Translate an extension entity into a native SUMO-backed actor."""

    entity_type = "base"
    physical_kind = ""

    @classmethod
    @abstractmethod
    def build_config(cls, specification: Dict[str, Any]) -> Dict[str, Any]:
        """Return a strict ``VehicleConfig`` or ``PedestrianConfig`` dict."""
        raise NotImplementedError


def register_runtime_entity(entity_type: str, *, physical_kind: str):
    """Register a scenario entity adapter with explicit physical semantics."""
    def decorator(cls: Type[RuntimeEntityAdapter]):
        if (not isinstance(entity_type, str)
                or not _NAME_RE.fullmatch(entity_type)):
            raise ValueError(
                "Runtime entity type must match [a-z][a-z0-9_]*")
        if physical_kind not in _PHYSICAL_KINDS:
            raise ValueError(
                f"physical_kind must be one of {sorted(_PHYSICAL_KINDS)}")
        if (not isinstance(cls, type)
                or not issubclass(cls, RuntimeEntityAdapter)):
            raise TypeError(
                "Runtime entity adapter must inherit RuntimeEntityAdapter")
        if entity_type in _REGISTRY:
            existing = _REGISTRY[entity_type]
            raise ValueError(
                f"Runtime entity {entity_type!r} is already registered by "
                f"{existing.__module__}.{existing.__name__}")
        cls.entity_type = entity_type
        cls.physical_kind = physical_kind
        _REGISTRY[entity_type] = cls
        return cls
    return decorator


def list_runtime_entity_types() -> List[str]:
    return sorted(_REGISTRY)


def describe_runtime_entity_type(entity_type: str) -> dict:
    try:
        adapter = _REGISTRY[entity_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown runtime entity {entity_type!r}; "
            f"available={sorted(_REGISTRY)}") from exc
    return {
        "entity_type": entity_type,
        "physical_kind": adapter.physical_kind,
        "adapter": f"{adapter.__module__}.{adapter.__name__}",
        "doc": (adapter.__doc__ or "").strip(),
    }


def expand_runtime_entities(
    specifications: List[dict],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Expand custom scenario entries before authoritative schema parsing."""
    if not isinstance(specifications, list):
        raise ValueError("entities must be a list")
    expanded = []
    for index, raw in enumerate(specifications):
        if not isinstance(raw, dict):
            raise ValueError(f"entities[{index}] must be an object")
        entity_type = raw.get("entity_type")
        if not isinstance(entity_type, str) or not entity_type:
            raise ValueError(
                f"entities[{index}].entity_type must be a non-empty string")
        try:
            adapter = _REGISTRY[entity_type]
        except KeyError as exc:
            raise ValueError(
                f"Unknown runtime entity {entity_type!r}; import its "
                f"extension first; available={sorted(_REGISTRY)}") from exc
        specification = dict(raw)
        specification.pop("entity_type")
        native = adapter.build_config(specification)
        if not isinstance(native, dict):
            raise TypeError(
                f"Runtime entity adapter {entity_type!r} must return a dict")
        expanded.append((adapter.physical_kind, dict(native)))
    return expanded


__all__ = [
    "RuntimeEntityAdapter",
    "describe_runtime_entity_type",
    "expand_runtime_entities",
    "list_runtime_entity_types",
    "register_runtime_entity",
]
