"""
Module Plugin Registry for VehicleWorld.

Provides a central registry for vehicle modules so that:
  - New modules can be added via a single @register_module decorator
  - Third-party packages can register custom modules on import
  - tool discovery and evaluation module lists are derived from the registry

Usage (built-in module):
    from module.base_module import BaseModule
    from registry import register_module
    @register_module("wiper", description="Windshield wipers.", category="body")
    class Wiper(BaseModule): ...

Usage (third-party):
    from module.base_module import BaseModule
    from registry import register_module, register_gt_rule
    @register_module("dashcam", description="Dashboard camera", category="sensor")
    class DashCam(BaseModule): ...

    @register_gt_rule("dashcam")
    def dashcam_rules(event_bus, constraint_engine): ...
"""

import re
from dataclasses import dataclass, field
from typing import Type, Dict, List, Optional, Callable


_PUBLIC_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass
class ModuleDescriptor:
    """Metadata for a registered VehicleWorld module."""
    name: str
    cls: Type
    description: str
    category: str = "custom"
    is_external: bool = False
    needs_settings: bool = False
    gt_rules: List[Callable] = field(default_factory=list)
    constraints: List[Callable] = field(default_factory=list)


class ModuleRegistry:
    """Singleton registry of all VehicleWorld modules.

    Both built-in and third-party modules register here.
    The registry is queried at VehicleWorld init time to
    dynamically instantiate modules and wire up rules.
    """
    _instance: Optional["ModuleRegistry"] = None

    def __init__(self):
        self._modules: Dict[str, ModuleDescriptor] = {}

    @classmethod
    def instance(cls) -> "ModuleRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None

    # ── Registration ────────────────────────────────────────

    def register(self, name: str, cls: Type, description: str,
                 category: str = "custom", is_external: bool = False,
                 needs_settings: bool = False,
                 gt_rules: list = None, constraints: list = None):
        """Register a module class with its metadata.

        Registration is intentionally strict: silently replacing an existing
        module makes a scenario depend on import order and is not reproducible.
        """
        if not isinstance(name, str) or not _PUBLIC_NAME_RE.fullmatch(name):
            raise ValueError(
                "Module name must match [A-Za-z][A-Za-z0-9_]*")
        if not isinstance(cls, type):
            raise TypeError("Registered vehicle module must be a class")
        from module.base_module import BaseModule
        if not issubclass(cls, BaseModule):
            raise TypeError(
                f"Vehicle module {name!r} must inherit BaseModule")
        if name in self._modules:
            existing = self._modules[name]
            raise ValueError(
                f"Vehicle module {name!r} is already registered by "
                f"{existing.cls.__module__}.{existing.cls.__name__}")
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Module category must be a non-empty string")
        self._modules[name] = ModuleDescriptor(
            name=name,
            cls=cls,
            description=description,
            category=category,
            is_external=is_external,
            needs_settings=needs_settings,
            gt_rules=gt_rules or [],
            constraints=constraints or [],
        )
        _attach_pending(self, name)

    def unregister(self, name: str):
        """Remove a module from the registry."""
        self._modules.pop(name, None)

    # ── Queries ─────────────────────────────────────────────

    def get(self, name: str) -> Optional[ModuleDescriptor]:
        return self._modules.get(name)

    def all_modules(self) -> Dict[str, ModuleDescriptor]:
        return dict(self._modules)

    def internal_module_names(self) -> List[str]:
        """Module names that live directly on VehicleWorld (not ExternalWorld)."""
        return [n for n, d in self._modules.items() if not d.is_external]

    def external_module_names(self) -> List[str]:
        """Module names that live on ExternalWorld."""
        return [n for n, d in self._modules.items() if d.is_external]

    def needs_settings_module_names(self) -> List[str]:
        """Module names that require VehicleSettings injection."""
        return [n for n, d in self._modules.items() if d.needs_settings]

    def modules_dict(self) -> Dict[str, str]:
        """Return {name: description} dict (replaces utils.modules_dict)."""
        return {n: d.description for n, d in self._modules.items()}

    def modules_by_category(self) -> Dict[str, List[str]]:
        """Group module names by category."""
        groups: Dict[str, List[str]] = {}
        for name, desc in self._modules.items():
            groups.setdefault(desc.category, []).append(name)
        return groups

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __len__(self) -> int:
        return len(self._modules)


# ── Decorators ──────────────────────────────────────────────

def register_module(name: str, description: str = "",
                    category: str = "custom", is_external: bool = False,
                    needs_settings: bool = False):
    """Class decorator: register a VehicleWorld module.

    Example:
        @register_module("wiper", description="Windshield wipers.", category="body")
        class Wiper:
            ...
    """
    def decorator(cls):
        desc = description or cls.__doc__ or name
        # Clean up multiline docstrings to a single line
        if "\n" in desc:
            desc = desc.strip().split("\n")[0].strip()
        ModuleRegistry.instance().register(
            name=name, cls=cls, description=desc,
            category=category, is_external=is_external,
            needs_settings=needs_settings,
        )
        return cls
    return decorator


def register_gt_rule(module_name: str):
    """Function decorator: attach a GT rule function to a registered module.

    The decorated function should have signature:
        def my_rule(event_bus, constraint_engine): ...

    It will be called during VehicleWorld.__init__ to wire up rules.

    Example:
        @register_gt_rule("dashcam")
        def dashcam_night_rule(event_bus, constraint_engine):
            # register EventBus handlers / constraints
            ...
    """
    def decorator(fn):
        registry = ModuleRegistry.instance()
        desc = registry.get(module_name)
        if desc is not None:
            desc.gt_rules.append(fn)
        else:
            # Module not yet registered — store as pending
            # (will be picked up when the module registers later)
            _pending_gt_rules.setdefault(module_name, []).append(fn)
        return fn
    return decorator


def register_constraint(module_name: str):
    """Function decorator: attach a constraint function to a registered module.

    The decorated function should have signature:
        def my_constraint(constraint_engine): ...
    """
    def decorator(fn):
        registry = ModuleRegistry.instance()
        desc = registry.get(module_name)
        if desc is not None:
            desc.constraints.append(fn)
        else:
            _pending_constraints.setdefault(module_name, []).append(fn)
        return fn
    return decorator


# Pending rules/constraints for modules not yet registered
_pending_gt_rules: Dict[str, List[Callable]] = {}
_pending_constraints: Dict[str, List[Callable]] = {}


def _attach_pending(registry: ModuleRegistry, module_name: str) -> None:
    """Attach hooks regardless of whether hook or module imported first."""
    desc = registry.get(module_name)
    if desc is None:
        return
    desc.gt_rules.extend(_pending_gt_rules.pop(module_name, []))
    desc.constraints.extend(_pending_constraints.pop(module_name, []))
