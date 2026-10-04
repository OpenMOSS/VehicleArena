from module import *  # noqa: F401 — triggers @register_module decorators
from module.vehicle_settings import VehicleSettings
from registry import ModuleRegistry
from event_bus import EventBus
from constraints import ConstraintEngine
from coupling_rules import register_all_rules
from external_world import ExternalWorld
from capabilities import (
    VehicleCapabilitySet,
    action_module_name,
    resolve_vehicle_capabilities,
)


class VehicleWorld:
    def __init__(
        self,
        capability_set: VehicleCapabilitySet = None,
        equipment_profile: str = "executive",
        chassis_profile: str = "sedan",
        enable_modules=None,
        disable_modules=None,
        chassis_overrides=None,
    ):
        self.capabilities = capability_set or resolve_vehicle_capabilities(
            equipment_profile=equipment_profile,
            chassis_profile=chassis_profile,
            enable_modules=enable_modules,
            disable_modules=disable_modules,
            chassis_overrides=chassis_overrides,
        )

        # ── Vehicle Settings (shared state inside this vehicle) ──
        self.settings = VehicleSettings()

        registry = ModuleRegistry.instance()
        selected_external = {
            name: descriptor
            for name, descriptor in registry.all_modules().items()
            if descriptor.is_external and self.capabilities.has_module(name)
        }
        self.externalWorld = ExternalWorld(selected_external)

        # Instantiate only modules installed on this vehicle.
        self._module_names = []
        self._external_module_names = []
        for name, desc in registry.all_modules().items():
            if not self.capabilities.has_module(name):
                continue
            if desc.is_external:
                self._external_module_names.append(name)
                continue
            mod = desc.cls()
            setattr(self, name, mod)
            self._module_names.append(name)

        for name in registry.needs_settings_module_names():
            mod = self._get_module(name)
            if mod is not None and hasattr(mod, "set_settings"):
                mod.set_settings(self.settings)

        self._event_bus = EventBus()
        self._constraint_engine = ConstraintEngine()
        self._constraint_engine.set_vehicle_world(self)
        for mod in self._iter_all_modules():
            mod._event_bus = self._event_bus
            mod._constraint_engine = self._constraint_engine

        # Coupling callbacks are resilient to unavailable optional modules.
        # Modules that do not exist cannot publish their own source events.
        register_all_rules(self._event_bus, self._constraint_engine)
        for name in self.available_module_names():
            descriptor = registry.get(name)
            if descriptor is None:
                continue
            for install_rule in descriptor.gt_rules:
                install_rule(self._event_bus, self._constraint_engine)
            for install_constraint in descriptor.constraints:
                install_constraint(self._constraint_engine)

        self._simulation = None

    def __getattr__(self, name):
        """Expose selected external modules through the normal module path."""
        external_world = self.__dict__.get("externalWorld")
        if external_world is not None and hasattr(external_world, name):
            return getattr(external_world, name)
        raise AttributeError(
            f"{type(self).__name__!s} has no module {name!r}")

    def _get_module(self, name):
        """Get a module by its registry name, checking both local and ExternalWorld."""
        if not self.has_module(name):
            return None
        if hasattr(self, name):
            return getattr(self, name)
        if hasattr(self.externalWorld, name):
            return getattr(self.externalWorld, name)
        return None

    def _iter_all_modules(self):
        """Iterate over all module instances (internal + external)."""
        for name in self._module_names:
            yield getattr(self, name)
        for attr in self._external_module_names:
            yield getattr(self.externalWorld, attr)

    def has_module(self, name: str) -> bool:
        """Return whether this exact vehicle has a module installed."""
        return self.capabilities.has_module(name)

    def available_module_names(self):
        """Sorted installed-module names used by tools and evaluation."""
        return sorted(self.capabilities.modules)

    def capability_snapshot(self) -> dict:
        """Serializable immutable equipment/chassis configuration."""
        return self.capabilities.to_dict()

    def filter_capability_code(self, code: str):
        """Drop one-line ``vw.<module>`` initialisers for absent equipment."""
        kept, skipped = [], []
        for line in (code or "").splitlines():
            module = action_module_name(line)
            if module and not self.has_module(module):
                skipped.append({
                    "action": line,
                    "module": module,
                    "reason": "capability_not_available",
                })
            else:
                kept.append(line)
        return "\n".join(kept), skipped

    # ── Flat access to ExternalWorld sub-modules ──
    # Allows ``vw.weather`` alongside ``vw.externalWorld.weather``.
    @property
    def weather(self):
        return self.externalWorld.weather

    @weather.setter
    def weather(self, value):
        self.externalWorld.weather = value

    @property
    def dayNight(self):
        return self.externalWorld.dayNight

    @dayNight.setter
    def dayNight(self, value):
        self.externalWorld.dayNight = value

    @property
    def speedLimit(self):
        return self.externalWorld.speedLimit

    @speedLimit.setter
    def speedLimit(self, value):
        self.externalWorld.speedLimit = value

    @property
    def map(self):
        return self.externalWorld.map

    @map.setter
    def map(self, value):
        self.externalWorld.map = value

    @property
    def is_simulating(self) -> bool:
        """True when running inside a SimulationEngine."""
        return self._simulation is not None

    def get_modules(self):
        """Returns a dict of all available module names and their descriptions."""
        descriptions = ModuleRegistry.instance().modules_dict()
        return {
            name: descriptions[name]
            for name in self.available_module_names()
            if name in descriptions
        }

    def get_module_API(self, modules):
        """Returns detailed API signatures and documentation for the specified modules.

        Args:
            modules (list[str]): List of module names to look up, e.g. ["fogLight", "weather"].

        Returns:
            str: Formatted API documentation for the requested modules.
        """
        from utils import get_api_content
        requested = list(modules or ())
        unavailable = [
            module for module in requested if not self.has_module(module)]
        if unavailable:
            return {
                "success": False,
                "error": "capability_not_available",
                "modules": sorted(unavailable),
                "equipment_profile":
                    self.capabilities.equipment_profile,
            }
        return get_api_content(modules=requested)
