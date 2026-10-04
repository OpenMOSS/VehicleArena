"""Optional LiDAR equipment backing the processed local BEV."""

from dataclasses import asdict, dataclass, replace

from module.base_module import BaseModule
from registry import register_module


@dataclass(frozen=True)
class LidarSpec:
    range_m: float = 80.0
    horizontal_fov_deg: float = 160.0

    def with_overrides(self, overrides=None):
        if not overrides:
            return self
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown lidar override fields: {sorted(unknown)}")
        spec = replace(self, **dict(overrides))
        if not 10.0 <= float(spec.range_m) <= 300.0:
            raise ValueError("lidar range_m must be between 10 and 300")
        if not 30.0 <= float(spec.horizontal_fov_deg) <= 360.0:
            raise ValueError(
                "lidar horizontal_fov_deg must be between 30 and 360")
        return spec

    def as_dict(self):
        return asdict(self)


DEFAULT_LIDAR_SPEC = LidarSpec()


def resolve_lidar_spec(module_name="lidar", overrides=None):
    if module_name != "lidar":
        raise ValueError(f"unknown lidar module: {module_name}")
    return DEFAULT_LIDAR_SPEC.with_overrides(overrides)


@register_module(
    "lidar",
    description=(
        "Optional processed local 3D geometry display, shown from behind "
        "and above the vehicle with configured range and field of view"),
    category="sensor",
)
class Lidar(BaseModule):
    """Installed LiDAR configuration.

    The engine renders the frozen Web3D geometry at an LLM wake, sharing the
    cockpit scene. This is not a raw point cloud or a sensor-origin ray/occlusion
    simulation. Signal state, lamp effects, weather and day/night appearance
    are excluded. A fixed 8 m immediate ego context surrounds the body; outside
    it, the display is masked by the configured planar range and field of view.
    """

    def __init__(self):
        self._overrides = {}

    def configure(self, overrides=None):
        self._overrides = dict(overrides or {})


__all__ = ["DEFAULT_LIDAR_SPEC", "Lidar", "LidarSpec", "resolve_lidar_spec"]
