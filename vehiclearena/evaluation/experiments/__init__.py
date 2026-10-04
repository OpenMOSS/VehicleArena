"""Reproducible experiment pipelines for VehicleArena."""

from evaluation.experiments.manifest import (
    ExperimentManifest,
    ExperimentVariant,
    load_manifest,
)

__all__ = [
    "ExperimentManifest",
    "ExperimentVariant",
    "load_manifest",
]
from evaluation.experiments.scene_catalog import (
    CATALOG_SCHEMA,
    build_scene_assets,
    generate_scene_catalog,
    iter_catalog_scenes,
)
from evaluation.experiments.scene_validator import validate_scene_catalog
from evaluation.experiments.map_coverage import (
    MAP_COVERAGE_SCHEMA,
    generate_map_coverage,
    validate_map_coverage,
)
from evaluation.experiments.map_visual_audit import (
    VISUAL_AUDIT_SCHEMA,
    render_map_visual_audit,
)

__all__ = [
    "CATALOG_SCHEMA",
    "build_scene_assets",
    "generate_scene_catalog",
    "iter_catalog_scenes",
    "validate_scene_catalog",
    "MAP_COVERAGE_SCHEMA",
    "generate_map_coverage",
    "validate_map_coverage",
    "VISUAL_AUDIT_SCHEMA",
    "render_map_visual_audit",
]
