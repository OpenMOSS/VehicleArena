# Testing and release checklist

[中文](../reference/testing.md) · [English index](README.md)

Validate an extension and run the repository tests from the root with the virtual environment active:

```bash
python scripts/validate_extension.py \
  --extension your_package.vehiclearena_extension \
  --scenario path/to/scenario.json \
  --rules path/to/rules
python -m pytest -q
```

An extension should have positive and negative tests for duplicate names, unknown parameters, and invalid fields; each equipment profile's actual installed modules; chassis dimensions and dynamics reaching SUMO; strict `llm` versus `sumo` control authority; runtime entities appearing in collision/perception/render snapshots; reproducible scenario generation; and cabin-rule primary/alternative actions, missing equipment, and negative checks.

Keep the physics general: an extension should not branch on a particular scene or vehicle ID, directly change coordinates or collision state, silently replace a registration, load arbitrary Python from scenario JSON, or introduce global mutable per-entity state. An extension's import entry point should register components, not start a simulation or access the network.

Record the VehicleArena and extension revisions, map-bundle manifest/hash, frozen scenario and hash, model/runtime settings, raw trajectory and tool calls, and all independent evaluation dimensions. Official manifests use the `vehiclearena-experiment-manifest-v0.2` schema. A changed source, map, or scene requires a fresh manifest for that run; treatment/reference *pairing* has a separate `protocol_hash` compatibility check. Infrastructure-invalid data and missing references remain missing rather than being converted into model scores.
