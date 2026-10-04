# Quickstart

[中文](../01-quickstart.md) · [English index](README.md)

Run commands from the repository root. Python 3.10+ and SUMO 1.25.0 are the frozen benchmark environment.

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install eclipse-sumo==1.25.0
python -m playwright install chromium
cd web3d && npm ci && cd ..
```

If no matching SUMO wheel exists, install SUMO 1.25.0 separately and make its executables and Python bindings available to this environment. No manual `PYTHONPATH` is required when using `scripts/run_experiments.py`.

## 2. Install the road maps

Download `vehiclearena-road-networks-2026.09.22.0.tar.gz` from the [VehicleArena dataset](https://huggingface.co/datasets/OpenMOSS-Team/VehicleArena). It matches `vehiclearena/simulation/map_bundle_manifest.json`; map files are not stored in Git.

```bash
MAP_BUNDLE=/absolute/path/to/vehiclearena-road-networks-2026.09.22.0.tar.gz
python scripts/manage_map_bundle.py install --source "$MAP_BUNDLE"
python scripts/check_environment.py
```

The current manifest expects 116 lane-level maps and 232 map files. The environment check must report `"ready": true`. Do not pass `--replace` for a first installation; use it only when intentionally replacing the installed bundle.

## 3. Run a physics-only scene

Manifests are generated artifacts; create them from the frozen scene catalog:

```bash
python scripts/run_experiments.py prepare-scene-manifests \
  --catalog vehiclearena/evaluation/experiments/scenarios \
  --output vehiclearena/evaluation/experiment_manifests/catalog

python scripts/run_experiments.py run \
  --manifest vehiclearena/evaluation/experiment_manifests/catalog/Basic.manifest.json \
  --output outputs/sumo-smoke \
  --variant basic_017_unsignalized_intersection \
  --sumo-reference
```

A `completed` entry in `status.json` and a `.json.gz` trajectory show that the scene, map conversion, SUMO process, and output pipeline ran. They do not by themselves prove safe arrival.

## 4. Run an LLM scene

Provide an OpenAI-compatible endpoint with tool calling and image input:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://your-endpoint.example.com/v1"
export OPENAI_MODEL="your-vision-tool-model"

python scripts/run_experiments.py run \
  --manifest vehiclearena/evaluation/experiment_manifests/catalog/Basic.manifest.json \
  --output outputs/llm-smoke \
  --variant basic_017_unsignalized_intersection \
  --allow-llm \
  --llm-base-url "$OPENAI_API_BASE" \
  --llm-model "$OPENAI_MODEL" \
  --llm-context-window-tokens 32768
```

Once driver tool calls work, add `--with-personal-agent`, `--personal-agent-model`, and `--passenger-judge-model` to exercise the passenger loop. For reproducible batch runs, use the [suite guide](llm-suite.md).

## 5. Development checks

```bash
python scripts/validate_extension.py \
  --extension docs.examples.fleet_extension \
  --scenario docs/examples/extended_scenario.json \
  --rules docs/examples/rules
python -m pytest -q
```

Record the code revision, map manifest, frozen scenes, prompts, tool schemas, model settings, and matched reference when publishing results.
