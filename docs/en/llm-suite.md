# Batch LLM runs and matched references

[中文](../llm-suite.md) · [English index](README.md)

`scripts/run_llm_suite.py` launches one process per selected scene and stores frozen manifests, model-call logs, images, trajectories, and attempt history. Install the dependencies and map bundle first. Never store API keys in the repository.

## Choose scenes

The published benchmark split contains 100 training/development tasks and 112 held-out tests (80 Basic, 32 MultiLLM). Without `--only` or `--selection-file`, the generic runner submits the entire scene directory; use the fixed test entry point below for the evaluation split. To run two examples:

```bash
export OPENAI_API_KEY="your-api-key"
python scripts/run_llm_suite.py \
  --output outputs/my-llm-suite \
  --base-url https://your-endpoint.example/v1 \
  --model your-vision-tool-model \
  --workers 5 --attempts 2 --prepare \
  --only basic_033_continuous_turns \
  --only multi_030_unprotected_left_turn__hongkong_central
```

To run the fixed evaluation split of 80 Basic and 32 MultiLLM scenes, use:

```bash
python scripts/run_test_suite.py \
  --output outputs/my-test-suite \
  --base-url https://your-endpoint.example/v1 \
  --model your-vision-tool-model \
  --workers 5
```

The wrapper validates the 112-scene selection and schedules active Basic:MultiLLM workers near 2:1. `--workers` limits scene processes, not aggregate API requests. Each scene allows up to four simultaneous model calls. Use a new output directory whenever code, scene selection, or model configuration changes; to resume the *same* batch, omit `--prepare` and keep its settings. Use `--adopt-active` when the original workers are still alive.

## Model roles and defaults

`--model` selects only the evaluated Driving Agent. Fixed peer drivers, Personal Agent, and Passenger Judge default to `Qwen3.8-27B`; use `--peer-model`, `--personal-agent-model`, and `--passenger-judge-model` to override individual roles. Endpoint options include `--peer-base-url` and `--aux-base-url`; keep all role configurations fixed across a model comparison except for the evaluated driver. MultiLLM peers receive rotating cautious, efficient, cooperative, and comfortable driving prompts.

The current suite defaults to a 1,000,000-token context window, 32,768 maximum output tokens, enabled thinking, temperature 0.7 for drivers/PA, and 0.0 for Judge. The PA uses event/random wakes. Judge checks a request at +0.1, +1, and +3 simulation seconds, at most three times. If your provider rejects a parameter, explicitly override it and retain the effective configuration in the result artifact.

## Follow progress and inspect results

```bash
python scripts/monitor_llm_suite.py outputs/my-test-suite
python scripts/summarize_llm_suite.py outputs/my-test-suite
python scripts/analyze_llm_sample.py --help
```

`configuration.json` records role settings and selected scenes; `manifests/` contains frozen inputs; `logs/` contains per-attempt process/model details; `suite-status.json` is a checkpoint; `Basic/` and `MultiLLM/` contain compressed trajectories and observation images. `completed` means that a valid trajectory was produced, not that the evaluated vehicle arrived safely. Check arrival, collisions, final states, and evaluation separately. Infrastructure errors may be retried; collisions and route failures remain model outcomes.

## Run a matched reference

A matched reference measures what would happen with SUMO driving the evaluated vehicle. In Basic, this makes all vehicles SUMO-controlled. In MultiLLM, fixed peer agents remain on their treatment-run model, personality prompt, and PA/Judge configuration. The physical inputs match, but traffic may respond differently to the changed evaluated driver.

```bash
python scripts/run_reference_suite.py \
  --source-output outputs/my-test-suite \
  --output outputs/my-test-reference \
  --base-url https://your-peer-endpoint.example/v1 \
  --model Qwen3.8-27B \
  --workers 5
```

The reference runner copies the treatment manifests and fixed-role behavior settings. Endpoint addresses may differ when they serve the same model and configuration. Pair results by `variant_id` and a validated `protocol_hash`; an absent or incompatible reference stays missing rather than becoming zero traffic impact. See the [evaluation guide](evaluation.md) for metrics and deadline calibration.
