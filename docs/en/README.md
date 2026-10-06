# VehicleArena documentation

[中文文档](../README.md) · [Project README](../../README.en.md) · [Paper](https://arxiv.org/abs/2609.35916)

Start with the [quickstart](quickstart.md), then read the [architecture](architecture.md), [batch LLM guide](llm-suite.md), and [evaluation protocol](evaluation.md). These pages describe the current implementation and frozen benchmark.

| Topic | English guide | Detailed source |
|---|---|---|
| Installation and first run | [Quickstart](quickstart.md) | [Chinese quickstart](../01-quickstart.md) |
| Agent/world boundary | [Architecture](architecture.md) | [Chinese architecture](../02-architecture.md) |
| Batch jobs and paired references | [LLM suite](llm-suite.md) | [Chinese LLM suite](../llm-suite.md) |
| Tasks, calibration, and metrics | [Evaluation](evaluation.md) | [Experiment pipeline](../../vehiclearena/evaluation/experiments/README.md) |
| Scenario JSON and SUMO | [Scenario schema](scenario-schema.md), [SUMO engine](sumo-engine.md) | [Reference overview](reference.md) |
| Custom equipment, entities, and rules | [Vehicle modules](vehicle-modules.md), [runtime entities](runtime-entities.md), [scene generation](scenario-generation.md), [cabin rules](cabin-rules.md) | [Extension overview](reference.md) |
| Tests and extension release | [Testing checklist](testing.md) | [Chinese testing guide](../reference/testing.md) |
| 3D scene viewer and live display | [Web3D viewer](web3d.md) | [Chinese Web3D guide](../../web3d/README.md) |
| Offline replay, images, and GIFs | [Rendering](rendering.md) | [Chinese rendering guide](../reference/rendering.md) |

The published split contains 100 training/development tasks and 112 held-out evaluation tasks (80 Basic, 32 MultiLLM). Frozen time limits are stored in `vehiclearena/evaluation/experiments/time_window_calibration.json`; do not infer task counts from old script filenames or dated notes.
