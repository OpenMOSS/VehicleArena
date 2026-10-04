# 五分钟开始

[English version](en/quickstart.md)

本页给出从源码 checkout 到第一个完整 SUMO episode 的最短路径。所有命令都在仓库根目录执行，不需要手工设置 `PYTHONPATH`。

## 1. Python 与 SUMO

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
python -m pip install eclipse-sumo==1.25.0
cd web3d && npm ci && cd ..
```

冻结实验以 SUMO 1.25.0 为参考。其他版本可用于开发，但正式模型对比必须记录版本并重新生成配对基准。

## 2. 安装离线地图

从 [VehicleArena 地图数据集](https://huggingface.co/datasets/OpenMOSS-Team/VehicleArena)下载与仓库内 `vehiclearena/simulation/map_bundle_manifest.json` 对应的 `vehiclearena-road-networks-2026.09.22.0.tar.gz`：

```bash
MAP_BUNDLE=/absolute/path/to/vehiclearena-road-networks-2026.09.22.0.tar.gz
python scripts/manage_map_bundle.py install --source "$MAP_BUNDLE"
python scripts/check_environment.py
```

第一次安装不要使用 `--replace`。只有明确替换本地整套地图时才增加该选项。成功检查应包含：

```json
{
  "ready": true,
  "maps": {
    "complete": true,
    "installed_files": 232,
    "lane_level_count": 116
  }
}
```

## 3. 首次物理运行

manifest 是生成物，不进入 Git；从已经冻结的场景目录重建：

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

`status.json` 中该 variant 为 `completed`，且生成 `.json.gz` 轨迹，即表示场景、地图转换、SUMO 物理和评测链路都能工作。

## 4. 首次 LLM 运行

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://your-endpoint.example.com/v1"
export OPENAI_MODEL="your-function-calling-and-vision-model"

python scripts/run_experiments.py run \
  --manifest vehiclearena/evaluation/experiment_manifests/catalog/Basic.manifest.json \
  --output outputs/llm-smoke \
  --variant basic_017_unsignalized_intersection \
  --allow-llm \
  --llm-base-url "$OPENAI_API_BASE" \
  --llm-model "$OPENAI_MODEL" \
  --llm-context-window-tokens 32768
```

确认 Driving Agent 能完成工具调用后，再增加 `--with-personal-agent`、`--personal-agent-model` 和 `--passenger-judge-model`，避免首次排错同时涉及三个模型角色。

## 5. 验证扩展开发环境

```bash
python scripts/validate_extension.py \
  --extension docs.examples.fleet_extension \
  --scenario docs/examples/extended_scenario.json \
  --rules docs/examples/rules

python -m pytest -q
```

生产实验应冻结 VehicleArena commit、地图 manifest、场景 JSON、Prompt、工具 Schema、模型配置和配对参考运行。正式划分为 100 个训练/开发任务及 112 个测试任务（80 Basic、32 MultiLLM）。批量运行和参考运行见[批量 LLM 测试](llm-suite.md)。
