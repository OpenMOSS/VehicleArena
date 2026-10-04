<h1 align="center">VehicleArena</h1>

<p align="center"><strong>A Realistic Urban Environment for Multi-Agent Driving</strong></p>

<div align="center">
  <a href="https://github.com/OpenMOSS/VehicleArena"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&amp;logoColor=white" alt="GitHub code"></a>
  <a href="https://huggingface.co/datasets/OpenMOSS-Team/VehicleArena"><img src="https://img.shields.io/badge/Hugging%20Face-Maps-FFD21E?logo=huggingface&amp;logoColor=black" alt="Hugging Face map data"></a>
  <a href="https://arxiv.org/abs/2609.35916"><img src="https://img.shields.io/badge/arXiv-2609.35916-B31B1B?logo=arxiv&amp;logoColor=white" alt="arXiv paper"></a>
  <a href="https://arxiv.org/pdf/2609.35916"><img src="https://img.shields.io/badge/PDF-Download-D32F2F?logo=adobeacrobatreader&amp;logoColor=white" alt="Download paper PDF"></a>
</div>

<p align="center"><a href="README.en.md">English documentation</a> · <a href="docs/README.md">中文文档</a></p>

VehicleArena 是一个持续运行的 3D 城市驾驶环境：Driving Agent 一边应对交通，一边处理乘客请求；多个目标独立的 Agent 通过共享道路相互影响。模型负责决策，SUMO 负责执行并记录物理后果。

**从这里开始：** [快速开始](#-快速开始) · [LLM 测试](#测试-llm-场景) · [评测与结果](#-评测与结果) · [完整文档](docs/README.md)

## 🌟 亮点

- **共享物理世界：** SUMO 以固定 `0.1 s` 步长持续推动车辆和行人；同步的 Web3D 视角让 Agent 观察真实执行结果。
- **驾驶与乘客协同：** Personal Agent 提出即时或延时请求，Driving Agent 自主决定何时唤醒、如何驾驶与处理请求，独立 Judge 根据世界状态验收。
- **决策不能代替执行：** LLM 可请求速度、制动、变道、路口动作和座舱操作，但不能直接修改车辆位置，也不能靠文字声明完成任务。
- **多主体交互：** 场景可同时运行多个独立 Driving Agent；其余车辆与行人保持 SUMO 原生控制。车辆底盘和传感器装备可配置。
- **分项评测：** 100 个训练/开发任务与 112 个测试任务，分别报告到达、驾驶、乘客请求、座舱和交通影响，不合成总分。

## 🎬 系统概览

![VehicleArena 的共享 3D 环境、Agent 角色与事件时间线](docs/assets/paper-architecture.png)

图 1：共享 3D 世界中的 Driving Agent、Personal Agent 和 Judge。下方时间线示意乘客延时请求、DA 自主唤醒与独立验收；时间间隔不按比例绘制。

![Driving Agent 唤醒、乘客请求与 SUMO 持续物理执行的示例](docs/assets/paper-da-loop.png)

图 2：一次驾驶与车内服务交织的示例。DA 在心跳或交通事件时唤醒，SUMO 在两次唤醒之间仍按 0.1 秒步长推动车辆；图中压缩了时间间隔并省略了部分唤醒。

## 🚀 快速开始

### 环境安装

要求 Python 3.10+。冻结基准使用 SUMO `1.25.0`。

```bash
git clone git@github.com:OpenMOSS/VehicleArena.git
cd VehicleArena

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium

# CameraVisual 使用仓库内的 Web3D 车内视角。
cd web3d && npm ci && cd ..

# 推荐使用 SUMO 官方 Python wheel；它同时提供 sumo、netconvert 和 Python bindings。
python -m pip install eclipse-sumo==1.25.0
```

若当前平台没有 `eclipse-sumo==1.25.0` wheel，请按 SUMO 官方方式安装同版本，并保证 `sumo`、`netconvert`、`libsumo`、`traci` 和 `sumolib` 都对当前虚拟环境可见。

### 安装地图

地图不进入 Git。从 [VehicleArena 的 Hugging Face 数据集仓库](https://huggingface.co/datasets/OpenMOSS-Team/VehicleArena)下载 `vehiclearena-road-networks-2026.09.22.0.tar.gz`：

```bash
MAP_BUNDLE=/absolute/path/to/vehiclearena-road-networks-2026.09.22.0.tar.gz
python scripts/manage_map_bundle.py install --source "$MAP_BUNDLE"
python scripts/check_environment.py
```

安装器会使用仓库内 manifest 校验归档 SHA-256 和每个地图文件。`check_environment.py` 输出 `"ready": true` 才表示 Python、SUMO 和 116 张高精度地图均可用。

### 运行 SUMO 冒烟测试

先运行一个不调用 LLM 的完整物理冒烟：

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

### 测试 LLM 场景

模型接口必须兼容 OpenAI Chat Completions，并支持 function calling 和图片输入。每次唤醒的 `CameraVisual` 来自同步 Web3D 车内视角；仅安装 `lidar` 的车辆额外收到同一冻结场景的 35° 斜俯视 `LidarBEV`：本车蓝色车身、其余车辆整车浅灰，无信号灯状态、灯光或天气效果。它是处理后的 3D 几何图，不是原始点云。详见 [3D 局部观察图](docs/web3d-lidar-bev.md)。

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://your-endpoint.example.com/v1"
export OPENAI_MODEL="your-model"

python scripts/run_experiments.py run \
  --manifest vehiclearena/evaluation/experiment_manifests/catalog/Basic.manifest.json \
  --output outputs/model-smoke \
  --variant basic_017_unsignalized_intersection \
  --allow-llm \
  --llm-base-url "$OPENAI_API_BASE" \
  --llm-model "$OPENAI_MODEL" \
  --llm-context-window-tokens 32768 \
  --with-personal-agent \
  --personal-agent-model "$OPENAI_MODEL" \
  --passenger-judge-model "$OPENAI_MODEL"
```

部分服务会限制 temperature、thinking 或 reasoning 参数；使用对应 CLI 选项显式传入。不要把真实密钥写进场景、源码或提交记录。

## 📊 评测与结果

### 任务划分

| 划分 | 数量 | 设置 |
|---|---:|---|
| 训练/开发 | 100 | Basic 单车场景 |
| 测试：Basic | 80 | 1 辆待评 LLM；其余车辆和行人由 SUMO 控制 |
| 测试：MultiLLM | 32 | 固定其他 LLM 车辆配置，只替换被测车的模型 |

正式测试集共 **112** 个任务，训练/开发集与测试集不重叠。使用 `scripts/run_test_suite.py` 运行固定测试集；通用 runner 若不指定场景选择，会遍历整个场景目录。

### SUMO baseline 如何确定时限

发布任务集时，先为每个场景做一次时限校准：SUMO 接管待评车辆，作为驾驶 oracle。Basic 中其他车辆也由 SUMO 驾驶；MultiLLM 中其他 Agent 保持与模型测试时相同的模型、性格模板及 PA/Judge 配置。校准只等待需要完成行程的 SUMO 车辆，不等待固定的 LLM peer；预置永久障碍不计入到达目标。

论文测试使用已冻结的逐场景 `total_time_s`，来源为 [`time_window_calibration.json`](vehiclearena/evaluation/experiments/time_window_calibration.json)。当前 112 个测试任务的保存值均为校准运行中上述 SUMO 车辆的最晚终态时间加 **10 个仿真秒**；终态包括到达或碰撞，现有测试校准运行未记录碰撞。更换被测 Driving Agent 模型或重复运行同一任务时，直接使用这个时限，不为每个模型重新跑校准。

仅当校准输入变化时才重新校准受影响场景：地图包中的道路数据改变，新增或修改 `scenario.json`，或改变 SUMO 版本、仿真/校准代码、MultiLLM 固定 peer 的模型及 PA/Judge 配置。只更换被测模型、修改文档或展示方式，不需要重跑。实现见 [`time_window_calibration.py`](vehiclearena/evaluation/experiments/time_window_calibration.py)；复现已发布结果时使用冻结时限。

### 论文主表：被测车结果

下图直接截取论文主表。每个模型在 80 个 Basic 和 32 个 Multi-Agent 测试任务上各运行 3 次。Arr. 为无碰撞且无路线失败的按时到达率（%）；Drive、Req.（PA/Judge 请求满足度）和 Cabin（车内设备规则）为 0–100 分，均为越高越好。指标分别报告，不合成总分。

![论文主表：九个模型在 Basic 和 Multi-Agent 任务中的到达率、驾驶、请求和车内设备得分](docs/assets/paper-main-results.png)

到达率最高的是 Qwen3.8-Max，但驾驶、乘客请求和车内设备三类分数的领先模型并不相同，因此单一分数不能概括模型表现。

固定测试集跑完后，可从同一批冻结 manifest 和角色配置生成配对参考运行，用于交通影响指标：

```bash
python scripts/run_reference_suite.py \
  --source-output outputs/my-test-suite \
  --output outputs/my-test-reference \
  --base-url https://your-peer-endpoint.example/v1 \
  --model Qwen3.8-27B
```

MultiLLM 运行必须至少提供一个 `--peer-model`，确保实验只改变被测车。其配对参考运行只把被测车交给 SUMO，其他 LLM 车辆仍保留固定模型配置；Basic 参考运行则全部交给 SUMO。普通使用者不需要重新生成场景或时间窗；维护者按上文条件处理校准输入的变化。详见[批量测试与配对参考](docs/llm-suite.md)。

## 🛠 开发与扩展

```bash
# 全量回归
python -m pytest -q

# 验证官方扩展示例
python scripts/validate_extension.py \
  --extension docs.examples.fleet_extension \
  --scenario docs/examples/extended_scenario.json \
  --rules docs/examples/rules
```

支持扩展车辆/座舱模块、装备档案、底盘参数、运行实体适配器、场景层和 YAML 车内规则。背景交通策略不属于 Python 扩展面；需要评测新的驾驶策略时，应接入 LLM 控制角色，或在 SUMO 侧定义背景交通行为。

网页三维显示器可以查看高精地图，也可以实时显示 SUMO 仿真：

```bash
cd web3d
npm ci
python3 server.py
```

另一个终端可运行：

```bash
python3 web3d/run_live.py \
  --scenario vehiclearena/evaluation/experiments/scenarios/Basic/basic_017_unsignalized_intersection/scenario.json \
  --session-id live --duration-s 15
```

道路几何和动态主体都来自 VehicleArena/SUMO；实时 LLM 实验参数及接口边界见 [Web3D 说明](web3d/README.md)。

## 📚 文档

- [五分钟开始](docs/01-quickstart.md)
- [English quickstart](docs/en/quickstart.md)
- [开发者文档索引](docs/README.md)
- [框架边界与数据流](docs/02-architecture.md)
- [场景字段](docs/reference/scenario-schema.md)
- [SUMO 执行引擎](docs/reference/sumo-engine.md)
- [评测协议](docs/en/evaluation.md)

## 📁 目录

```text
vehiclearena/
├── simulation/       # SUMO 执行层、地图转换、实体状态
├── module/           # 车辆、座舱、导航和感知工具
├── evaluation/       # Harness、两组实验、评测和回归
└── visualization/    # 驾驶视图、地图截图和 GIF
docs/                 # 使用、评测与扩展文档
scripts/              # 环境检查、资产管理和实验入口
web3d/                 # 基于 Three.js 的 SUMO 实时三维显示器
```

## 📎 引用

如果 VehicleArena 对你的研究有帮助，请引用[论文](https://arxiv.org/abs/2609.35916)：

```bibtex
@article{yang2026vehiclearena,
  title={VehicleArena: A Realistic Urban Environment for Multi-Agent Driving},
  author={Yang, Jie and Chen, Jiajun and Zhou, Jiazheng and Huang, Mianqiu and Zheng, Yining and Wang, Yuxin and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2609.35916},
  year={2026}
}
```
