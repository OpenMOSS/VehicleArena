# 批量 LLM 测试

[English version](en/llm-suite.md)

`scripts/run_llm_suite.py` 为每个场景启动独立进程，保存冻结 manifest、
模型调用记录、图像、轨迹和逐次尝试日志。需要先按快速开始安装依赖和地图包。

## 启动

密钥通过环境变量传入，不写入脚本、配置或 Git。正式划分为 100 个训练/开发任务
和 112 个测试任务（80 Basic、32 MultiLLM）。通用 runner 的无筛选运行会遍历
整个场景目录；论文测试应使用下方的固定测试集入口。时间窗已记录在
`vehiclearena/evaluation/experiments/time_window_calibration.json`。
例如运行两个场景（不传 `--only` 或 `--selection-file` 会运行整个场景目录，而不是固定测试集）：

```bash
export OPENAI_API_KEY="your-api-key"
python scripts/run_llm_suite.py \
  --output outputs/my-llm-suite \
  --base-url https://your-endpoint.example/v1 \
  --model your-vision-tool-model \
  --context-window 32768 \
  --workers 5 --attempts 2 --prepare \
  --only basic_033_continuous_turns \
  --only multi_030_unprotected_left_turn__hongkong_central
```

每批使用新的输出目录，避免混用不同代码版本的结果。后台运行可在命令前加
`nohup`，将标准输出和错误重定向到本地日志，并在末尾加 `&`。仅当模型服务
需要绕过环境代理时添加 `--direct-api`。

脚本默认启用 PA 与 Judge。`--model` 只配置被测 Driving Agent；固定 peer、PA、
Judge 默认分别使用 `Qwen3.8-27B`，可通过独立选项覆盖。多车轮流分配
cautious、efficient、cooperative、comfortable 驾驶模板。默认启用 thinking，
上下文窗口为 1,000,000 tokens，单次输出上限 32,768 tokens；Judge 温度为 0，
其余角色为 0.7。PA 为事件/随机触发，Judge 默认在请求后 +0.1、+1、+3 仿真秒
检查，最多 3 次。模型服务不支持这些参数时，应显式调整并在报告中保留配置。

固定测试集可用如下入口运行；脚本会严格检查 80 个 Basic 和 32 个 MultiLLM
场景，并按约 2:1 保持活动场景比例：

```bash
python scripts/run_test_suite.py \
  --output outputs/my-test-suite \
  --base-url https://your-endpoint.example/v1 \
  --model your-vision-tool-model \
  --workers 5
```

`--workers 5` 表示最多 5 个场景进程，不是总 API 请求上限。每个场景允许
最多 4 个模型请求同时执行，因此这组配置最多产生 20 个并行请求。

## 进度与结果

```bash
python scripts/monitor_llm_suite.py outputs/my-llm-suite
python scripts/summarize_llm_suite.py outputs/my-llm-suite
python scripts/analyze_llm_sample.py --help
```

- `configuration.json`：模型参数、场景选择与驾驶模板。
- `manifests/`：冻结的场景、来源指纹和模板分配。
- `logs/`：逐场景、逐次尝试的模型调用与进程日志。
- `suite-status.json`：已结束任务的检查点；`missing` 可能仍在运行或排队，
  需用 monitor 区分。
- `Basic/`、`MultiLLM/`：压缩轨迹及 `observations/` 图片。
- `attempts/`：重试前已有轨迹和图像的归档，不覆盖历史失败证据。

`completed` 只表示任务生成了有效轨迹，不等于车辆全部到达或安全通过。
必须另外检查 `all_vehicles_arrived`、碰撞、车辆终态和评分。接口/进程失败
可按 `--attempts` 重试；驾驶策略导致的碰撞、走错路和超时不靠重试抹掉。

进度诊断使用模型请求起止日志、控制器检查点和进程存活状态；不启用定时
全线程堆栈转储（曾在 CPython 线程帧检查中触发 SIGSEGV）。仅保留致命错误
的当前线程转储。控制器原有 `--timeout` 进程超时及重试机制不变。

恢复同一代码版本的批次时去掉 `--prepare`，保持模型和场景选择参数一致；
若旧 worker 仍存活，需显式 `--adopt-active`，不要重复启动同一场景。
代码或场景改变后应新建批次，不修改旧轨迹指纹以冒充新版本结果。

输出目录、日志、地图 JSON、密钥配置和崩溃转储均应保留在本地，不提交 Git。

## 配对参考运行

为区分场景本身的交通困难与被测车带来的影响，对相同场景再运行一遍参考组。
Basic 的参考组把全部 LLM 车辆换成 SUMO；MultiLLM 只把被测车换成 SUMO，
其他 LLM 车辆保持与测试组相同的模型、性格模板及 PA/Judge 配置。
两组共享冻结场景及背景交通设置，但车辆轨迹可以随交互而不同。
参考组不是一个额外被评分的模型。

```bash
python scripts/run_reference_suite.py \
  --source-output outputs/my-test-suite \
  --output outputs/my-test-reference \
  --base-url https://your-peer-endpoint.example/v1 \
  --model Qwen3.8-27B \
  --workers 5
```

参考运行可使用不同的服务地址，但固定 peer 的行为配置必须与测试组一致。
聚合器按同一 `variant_id` 和有效 `protocol_hash` 配对；缺失或不兼容的参考结果
保持缺失，不当作零影响。完整场景时间窗校准的来源和例外见
[实验管线](../vehiclearena/evaluation/experiments/README.md)。
