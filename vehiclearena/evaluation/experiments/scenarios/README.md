# VehicleArena 正式实验场景

[English evaluation guide](../../../../docs/en/evaluation.md)

正式划分为 100 个训练/开发任务和 112 个测试任务（80 Basic、32 MultiLLM）。

每个场景目录包含 `scenario.json`（可执行物理初态）和 `expected.json`（机器可读验收条件）。每个实验的 `matrix.json` 定义模型替换规则；公平比较依赖同一份冻结物理场景，不使用运行时随机种子。

正式测试使用 `../time_window_calibration.json` 中的冻结 `total_time_s`。当前 112 项测试均为所需 SUMO 车辆最晚终态时间加 10 个仿真秒；重新运行校准代码时还会考虑最后一个计入校准的预设事件，可能与这些冻结值不同。复现论文时不要重新校准。

维护者生成（会改写场景，请勿在普通测试前执行）：`python scripts/run_experiments.py prepare-scenes --output vehiclearena/evaluation/experiments/scenarios`

校验：`python scripts/run_experiments.py validate-scenes --catalog vehiclearena/evaluation/experiments/scenarios --boot`
