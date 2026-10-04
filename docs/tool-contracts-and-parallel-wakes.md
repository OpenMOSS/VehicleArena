# 工具契约与同场景多车并发

## 工具契约

- 从实际 Python 签名解析 Optional、Union、Literal 和数组元素类型；Optional 数值不再误报成字符串。可省略参数由签名默认值决定。
- 读取原始方法文档，兼容 `Args: name: ...` 与 `name (type): ...`，保留 `front row` 等多词枚举。
- `todo_manage.operations` 根据操作声明必填字段：创建需要 `text + ttl_s`，延期需要 `id + extra_s`，完成/取消需要 `id`，更新需要 `id` 且至少有 `text/ttl_s` 之一。
- 车辆模块工具在执行前校验参数；类型或枚举错误返回参数名与期望 schema，不执行方法。原有安全约束和业务校验继续生效。

## 多车并发

每个仿真边界仍然冻结物理世界。车辆回调在仿真线程上协作运行，仅 `AgentClient.chat/chat_with_tools` 的网络请求及重试等待进入有界线程池。一辆车等待模型时，其他车辆可继续各自的 PA、Judge、Driving Agent 决策链。同一辆车内部依赖顺序不变，也不共享提示词、Todo 或模型会话。

SUMO、工具执行和回调仍在原线程；Playwright 渲染还会回到原始主协程。各车使用独立命令缓存，全部回调结束后按场景车辆顺序合并，统一提交，再推进物理时间。某车回调失败只丢弃该车尚未提交的命令，保留其他车有效命令；随后仍按原策略将场景标记为基础设施失败并结束，不掩盖接口故障。

场景 JSON 可配置 `"max_parallel_model_calls": 4`，默认 4；设为 1 恢复串行。也可以使用 `MultiSimEngine(scenario, max_parallel_model_calls=...)` 覆盖。此上限按场景计算：外层同时运行 N 个场景时，最多会有 N × 4 个模型请求。行人回调仍沿用原有顺序执行。

这不是让仿真在模型未完成时继续前进；最慢车辆仍决定当前边界的结束时间。自定义模型客户端需使用 `simulation.model_concurrency.model_io` 包装纯模型 I/O 才会让出执行权，不应将整个车辆回调放入线程池。

PA 使用事件/随机唤醒；Judge 默认在请求后的 +0.1、+1、+3 s 检查，最多
3 次，提前完成则关闭验收。兼容的固定间隔模式需显式配置。详见
[PA/Judge 调度](pa-judge-scheduling.md)。
