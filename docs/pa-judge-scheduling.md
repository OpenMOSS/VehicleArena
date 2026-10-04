# PA 事件/随机触发与有限窗口 Judge

启用 PA 后，默认由事件和独立随机计时触发。PA 代表乘客提出请求，不控制车辆或改变场景目的地。Judge 根据执行后的证据验收请求，不替 DA 决策。批量运行与结果查看见[LLM 测试指南](llm-suite.md)。

## 触发

```text
共享事件 + 运动状态边沿 + 独立随机计时
    → personal_agent_due → PA
    → personal_agent_update → Driver（同一边界合并）
    → 若有新请求，安排独立 Judge 检查
```

| 参数/事件 | 默认行为 |
|---|---|
| 共享订阅 | simulation_start、acoustic_cue、weather_changed、daynight_changed |
| 急刹 | 加速度 ≤ -3 m/s² 持续 0.3 s；恢复到 > -1.5 后允许重新检测 |
| 长停 | 速度 < 0.5 km/h 持续 15 s；恢复到 > 2 后重新计时 |
| 随机唤醒 | 每次从 U(15,45) 仿真秒采样，不由驾驶心跳重置 |
| 种子 | SHA256(seed, scenario_id, vehicle_id)，各车独立，可复现 |
| PA 冷却 | 5 仿真秒；冷却期间按事件种类合并，不重复调用 |

共享事件只传种类，事实来自当前乘客观察；不向 PA 复制内部事件详情或评测真值。
持续急刹/停车只报告一次。长停解除时，未投递的长停事件丢弃。
驾驶 heartbeat、普通工具回执、Judge 检查、未完成和超时不触发 PA。
原始雷达预警不在默认订阅中：未确认是乘客可感知的座舱告警前，不直接开放。

PA 可以结束本轮而不提要求。PA 处理完仍以 personal_agent_update 唤醒 Driver；
与同一时刻 heartbeat 合并，只调用 Driver 一次。Judge-only 则不调用 PA 或 Driver，
也不改变驾驶 heartbeat 计划。到达、碰撞失效和仿真结束后仅收尾。

## Judge 窗口

每条新请求记录 request_id、创建时间和固定验收截止时间 acceptance_deadline_s。
不同请求可以有重叠窗口，互不覆盖，不因驾驶唤醒续期。
默认在请求后 +0.1/+1/+3 仿真秒调用 LLM Judge，最多 3 次。
截止时间仅限制验收预算，不表示乘客请求在语义上失效。
可配置 check_offsets_s 和 acceptance_timeout_s；提前截止则在对应的首个
0.1 s 物理边界收尾。兼容配置 window_s/max_checks 支持等间隔检查；
request_ttl_s 是验收预算的旧别名，不能与新 timeout 同时设置。
不同请求分别调用 Judge。

中间检查通过 `submit_passenger_check` 记录每项条件的状态，不给 A–F 分。最终检查通过 `submit_passenger_judgement` 提交 A–F 或 NA、条件状态和证据。请求状态有三种：

- completed：证据支持完成，提前停止。
- pending：尚未满足，继续下个窗口。
- uncertain：证据不足，继续检查。

持续性请求 request_kind=ongoing 在最后窗口前不能完成；过早的 completed 会被改为 pending。
最终仅评价已观察窗口内是否满足，不声称未来或整段旅程永久完成。
Prompt 明确区分中间轮和最后一轮：最后一轮若证据已支持窗口内达标，应返回
completed，不能仅因“请求是持续性的、无法保证未来”返回 pending/uncertain。
存在未满足的行为则 pending，窗口内证据不足则 uncertain；仅提交控制或保持
停车并不自动证明驾驶平稳。代码不强制将持续请求改判成功。
物理请求不要求 DA 必须口头回复；Judge 根据请求条件和可观察到的执行结果判定，不能把仅排队的命令或口头承诺当成物理完成。
这些是基于证据的模型评判，不是通用形式化任务验证。

验收期限/次数耗尽：pending 记 uncompleted（期限内未完成）；uncertain 或没有有效结论记 unverified。
场景提前结束时立即检查剩余请求一次，标明 episode_ended 和实际窗口。
未完成/超时只记录，不发事件，不调用 PA/Driver，不自动续期。

证据包含原请求、请求后的累计驾驶回应/执行回执、运动轨迹、请求时与当前的世界快照，
以及上一次检查结论。旧结论不能代替当前证据。

## 上下文与审计

PA 看到当前乘客可感知状态、最近 10 秒运动、触发种类，以及最近最多 6 条请求和
乘客可见的驾驶回应。不会看到 Judge 分数、完成结论、评测真值和驾驶员私有推理。
系统不对自然语言请求做语义去重，也不自动催办。

结果保存：

- personal_agent.config / trigger_log：参数与实际触发时间、原因。
- personal_agent.request_records：请求、验收截止时间、检查次数、终态，仅作审计。
- passenger_judge.check_history：全部检查记录。
- passenger_judge.judgements：每个请求只有一条最终记录，避免重复计分。
- 原有模型调用、证据、回应与事件投递日志。

## 配置

单场景命令见[快速开始](01-quickstart.md)。如需显式设置触发与验收窗口，在 `scripts/run_experiments.py run` 命令中加入：

```text
--with-personal-agent
--personal-agent-trigger-mode event_random
--personal-agent-seed 0
--personal-agent-random-min-s 15
--personal-agent-random-max-s 45
--personal-agent-cooldown-s 5
--personal-agent-stopped-after-s 15
--passenger-judge-check-offsets-s 0.1 1 3
--passenger-judge-max-checks 3
--passenger-judge-acceptance-timeout-s 3
```

Python 的 personal_agent_runtime_config 还可设置 event_types、hard_brake_mps2、
hard_brake_duration_s。模型/提供方参数继续使用原有配置方式。
`legacy` 模式保留单窗口判分，用于读取旧配置；新批次应使用当前事件/随机触发和多窗口参数。源码变更后需生成新 manifest，不能把已有轨迹重标成新模式。

## 测试

`vehiclearena/evaluation/test_passenger_orchestration.py` 使用确定性模型 stub，验证随机流复现与逐车隔离、
心跳独立、事件合并/重新进入、PA no-op 唤醒 Driver、Judge-only 隔离、多窗口提前结束、
TTL/次数限制、持续请求、多个在途请求终态收尾、观察隔离及真实 SUMO 调度。
离线测试不调用真实 API，不代表真实模型的完成判断已校准。
