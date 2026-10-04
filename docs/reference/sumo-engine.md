# SUMO 执行引擎

[English version](../en/sumo-engine.md)

SUMO 是 VehicleArena 唯一的车外物理与背景交通执行引擎。

## 权威边界

SUMO 负责：

- 所有车辆和行人的位置、速度、加速度与碰撞；
- 背景车辆的跟驰、换道、限速、信号和路权；
- 背景行人的步行、过街和信号等待；
- LLM 控制命令的连续动力学执行。

VehicleArena 负责：

- 将高精度车道地图编译为 SUMO net；
- 为 LLM 构造视觉、雷达、导航和车辆工具；
- 把 LLM 的目标速度、制动、变道和逐路口方向命令提交给 SUMO；
- 维护车内模块、事件、轨迹日志和评测。

LLM车辆关闭SUMO自主换道与路线决策，但仍受车辆尺寸、加减速能力和共享碰撞世界约束。导航小地图的完整推荐路线不会作为物理路线安装；每次`navigation_select_maneuver`只把当前边和一条已选连接线后的目标边交给SUMO。该选择持续到车辆完整通过当前路口；只有车道即将结束且尚未选择下一方向时，物理停车点才会限制车辆继续前进。当前车道终止于目的地时，系统拒绝追加连接线并允许SUMO在该边终点完成行程。背景实体不接受VehicleArena行为回调；车辆速度用`setSpeed(-1)`交还SUMO，行人速度同样由SUMO决定。

## 背景车初态

普通 SUMO 背景车的 `initial_physical_state.speed_kmh` 只指定初始速度，
随后由 SUMO 跟驰、限速和换道；正式场景不再保留这些车不使用的
`target_speed_kmh`、`desired_speed_kmh` 初态字段，目录校验会拒绝它们。
LLM 车辆的持续目标速度不受此次清理影响。历史实体 ID `slow_lead`
只为轨迹兼容保留，不表示持续慢行；相关场景允许按实际交通保持车道，
不保证一定发生超车。没有新增 NPC 锁速或脚本换道。

已有专门制动实验的 `experiment_world_events` 属于显式实验干预，
与普通 NPC 自主策略分开；本次不增删这些事件。

## 碰撞后运动与记录

SUMO 报告真实碰撞后，该车驾驶任务立即进入碰撞终态，不再接受驾驶决策。
系统向 SUMO 提交零目标速度，但不把当前速度或加速度字段直接清零。
碰撞帧和之后的轨迹继续使用 SUMO 实测值：如仍在移动，应记录真实减速，
直到物理上停稳；碰撞标记不等于已经静止。车车和车人碰撞中的车辆使用同一
规则，背景车与 LLM 车不区别处理。没有增加碰撞前自动避险，也不修改旧轨迹。

## 地图编译

道路车道、内部连接器、停止线、斑马线和交通灯链接被转换为 SUMO 原生拓扑。VehicleArena 在每个 0.1 s 步开始前同步场景信号与道路约束，步结束后读取所有实体状态和碰撞。

## 安装检查

```bash
sumo --version
netconvert --version
python scripts/check_environment.py
python -m pytest -q vehiclearena/evaluation/test_sumo_engine.py
```

`sumo-gui` 只用于调试和原生截图。正式驾驶的 `CameraVisual` 由同步 SUMO 状态驱动的 Web3D 车内视角生成；可选 `LidarBEV` 复用同一浏览器场景生成 35° 斜俯视几何图，导航小地图仍使用规则渲染。旧 2D BEV 函数仅供离线兼容，正式引擎不再调用它。
