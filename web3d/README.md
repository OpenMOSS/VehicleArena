# VehicleArena Web3D

[English version](../docs/en/web3d.md)

该页面是 SUMO 物理世界的只读三维显示器。道路、车道、连接器、停止线、斑马线和信号灯位置来自车道级地图；车辆、行人、灯态、天气和仿真时间来自每次 SUMO 状态回写。简化建筑仅用于提供空间参照。

信号灯按“路口 + 进口道”合并：同一进口道只有一根立杆，立杆位于行驶方向右侧路面之外约 `0.9 m`，横臂承载每条受控车道的独立灯头。主灯不使用固定偏移量，而是根据 SUMO 路口连接器几何自动布置在路口对面的出口边缘；因此车辆停在停止线附近时仍能从前风挡看见信号灯。灯头继续根据各自 SUMO 连接器独立显示灯色。

停止线前约 `8 m` 绘制车道级地面箭头，允许多个方向时使用共用主杆的组合箭头。路口内部延续进口道分隔线，并按“每个入口、每种转向一条”绘制真实 SUMO 左右转连接引导线；掉头复用左转引导线，避免把所有车道连接画成线网。

实时帧中的天气直接驱动三维显示：雨天绘制随风倾斜的蓝色雨丝并降低路面粗糙度，雪天绘制圆形雪花并改变路面与路肩积雪色调，雾、昼夜和云量共同改变能见度、天空、太阳光强和曝光。`rainy/heavy_rain`、`snowy/heavy_snow` 使用不同粒子密度，`hail` 同时呈现雨雪粒子。车内视角保留完整天气遮挡；跟车与地图诊断视角降低近镜头粒子和雾强度，避免观察工具自身遮掉场景。粒子位置由主体编号和仿真时间确定，同一冻结时刻可以复现相同画面。

## 启动

先安装仓库要求的 Python 与离线地图资产，然后执行：

```bash
cd web3d
npm ci
python3 server.py
```

浏览器打开 <http://127.0.0.1:8765> 可检查静态地图。页面提供车内、跟车和地图三种视角，也可以用 `npm run serve` 启动。远程服务器可使用 SSH 端口转发：

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

URL 可以选择地图、路口和导出半径：

```text
http://127.0.0.1:8765/?map_id=beijing_tiananmen&junction_id=n31194143&radius_m=145&view=cockpit
```

## 实时运行 SUMO 场景

先打开实时页面：

```text
http://127.0.0.1:8765/?session_id=live&view=cockpit
```

再在仓库根目录运行一个不调用模型的 SUMO 参考场景：

```bash
python3 web3d/run_live.py \
  --scenario vehiclearena/evaluation/experiments/scenarios/Basic/basic_017_unsignalized_intersection/scenario.json \
  --session-id live \
  --focus-entity ego \
  --duration-s 15
```

`run_live.py` 会把场景里的所有主体切换成 SUMO 控制，只用于物理和显示冒烟。默认按照真实时间播放；`--fast` 可以取消节流来检查传输吞吐。

要观看真实 LLM 实验，在原有 `scripts/run_experiments.py run` 命令后增加：

```bash
  --no-resume \
  --web3d-stream-url http://127.0.0.1:8765/api/live/frame \
  --web3d-session-id live \
  --web3d-focus-entity ego \
  --web3d-realtime
```

## 数据边界

- 静态道路几何来自 VehicleArena，不维护另一份网页地图。
- 当前路网只有二维坐标，显示采用 `flat_2d`：道路、车辆、标线及灯杆基座的 `elevation_m` 均为 0。`z_level` 是拓扑层级，不能乘以固定米数当成海拔；相机眼位始终相对车身计算。此显示投影不修改 SUMO 路由、层级或碰撞规则，也不代表已重建真实立交/隧道：不同层道路的平面投影仍可能重叠。支持真实高差需要完整的道路高程、坡道和地形数据，不能仅恢复 `z_level * 5`。
- `MultiSimEngine` 在 SUMO 状态同步之后调用只读 world observer。
- 发布器使用有界队列，只保留最新帧；网页或网络失败不会改变物理结果。
- 服务端通过 `/api/live/frame` 接收帧，通过 `/api/live/ws` 向浏览器推送。
- 浏览器仅在相邻 SUMO 快照之间做显示插值，不规划路线、不推进物理，也不反写状态。
- 长路线按量化的局部地图块切换；静态块由服务端缓存，车辆接近当前块边界时才加载下一块。
- 静态预览中的演示主体保持冻结，不再由网页自行生成运动。
- 目前车辆使用与 SUMO 长宽一致的程序化中精度模型，包含圆角车身、玻璃、车轮、后视镜及可变灯态。行人使用低面数静态人体模型，位置和朝向来自 SUMO；显示模型不改变 SUMO 的物理位置和碰撞域。建筑仍为简化体块，模型后续可替换为 glTF 资产而不改变物理接口。
