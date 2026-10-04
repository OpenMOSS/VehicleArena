# 截图与 GIF

[English version](../en/rendering.md)

## 离线 HTML 双视角回放

离线导出代码统一位于 [`scripts/visualization/`](../../scripts/visualization/README.md)，
包含 HTML、第一视角视频、上帝视角视频和事件 SVG 导出工具。

已有实验轨迹可以导出为单个 HTML 文件，直接在浏览器打开：

```bash
python scripts/visualization/export_trace_replay.py path/to/trace.json.gz -o outputs/replay.html
```

页面同时显示上帝视角和第一视角，共用播放、倍速、时间轴及事件跳转。
可选择观察车辆、跟随车辆、查看全图，或显示完整行驶轨迹。上帝视角使用
地图的道路边界、路口连接路面、停止线和斑马线；记录中的在场车辆和行人
随时间回放。点击信号灯可查看其车道、转向及对应连接器的灯色。

第一视角是根据轨迹和地图重建的简化透视画面，不是实验时保存的原始
CameraVisual 图像。红绿灯按地图配时表和仿真时刻还原，不代表另有逐帧
SUMO 信号日志；同一转向组的配时不一致或缺少配时则显示灰色，点击查看
各连接器状态。页面不生成地图中没有的建筑、树木，也不重建天气或光照。

第一视角中的场景灯组显示红、黄、绿三个灯位；左上方另有“当前车道信号”
回放提示，显示前方 120 米内本车车道对应的各转向灯色。停车时高处灯组可能
超出相机画面，这个提示仍保持可见；进入连接器后显示“已进入路口”。
提示也包含在导出的视频中。

支持 MediaRecorder 的浏览器中，点击“录制第一视角”会开始播放并录制，
再次点击“停止并下载”导出视频（通常为 WebM）。视频采用录制时选择的
播放速度；HTML 自身不依赖服务器、外部脚本或独立视频文件。

VehicleArena 提供两个用途不同的绘制后端：

| 后端 | 数据来源 | 用途 |
|---|---|---|
| `vehiclearena` | 高精地图和统一运行态 | 可重放世界记录、调试以及定制显示 |
| `sumo` | 当前正在执行物理的 `sumo-gui` 实例 | 核对 SUMO 原生路网、车辆、信号和碰撞 |

正式实验中的 `CameraVisual` 来自同步 SUMO 状态的 Web3D 车内视角，
可选 `LidarBEV` 从同一冻结状态生成局部几何图。两者都不是本页的离线回放画面，
也不能用 SUMO 全局截图替代，否则会泄露驾驶员视野外的信息。

## 通用命令

VehicleArena 绘制：

```bash
python3 vehiclearena/evaluation/render_world_timeline.py \
  --renderer vehiclearena --network beijing_guomao \
  --vehicles vehiclearena/evaluation/example_render_vehicles.json \
  --output /tmp/world.gif --start 0 --end 10 \
  --frame-interval 0.2 --full-map
```

同一入口切换到 SUMO 原生绘制：

```bash
sudo apt-get install sumo sumo-tools xvfb
python3 vehiclearena/evaluation/render_world_timeline.py \
  --renderer sumo --network beijing_guomao \
  --vehicles vehiclearena/evaluation/example_render_vehicles.json \
  --output /tmp/world-sumo.gif --start 0 --end 10 \
  --frame-interval 0.2 --full-map
```

原生绘制默认加载 `visualization/sumo_vehiclearena.view.xml`：使用与
VehicleArena 前视图接近的低对比道路配色、隐藏宏观连接器，并关闭不必要的
图例。可用 `--sumo-gui-settings` 替换设置文件，或用 `--sumo-schema` 选择
该文件中的其他方案。

输出为 `.png` 时用 `--at 12` 选择时刻；输出为 `.gif` 时使用
`--start`、`--end` 和 `--frame-interval`。`--center-node` 配合 `--radius`
可绘制局部区域，否则 `--full-map` 绘制全图。SUMO原生绘制的时刻必须
落在全局 `0.1s` 物理时间格上。

SUMO模式通过 TraCI 启动唯一的 `sumo-gui` 物理实例，并在公开仿真步边界
请求截图，不会另跑一份用于显示的仿真。服务器没有 `DISPLAY` 时默认自动
启动 Xvfb；`--no-auto-xvfb` 可要求调用方提供显示环境。SUMO在第一次
`simulationStep` 后才生成可绘制车辆，因此从 `0s` 请求的原生序列从
`0.1s` 开始，命令结果会返回实际起止时间。

SUMO原生图片不能保存为 VehicleArena 的重放 JSON；需要可重放记录时使用
`--renderer vehiclearena --recording-json ...`。
