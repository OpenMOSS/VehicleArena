# Trace 可视化导出

已有实验 trace 的离线可视化工具集中在本目录。以下命令从项目根目录运行，
将 `path/to/trace.json.gz` 替换为实际轨迹路径。

| 文件 | 用途 |
| --- | --- |
| `export_trace_replay.py` | 生成独立 HTML，包含双视角、环境、时间轴和扣分事件 |
| `replay_assets/views.js` | HTML 双视角绘制、信号灯提示和浏览器视频录制 |
| `replay_assets/views.css` | HTML 双视角布局和样式 |
| `export_trace_fpv.py` | 导出第一视角 MP4 或单帧 PNG |
| `export_trace_video.py` | 导出上帝视角 MP4 |
| `export_event_svg.py` | 导出红灯事件的 SVG 场景图 |

## HTML 生成流程

`trace + vehiclearena/simulation/road_networks/ 中的对应路网 → build_payload() → HTML_TEMPLATE + replay_assets → 单个 HTML`

```bash
.venv/bin/python scripts/visualization/export_trace_replay.py \
  path/to/trace.json.gz -o outputs/replay.html
```

HTML 导出只使用 Python 标准库，支持 `.json` 和 `.json.gz`。
地图、轨迹、脚本和样式均嵌入输出文件，浏览器打开时无需服务器。
第一视角是地图与轨迹重建画面，灯色按地图配时表还原。
完整功能和数据限制见 [渲染说明](../../docs/reference/rendering.md)。

## 视频与事件图

```bash
.venv/bin/python scripts/visualization/export_trace_fpv.py \
  path/to/trace.json.gz -o outputs/first_person.mp4
.venv/bin/python scripts/visualization/export_trace_fpv.py \
  path/to/trace.json.gz --snapshot 286.9 -o outputs/first_person.png
.venv/bin/python scripts/visualization/export_trace_video.py \
  path/to/trace.json.gz -o outputs/top_down.mp4
.venv/bin/python scripts/visualization/export_event_svg.py \
  path/to/trace.json.gz -o outputs/event_scenes
```

这三个导出脚本读取 gzip trace；视频脚本依赖 Pillow 和 imageio-ffmpeg。
每个脚本可通过 `--help` 查看可选参数。HTML 页面也支持浏览器录制第一视角视频。

## 验证

```bash
.venv/bin/python -m pytest vehiclearena/evaluation/test_trace_replay.py -q
```

以上脚本原先位于 `scripts/` 根目录，调用时请使用新的 `scripts/visualization/` 路径。
