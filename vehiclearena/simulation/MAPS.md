# 地图离线包

[English map-install guide](../../docs/en/quickstart.md)

高精度地图属于大型运行数据，不进入 VehicleArena Git 仓库。仓库只保留地图
加载器和 `map_bundle_manifest.json` 校验清单。完整数据包包含 116 张基础路网及
对应的 116 张车道级地图。

归档名采用日历版本 `YYYY.MM.DD.PATCH`：日期表示地图包版本日期，`PATCH`
从 0 开始，同一天修订时递增。当前版本为 `2026.09.22.0`。

## 安装离线包

从 [VehicleArena 的 Hugging Face 数据集仓库](https://huggingface.co/datasets/OpenMOSS-Team/VehicleArena)
下载当前清单指定的 `vehiclearena-road-networks-2026.09.22.0.tar.gz` 后，在仓库根目录执行：

```bash
python scripts/manage_map_bundle.py install \
  --source /path/to/vehiclearena-road-networks-2026.09.22.0.tar.gz
python scripts/manage_map_bundle.py status
```

安装器会依次验证公开清单中的整包 SHA-256、包内清单以及每个地图文件的
SHA-256。任何一步不一致都会停止，不会安装部分损坏的数据。更新已有地图时
显式增加 `--replace`；该选项只替换目标目录内的 `*.json`，不会删除加载器代码。

`--source` 也接受 `file://` 或 HTTPS 地址，因此同一套命令可以用于 U 盘、共享盘、
内网制品库或对象存储。正式发布时只需分发归档文件，仓库中的校验清单保持不变。

## 制作新版本

地图维护者在本地地图齐全时执行：

```bash
python scripts/manage_map_bundle.py pack \
  --map-dir vehiclearena/simulation/road_networks \
  --output /path/to/vehiclearena-road-networks-2026.10.03.0.tar.gz \
  --manifest /path/to/vehiclearena-road-networks-2026.10.03.0.manifest.json
```

确认新离线包通过安装和全地图回归后，再把生成清单中的内容更新到
`vehiclearena/simulation/map_bundle_manifest.json`。归档本身不能加入 Git。

## 数据边界

- Git 跟踪：加载器、地图目录说明、公开清单、场景定义和地图生成代码。
- 离线包：`vehiclearena/simulation/road_networks/*.json`。
- 运行产物：截图、GIF、轨迹和评测输出，仍由 `.gitignore` 排除。
