# Screenshots, GIFs, and offline replay

[中文](../reference/rendering.md) · [English index](README.md)

An existing compressed trajectory can be exported as a standalone HTML replay:

```bash
python scripts/visualization/export_trace_replay.py path/to/trace.json.gz -o outputs/replay.html
```

The replay shows a shared timeline, event navigation, a world view, and a simplified reconstructed first-person view. That reconstructed view is **not** the original `CameraVisual` image saved during an experiment. Signals are reconstructed from map timing and simulation time; ambiguous or unavailable timing appears gray. The replay does not invent missing buildings, trees, weather, or lighting. A browser with MediaRecorder can record the first-person replay to a video file.

Two rendering backends serve different purposes. `vehiclearena` draws from lane-level maps and the recorded world state, allowing replay and custom display. `sumo` requests images from the live `sumo-gui` instance that is executing physics, which is useful for validating native roads, vehicles, signals, and collisions. During an evaluated run, the Driving Agent instead receives the synchronized Web3D cockpit image and, if equipped, a local geometry view. Neither the offline replay nor a global SUMO screenshot is a substitute for those observations.

```bash
python vehiclearena/evaluation/render_world_timeline.py \
  --renderer vehiclearena --network beijing_guomao \
  --vehicles vehiclearena/evaluation/example_render_vehicles.json \
  --output /tmp/world.gif --start 0 --end 10 \
  --frame-interval 0.2 --full-map
```

Switch `--renderer vehiclearena` to `--renderer sumo` for native `sumo-gui` output; a display server or Xvfb must be available. For a PNG, use `--at 12`; for a GIF, use `--start`, `--end`, and `--frame-interval`. `--center-node` and `--radius` render a local region, while `--full-map` renders the whole network. Native SUMO screenshot times align to the 0.1-second physics grid. To save a replayable world record, use the VehicleArena renderer with `--recording-json`, not a native SUMO image.
