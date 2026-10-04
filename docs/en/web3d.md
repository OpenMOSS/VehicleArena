# Web3D viewer

[中文](../../web3d/README.md) · [English index](README.md)

Web3D is a read-only 3D display of the SUMO world. Road geometry comes from VehicleArena lane-level maps; moving participants, signal colors, weather, and simulation time come from SUMO state snapshots. Browser animation interpolates between snapshots but never advances the physics or writes control back to the simulation. Procedural vehicle and pedestrian meshes are visual representations, not alternative collision bodies.

## Start the viewer

Install the Python dependencies and road bundle described in the [quickstart](quickstart.md), then install the Web3D packages:

```bash
cd web3d
npm ci
python3 server.py
```

Open `http://127.0.0.1:8765` for a static map preview. The interface offers cockpit, following, and map views. On a remote server, forward port 8765 over SSH. URL parameters can select a map, junction, export radius, and view, for example:

```text
http://127.0.0.1:8765/?map_id=beijing_tiananmen&junction_id=n31194143&radius_m=145&view=cockpit
```

## View a live SUMO smoke test

Open `http://127.0.0.1:8765/?session_id=live&view=cockpit`, then run from the repository root:

```bash
python web3d/run_live.py \
  --scenario vehiclearena/evaluation/experiments/scenarios/Basic/basic_017_unsignalized_intersection/scenario.json \
  --session-id live \
  --focus-entity ego \
  --duration-s 15
```

`run_live.py` temporarily makes every participant SUMO-controlled for this display/physics check. To watch a real LLM episode, add `--no-resume`, `--web3d-stream-url http://127.0.0.1:8765/api/live/frame`, `--web3d-session-id live`, `--web3d-focus-entity ego`, and `--web3d-realtime` to the normal `scripts/run_experiments.py run` command.

The current road maps contain 2D coordinates. `z_level` denotes topology, not measured road elevation, so this viewer uses a flat 2D projection for road surfaces. Its weather, lights, traffic signals, and visual meshes do not change SUMO collision or routing outcomes.
