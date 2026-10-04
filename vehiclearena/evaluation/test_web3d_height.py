"""Topology layers must not turn an XY-only map into buried/floating roads.

Instrument the browser module only in tests; production observations gain no
hidden world state. Synthetic layers test camera geometry, not real altitude.
"""
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from visualization.web3d_camera import Web3DCameraRenderer
from visualization.web3d_scene import build_web3d_scene


@pytest.fixture
def height_page():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Playwright Chromium is not installed")
    renderer = Web3DCameraRenderer()
    try:
        renderer._start()
        page = renderer._browser.new_page(viewport={"width": 1024, "height": 960})
        def instrument(route):
            response = route.fetch()
            route.fulfill(response=response, body=response.text() + """
              window.heightProbe = ({yaw = 0, height = null} = {}) => {
                if (height !== null) egoActor.object.position.y = height;
                placeCockpitCamera(yaw);
                return {eye: camera.position.toArray(),
                  direction: camera.getWorldDirection(new THREE.Vector3()).toArray(),
                  actor: egoActor.object.position.toArray(),
                  forward: egoActor.direction.toArray()};
              };
            """)
        page.route("**/app.js", instrument)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"http://127.0.0.1:{renderer._server.server_address[1]}/?view=cockpit&capture=1",
                  wait_until="networkidle")
        page.wait_for_function("typeof window.heightProbe === 'function'")
        yield page
        assert not errors
    finally:
        renderer.close()


def layer_scene():
    return {
        "map": {"id": "height-test", "center_world_xy": [0, 0], "junction_id": "j"},
        "static": {
            "roads": [{"polygon_xz": [[-6, -120], [6, -120], [6, 30], [-6, 30]],
                       "elevation_m": 0}],
            "junction_surfaces": [], "markings": [], "ground_arrows": [],
            "stop_lines": [], "crosswalks": [], "buildings": [], "signals": [],
        },
        "snapshot": {"actors": [], "sim_time_s": 0},
    }


def layer_frame(level, revision=0):
    return {
        "schema": "vehiclearena-web3d-frame-v0.1", "sequence": revision,
        "sim_time_s": revision, "focus_entity_id": "focus", "stream_status": "running",
        "scene": {"map_id": "height-test", "radius_m": 100, "revision": revision,
                  "center_world_xy": [0, 0]},
        "environment": {"weather": "sunny", "daylight_level": 100}, "signals": {},
        "actors": [{"id": "focus", "kind": "vehicle", "control": "llm", "is_focus": True,
                    "position_world_xy": [0, 0], "yaw_rad": np.pi / 2,
                    "dimensions_m": [1.9, 1.55, 4.6], "z_level": level,
                    "speed_mps": 0, "signals": {}, "crashed": False}],
    }


@pytest.mark.parametrize("level", [-2, -1, 0, 1, 2])
def test_topology_layer_cannot_bury_actor_or_camera(height_page, level):
    page = height_page
    page.route("**/api/scene?*", lambda route: route.fulfill(json=layer_scene()))
    data = layer_frame(level)
    page.evaluate("frame => window.vehicleArenaCaptureFrame(frame)", data)
    for yaw in [0, -0.8, 0.8]:
        probe = page.evaluate("args => window.heightProbe(args)", {"yaw": yaw})
        assert probe["actor"][1] == 0
        assert probe["eye"][1] - probe["actor"][1] == pytest.approx(1.62)
        # Front and side glances keep identical physical eye positions and pitch.
        assert np.array(probe["eye"])[[0, 2]] == pytest.approx(
            np.array(probe["actor"])[[0, 2]] + .72 * np.array(probe["forward"])[[0, 2]])
        assert probe["direction"][1] == pytest.approx(-.28 / np.hypot(42, .28))
    page.evaluate("frame => window.vehicleArenaCaptureFrame(frame)", data)
    cockpit = np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB"), dtype=int)
    # The blue hood must remain visible in the lower front view on every layer.
    lower = cockpit[700:950]
    assert ((lower[:, :, 2] > lower[:, :, 0] + 35) &
            (lower[:, :, 2] > lower[:, :, 1] + 10)).sum() > 1000
    bev = page.evaluate("request => window.vehicleArenaCaptureLidarBEV(request)", {
        "sim_time_s": 0, "focus_entity_id": "focus", "sensor": {}})
    assert bev["ego_ndc"][:2] == pytest.approx([0, -.5])
    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(
        bev["png_data_url"].split(",", 1)[1]))).convert("RGB"), dtype=int)
    # Terrain must not cover the ego in the oblique view either.
    assert ((pixels[:, :, 2] > pixels[:, :, 0] + 35)).sum() > 1000
    # If actual elevations are supported later, eye placement must remain
    # body-relative. This is deliberately not a claim of 3D map support.
    for height in [-10, -5, 5, 10]:
        for yaw in [0, -.8, .8]:
            probe = page.evaluate("args => window.heightProbe(args)",
                                  {"height": height, "yaw": yaw})
            assert probe["eye"][1] == pytest.approx(height + 1.62)
            assert probe["direction"][1] == pytest.approx(-.28 / np.hypot(42, .28))


def test_layer_changes_do_not_cause_visual_height_jumps(height_page):
    page = height_page
    page.route("**/api/scene?*", lambda route: route.fulfill(json=layer_scene()))
    images = []
    for revision, level in enumerate([0, -1, 1, 0]):
        page.evaluate("f => window.vehicleArenaCaptureFrame(f)", layer_frame(level, revision))
        assert page.evaluate("() => window.heightProbe().actor[1]") == 0
        images.append(page.screenshot())
    assert all(image == images[0] for image in images)


def test_topology_metadata_cannot_raise_arrows_or_signal_masts(height_page):
    page = height_page
    scene = build_web3d_scene("beijing_tiananmen", include_demo_actors=False)
    assert scene["static"]["ground_arrows"] and scene["static"]["signals"]
    page.route("**/api/scene?*", lambda route: route.fulfill(json=scene))
    images = []
    for revision, level in enumerate([0, -1, 1]):
        for record in scene["static"]["ground_arrows"] + scene["static"]["signals"]:
            record["z_level"] = level
        data = layer_frame(level, revision)
        data["actors"][0]["position_world_xy"] = scene["map"]["center_world_xy"]
        page.evaluate("f => window.vehicleArenaCaptureFrame(f)", data)
        images.append(page.screenshot())
    assert all(image == images[0] for image in images)


def test_tokyo_recorded_pose_shows_road_instead_of_grass(height_page):
    page = height_page
    data = layer_frame(-1)
    data["scene"].update(map_id="tokyo_shinjuku", radius_m=220, center_world_xy=[330, -165])
    data["actors"][0].update(position_world_xy=[387.0415, -140.6118], yaw_rad=.975481)
    # Real exporter and real browser, at Basic016's recorded 27 s body centre.
    page.evaluate("f => window.vehicleArenaCaptureFrame(f)", data)
    pixels = np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB"), dtype=int)
    road = pixels[690:790, 440:590]
    assert np.abs(road[:, :, 1] - road[:, :, 0]).mean() < 12
    assert road.mean() > 25
    hood = pixels[800:850]
    assert ((hood[:, :, 2] > hood[:, :, 0] + 35) &
            (hood[:, :, 2] > hood[:, :, 1] + 10)).sum() > 500


def test_tokyo_inferred_lower_road_keeps_paint_on_road():
    scene = build_web3d_scene("tokyo_shinjuku", center_world_xy=[330, -165], radius_m=220)
    road = next(r for r in scene["static"]["roads"]
                if r["id"] == "n2857742920_n7682112321::lane_1")
    assert road["elevation_m"] == 0
    assert scene["map"]["elevation_mode"] == "flat_2d"
    for kind in ["roads", "junction_surfaces", "markings", "ground_arrows", "stop_lines", "signals"]:
        assert all(item["elevation_m"] == 0 for item in scene["static"][kind])
    roads = {r["id"]: r for r in scene["static"]["roads"]}
    for guide in scene["static"]["markings"]:
        if guide.get("connector_id"):
            assert guide["elevation_m"] == roads[guide["connector_id"]]["elevation_m"]
    for stop in scene["static"]["stop_lines"]:
        if stop["lane_id"] in roads:
            assert stop["elevation_m"] == roads[stop["lane_id"]]["elevation_m"]
