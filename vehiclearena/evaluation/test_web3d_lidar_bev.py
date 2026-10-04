"""Production browser BEV: frozen frames, ownership, appearance and isolation."""
import base64
import copy
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from simulation.road_networks import load_road_network
from simulation.traffic_manager import TrafficCoordinator
from visualization.agent_visual_renderer import RenderedAgentImage, multimodal_image_message
from visualization.web3d_camera import Web3DCameraRenderer


@pytest.fixture
def browser_renderer():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as pw:
        if not Path(pw.chromium.executable_path).exists():
            pytest.skip("Playwright Chromium is not installed")
    renderer = Web3DCameraRenderer()
    try:
        renderer._start()
        yield renderer
    finally:
        renderer.close()


def frame():
    return {
        "schema": "vehiclearena-web3d-frame-v0.1", "sequence": 0,
        "sim_time_s": 9.3, "focus_entity_id": "approach_4", "stream_status": "running",
        "scene": {"map_id": "hongkong_central", "radius_m": 100, "revision": 0,
                  "center_world_xy": [-457.5767, -283.6718]},
        "environment": {"weather": "sunny", "daylight_level": 100}, "signals": {},
        "actors": [
            {"id": vid, "kind": "vehicle", "control": "llm", "is_focus": focus,
             "position_world_xy": position, "yaw_rad": yaw, "dimensions_m": [1.9, 1.55, 4.6],
             "speed_mps": 0, "signals": {}, "crashed": False}
            for vid, focus, position, yaw in [
                ("approach_4", True, [-457.5767, -283.6718], -2.515675),
                ("peer", False, [-464.6725, -290.994], 0.42),
                # A literal ego ID must not steal another car's camera.
                ("ego", False, [-516.3828, -333.8294], 0.1),
            ]],
    }


def load(page, data):
    result = page.evaluate("frame => window.vehicleArenaCaptureFrame(frame)", data)
    assert result["focus_entity_id"] == data["focus_entity_id"]
    return result


def capture_bev(page, data, sensor=None):
    result = page.evaluate("request => window.vehicleArenaCaptureLidarBEV(request)", {
        "sim_time_s": data["sim_time_s"], "focus_entity_id": data["focus_entity_id"],
        "sensor": sensor or {},
    })
    png = base64.b64decode(result.pop("png_data_url").split(",", 1)[1])
    return png, result


def test_production_bev_style_and_cockpit_restore(browser_renderer, tmp_path):
    page = browser_renderer._page("approach_4")
    errors = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    data = frame()
    load(page, data)
    optical_before = page.screenshot(animations="disabled")
    bev, meta = capture_bev(page, data)
    assert meta["view"] == "web3d_ego_oblique"
    assert meta["focus_entity_id"] == "approach_4"
    assert meta["ego_ndc"][:2] == pytest.approx([0, -0.5])
    assert meta["elevation_deg"] == 35 and meta["fov_deg"] == 40
    pixels = np.asarray(Image.open(io.BytesIO(bev)).convert("RGB"), dtype=np.int16)
    assert pixels.shape == (1024, 1024, 3)
    colored = pixels.max(axis=2) - pixels.min(axis=2) > 0
    ys, xs = np.where(colored)
    assert len(xs) > 1000
    assert 300 < xs.min() < xs.max() < 720
    assert 400 < ys.min() < ys.max() < 1000
    assert np.all(pixels[colored, 2] > pixels[colored, 0])
    # The whole background vehicle, including windows and tires, is grey.
    assert np.all(pixels[:450].max(axis=2) == pixels[:450].min(axis=2))
    assert pixels[:450].max() > 160
    load(page, data)
    assert page.screenshot(animations="disabled") == optical_before
    assert page.viewport_size == {"width": 1024, "height": 960}
    assert not errors
    (tmp_path / "production-bev.png").write_bytes(bev)
    (tmp_path / "production-cockpit.png").write_bytes(optical_before)


def test_bev_ignores_optical_state_and_reuses_materials(browser_renderer):
    page = browser_renderer._page("approach_4")
    data = frame()
    load(page, data)
    original, before = capture_bev(page, data)
    changed = copy.deepcopy(data)
    changed.update(sim_time_s=17.0, sequence=1)
    changed["environment"] = {"weather": "heavy_rain", "daylight_level": 0, "is_night": True}
    for actor in changed["actors"]:
        actor["signals"] = {"high_beam": True, "brake_light": True, "left_indicator": True}
    load(page, changed)
    for _ in range(3):
        image, meta = capture_bev(page, changed)
        assert image == original
        assert meta["material_cache_size"] == before["material_cache_size"]


def test_bev_range_fov_and_frame_checks(browser_renderer):
    page = browser_renderer._page("approach_4")
    data = frame()
    load(page, data)
    original, _ = capture_bev(page, data)
    narrow, _ = capture_bev(page, data, {"range_m": 10, "horizontal_fov_deg": 30})
    assert original != narrow
    with pytest.raises(Exception, match="frame/focus mismatch"):
        capture_bev(page, {**data, "focus_entity_id": "ego"})
    with pytest.raises(Exception, match="frame/focus mismatch"):
        capture_bev(page, {**data, "sim_time_s": 1.0})
    with pytest.raises(Exception, match="Invalid LiDAR"):
        capture_bev(page, data, {"range_m": 0})


def test_bev_failure_restores_original_scene(browser_renderer):
    page = browser_renderer._page("approach_4")
    data = frame()
    load(page, data)
    before = page.screenshot(animations="disabled")
    page.evaluate("""async () => {
      const {LidarBEVRenderer} = await import('/lidar_bev.js');
      const render = LidarBEVRenderer.prototype.render;
      LidarBEVRenderer.prototype.render = function (...args) {
        const canvas = this.renderer.domElement;
        const encode = canvas.toDataURL;
        canvas.toDataURL = () => { throw new Error('test encode failure'); };
        try { return render.apply(this, args); }
        finally { canvas.toDataURL = encode; LidarBEVRenderer.prototype.render = render; }
      };
    }""")
    with pytest.raises(Exception, match="test encode failure"):
        capture_bev(page, data)
    load(page, data)
    assert page.screenshot(animations="disabled") == before
    assert capture_bev(page, data)[1]["view"] == "web3d_ego_oblique"


def test_python_pair_reuses_one_encoder_page_and_frozen_state(browser_renderer):
    manager = TrafficCoordinator(load_road_network("beijing_guomao"))
    manager.register_vehicle("focus", "n33399858", "n35722739", is_llm=True)
    engine = SimpleNamespace(traffic_mgr=manager, scenario=SimpleNamespace(
        road_network_id="beijing_guomao", physics_step_s=0.1))
    v = manager.vehicles["focus"]
    before = (v.pose_x_m, v.pose_y_m, v.yaw_rad)
    camera, bev = browser_renderer.render_observations(engine, "focus", 0.0, 0)
    assert (camera.width, camera.height) == (1024, 960)
    assert (bev.width, bev.height, bev.kind) == (1024, 1024, "lidar_bev")
    assert bev.view == "web3d_ego_oblique"
    assert camera.sim_time_s == bev.sim_time_s == 0
    assert browser_renderer._encoders["focus"].frame_count == 1
    assert len(browser_renderer._pages) == 1
    assert (v.pose_x_m, v.pose_y_m, v.yaw_rad) == before
    assert "[LidarBEV]" in multimodal_image_message(bev, label="LidarBEV")["content"][0]["text"]


def test_engine_routes_optional_bev_through_shared_owner_capture(monkeypatch):
    import threading
    from greenlet import getcurrent
    owner = (threading.get_ident(), getcurrent())
    rendered, received = [], []
    image = RenderedAgentImage("lidar_bev", 0, 1024, 1024, b"test", "web3d_ego_oblique")

    class Camera:
        def render(self, engine, vid, t, tick):
            rendered.append((vid, "camera"))
            return None

        def render_observations(self, engine, vid, t, tick, overrides):
            assert (threading.get_ident(), getcurrent()) == owner
            assert overrides == {"range_m": 30}
            rendered.append((vid, "pair"))
            return None, image

        def close(self):
            pass

    monkeypatch.setattr("visualization.web3d_camera.Web3DCameraRenderer", Camera)
    scenario = MultiScenario.from_dict({
        "scenario_id": "optional_bev_capture", "road_network_id": "beijing_guomao",
        "total_time_s": 0.1,
        "vehicles": [{"vehicle_id": vid, "initial_node": node,
                      "agent_config": {"type": "llm"}, "equipment_profile": "executive",
                      **({"enable_modules": ["lidar"], "sensor_overrides": {"lidar": {"range_m": 30}}}
                         if vid == "equipped" else {})}
                     for vid, node in [("equipped", "n33399858"), ("plain", "n35553582")]]})
    engine = MultiSimEngine(scenario)
    def callback(vw, t, *args, **kwargs):
        received.append((vw.has_module("lidar"), kwargs.get("_lidar_visual")))
        return []
    engine.run({"equipped": callback, "plain": callback})
    assert not engine.agent_callback_errors
    assert ("equipped", "pair") in rendered and ("plain", "camera") in rendered
    assert all((visual is image) if equipped else visual is None for equipped, visual in received)


def test_real_sumo_capture_reaches_model_messages(tmp_path):
    from evaluation.multi_agent_runner import make_llm_agent_callback

    class FinishClient:
        model = "mock-bev-finish"
        api_base = "local"
        temperature = 0.0
        max_tokens = 64

        def __init__(self):
            self.calls = []
            self.last_call_metadata = {}

        def chat_with_tools(self, messages, tools):
            self.calls.append(copy.deepcopy(messages))
            self.last_call_metadata = {"ok": True, "finish_reason": "tool_calls"}
            call = SimpleNamespace(id="finish-1", type="function", function=SimpleNamespace(
                name="finish", arguments='{"reason":"observation received"}'))
            return SimpleNamespace(role="assistant", content="", tool_calls=[call]), 2, 1, 1

    scenario = MultiScenario.from_dict({
        "scenario_id": "real_bev_message", "road_network_id": "beijing_guomao",
        "total_time_s": 0.0,
        "vehicles": [{"vehicle_id": "ego", "initial_node": "n33399858",
                      "destination_node": "n35722739", "agent_config": {"type": "llm"},
                      "equipment_profile": "executive", "enable_modules": ["lidar"]}]})
    client = FinishClient()
    callback = make_llm_agent_callback("ego", client, max_turns=1)
    callback._state["visual_observation_output_dir"] = str(tmp_path)
    engine = MultiSimEngine(scenario)
    engine.run({"ego": callback})
    assert not engine.agent_callback_errors
    assert len(client.calls) == 1
    blocks = [block for message in client.calls[0] if isinstance(message.get("content"), list)
              for block in message["content"]]
    labels = [block["text"] for block in blocks if block["type"] == "text"]
    assert any("[CameraVisual]" in label for label in labels)
    assert any("[LidarBEV]" in label for label in labels)
    bev_record = next(record for record in callback._state["visual_observation_log"]
                      if record["label"] == "LidarBEV")
    assert bev_record["view"] == "web3d_ego_oblique"
    assert len([block for block in blocks if block["type"] == "image_url"]) == 2
    assert len(list(tmp_path.rglob("*.png"))) == 2


@pytest.mark.parametrize("asset", ["lidar_bev.js", "index.html", "styles.css", "server.py"])
def test_browser_assets_participate_in_experiment_fingerprint(tmp_path, asset):
    from evaluation.experiments.manifest import source_fingerprint
    source = tmp_path / "vehiclearena"
    source.mkdir()
    (source / "engine.py").write_text("version = 1\n")
    web = tmp_path / "web3d"
    web.mkdir()
    (web / asset).write_text("first")
    original = source_fingerprint(source)
    (web / asset).write_text("second")
    changed = source_fingerprint(source)
    assert changed != original
    (web / "node_modules").mkdir()
    (web / "node_modules" / "dependency.js").write_text("ignored")
    (web / "outputs").mkdir()
    (web / "outputs" / "preview.js").write_text("ignored")
    assert source_fingerprint(source) == changed
