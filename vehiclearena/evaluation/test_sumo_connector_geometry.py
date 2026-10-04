"""Authored connection geometry must survive netconvert border clipping."""

import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from simulation.sumo_map import (
    SumoMapConverter, _clip_lane_extensions, _parse_shape,
)


def _split_fixture(tmp_path):
    root = ET.fromstring('''<net>
      <edge id="in"><lane id="in_0" shape="-2,0 0,0" length="2"/></edge>
      <edge id="out"><lane id="out_0" shape="3,0 5,0" length="2"/></edge>
      <edge id=":j_0"><lane id=":j_0_0" shape="0,0 1,1" length="1.414"/></edge>
      <edge id=":j_1"><lane id=":j_1_0" shape="1,1 2,1" length="1"/></edge>
      <connection from="in" to="out" fromLane="0" toLane="0" via=":j_0_0" tl="j" linkIndex="0"/>
      <connection from=":j_0" to="out" fromLane="0" toLane="0" via=":j_1_0"/>
      <connection from=":j_1" to="out" fromLane="0" toLane="0"/>
      <junction id="j" type="traffic_light"><request index="0" response="1" foes="1" cont="1"/></junction>
      <junction id=":j_1_0" type="internal" x="1" y="1" incLanes="in_0" intLanes=":foe_0"/>
      <tlLogic id="j"><phase duration="10" state="g"/></tlLogic>
    </net>''')
    path = tmp_path / "net.xml"
    ET.ElementTree(root).write(path)
    raw = {
        "lanes": [
            {"id": "a", "centerline_xy": [[-2, 0], [0, 0]]},
            {"id": "b", "centerline_xy": [[3, 0], [5, 0]]},
        ],
        "connectors": [{"id": "turn", "from_lane": "a", "to_lane": "b",
                        "centerline_xy": [[0, 0], [1, 1], [2, 1], [3, 0]]}],
    }
    manifest = {"connector_via_lane": {"turn": ":j_0_0"}}
    return path, root, raw, manifest


def test_restoration_preserves_yield_split_signals_and_conflict_graph(tmp_path):
    path, before, raw, manifest = _split_fixture(tmp_path)
    SumoMapConverter._restore_connector_geometry(raw, path, manifest)
    after = ET.parse(path).getroot()
    for tag in ("connection", "tlLogic"):
        assert [(x.attrib, [child.attrib for child in x]) for x in before.findall(tag)] == [
            (x.attrib, [child.attrib for child in x]) for x in after.findall(tag)]
    def priorities(root):
        return [(x.get("id"), x.get("incLanes"), x.get("intLanes"),
                 [y.attrib for y in x.findall("request")])
                for x in root.findall("junction")]
    assert priorities(before) == priorities(after)
    lanes = {x.get("id"): x for x in after.findall(".//lane")}
    assert _parse_shape(lanes[":j_1_0"].get("shape")) == [[1, 1], [2, 1], [3, 0]]
    assert manifest["connector_geometry"]["turn"]["lanes"] == [":j_0_0", ":j_1_0"]
    SumoMapConverter._validate_connector_geometry(after, manifest)


def test_restoration_keeps_bent_approach_trim_instead_of_straight_bridge(tmp_path):
    path, _, raw, manifest = _split_fixture(tmp_path)
    raw["lanes"][0]["centerline_xy"] = [[-2, 0], [0, 0], [0.5, -1], [1, 0]]
    raw["connectors"][0]["centerline_xy"][0] = [1, 0]
    SumoMapConverter._restore_connector_geometry(raw, path, manifest)
    root = ET.parse(path).getroot()
    points = _parse_shape(root.find("edge[@id=':j_0']/lane").get("shape"))
    assert [0.5, -1] in points
    SumoMapConverter._validate_connector_geometry(root, manifest)


def test_clipped_lane_uses_shared_vehicle_length_without_changing_sidewalk(tmp_path):
    path, root, raw, manifest = _split_fixture(tmp_path)
    incoming = root.find("edge[@id='in']")
    incoming.find("lane").set("id", "in_1")
    incoming.find("lane").set("shape", "-3,0 0.3,0")
    incoming.find("lane").set("length", "2")
    ET.SubElement(incoming, "lane", {
        "id": "in_2", "shape": "-2,3 0,3", "length": "2"})
    ET.SubElement(incoming, "lane", {
        "id": "in_0", "shape": "-2,6 0,6", "length": "2",
        "allow": "pedestrian"})
    root.find("connection[@from='in']").set("fromLane", "1")
    raw["lanes"][0]["centerline_xy"] = [[-3, 0], [0, 0]]
    manifest["lane_by_edge_index"] = {"in": {1: "a", 2: "b"}}
    ET.ElementTree(root).write(path)

    SumoMapConverter._restore_connector_geometry(raw, path, manifest)
    result = ET.parse(path).getroot()
    lanes = {lane.get("id"): lane for lane in result.findall(".//lane")}
    assert _parse_shape(lanes["in_1"].get("shape")) == [[-3, 0], [0, 0]]
    assert {lanes[lane_id].get("length") for lane_id in ("in_1", "in_2")} == {"2.500"}
    assert lanes["in_0"].get("length") == "2"
    SumoMapConverter._validate_connector_geometry(result, manifest)


@pytest.mark.parametrize("entry,exit", [(True, False), (False, True), (True, True)])
def test_extended_lane_ends_do_not_add_connector_backtracking(tmp_path, entry, exit):
    path, root, raw, manifest = _split_fixture(tmp_path)
    if entry:
        root.find("edge[@id='in']/lane").set("shape", "-2,0 0.3,0")
    if exit:
        root.find("edge[@id='out']/lane").set("shape", "2.7,0 5,0")
    ET.ElementTree(root).write(path)
    SumoMapConverter._restore_connector_geometry(raw, path, manifest)
    result = ET.parse(path).getroot()
    lanes = {x.get("id"): x for x in result.findall(".//lane")}
    assert _parse_shape(lanes['in_0'].get('shape'))[-1] == [0, 0]
    assert _parse_shape(lanes['out_0'].get('shape'))[0] == [3, 0]
    assert _parse_shape(lanes[':j_0_0'].get('shape')) == [[0, 0], [1, 1]]
    assert _parse_shape(lanes[':j_1_0'].get('shape')) == [[1, 1], [2, 1], [3, 0]]
    SumoMapConverter._validate_connector_geometry(result, manifest)


def test_closed_uturn_is_preserved_when_both_lane_ends_are_extended(tmp_path):
    path, root, raw, manifest = _split_fixture(tmp_path)
    raw['lanes'][1]['centerline_xy'] = [[0, 0], [-2, 0]]
    raw['connectors'][0]['centerline_xy'] = [[0, 0], [1, 1], [2, 0], [1, -1], [0, 0]]
    root.find("edge[@id='in']/lane").set('shape', '-2,0 0.3,0')
    root.find("edge[@id='out']/lane").set('shape', '0.3,0 -2,0')
    ET.ElementTree(root).write(path)
    SumoMapConverter._restore_connector_geometry(raw, path, manifest)
    result = ET.parse(path).getroot()
    points = []
    for lane_id in [':j_0_0', ':j_1_0']:
        points.extend(_parse_shape(result.find(f'.//lane[@id="{lane_id}"]').get('shape')))
    assert points[0] == points[-1] == [0, 0]
    assert all(p in points for p in [[1, 1], [2, 0], [1, -1]])
    SumoMapConverter._validate_connector_geometry(result, manifest)


def test_short_lane_shifted_outside_its_source_restores_the_original_bend():
    authored = [[0, 0], [0.5, 0.5], [1, 0], [1, 0]]
    assert _clip_lane_extensions(authored, [[-1, 0], [-0.8, 0]]) == authored[:-1]


@pytest.mark.parametrize('network,connector_id', [
    ('tokyo_shinjuku', 'connector::intersection::n1504865089::16'),
    ('hongkong_central', 'connector::n1615604003::2'),
])
@pytest.mark.skipif(shutil.which('netconvert') is None, reason='SUMO unavailable')
def test_recorded_lateral_jump_sites_have_continuous_native_motion(tmp_path, network, connector_id):
    import json
    import libsumo as sim
    source = Path(__file__).resolve().parents[1] / 'simulation/road_networks' / f'{network}_lane_level.json'
    raw = json.loads(source.read_text())
    bundle = SumoMapConverter().convert(source, cache_root=tmp_path)
    root = ET.parse(bundle.net_file).getroot()
    manifest = json.loads(Path(bundle.manifest_file).read_text())
    source_id, chain, target_id = SumoMapConverter._connector_chains(root, manifest)[connector_id]
    before = _parse_shape(root.find(f'.//lane[@id="{source_id}"]').get('shape'))
    turn = _parse_shape(root.find(f'.//lane[@id="{chain[0]}"]').get('shape'))
    authored = next(c for c in raw['connectors'] if c['id'] == connector_id)
    assert before[-1] == turn[0] == authored['centerline_xy'][0]
    incoming = [before[-1][i] - before[-2][i] for i in (0, 1)]
    outgoing = [turn[1][i] - turn[0][i] for i in (0, 1)]
    assert sum(a*b for a, b in zip(incoming, outgoing)) > 0

    # A 1.75 m lateral offset made the old backwards bridge flip the vehicle
    # by roughly 3.5 m. Drive slowly enough to sample even the 14 cm HK bridge.
    edge, lane_index = source_id.rsplit("_", 1)
    target_edge = target_id.rsplit("_", 1)[0]
    sim.start([
        "sumo", "-n", bundle.net_file, "--step-length", "0.1",
        "--lateral-resolution", "0.2", "--no-step-log", "true",
        "--no-warnings", "true",
    ])
    try:
        for tls in sim.trafficlight.getIDList():
            state = sim.trafficlight.getRedYellowGreenState(tls)
            sim.trafficlight.setRedYellowGreenState(tls, "G" * len(state))
        sim.route.add("probe-route", [edge, target_edge])
        sim.vehicle.add(
            "probe", "probe-route", departLane=lane_index,
            departPos=str(sim.lane.getLength(source_id) - 1), departSpeed="0.6")
        sim.simulationStep()
        sim.vehicle.setSpeedMode("probe", 0)
        sim.vehicle.setLaneChangeMode("probe", 0)
        sim.vehicle.setSpeed("probe", 0.6)
        sim.vehicle.setLateralLanePosition("probe", 1.75)
        previous = None
        visited = set()
        for _ in range(60):
            sim.simulationStep()
            position = sim.vehicle.getPosition("probe")
            visited.add(sim.vehicle.getLaneID("probe"))
            assert sim.vehicle.getLateralLanePosition("probe") == pytest.approx(1.75)
            if previous is not None:
                # Allow the lateral point to rotate around ordinary curve
                # vertices, in addition to the 6 cm longitudinal movement.
                assert math.dist(previous, position) <= 0.06 + 0.15
            previous = position
        assert set(chain) & visited
    finally:
        sim.close()


def test_clipped_yield_point_keeps_nonzero_ordered_stages(tmp_path):
    path, root, raw, manifest = _split_fixture(tmp_path)
    root.find("edge[@id=':j_0']/lane").set("shape", "0,0 9,0")
    root.find("edge[@id=':j_1']/lane").set("shape", "9,0 10,0")
    ET.ElementTree(root).write(path)
    SumoMapConverter._restore_connector_geometry(raw, path, manifest)
    assert manifest["connector_geometry"]["turn"]["split_method"] == "relative_length"
    result = ET.parse(path).getroot()
    assert float(result.find("edge[@id=':j_0']/lane").get("length")) > 0
    assert float(result.find("edge[@id=':j_1']/lane").get("length")) > 0
    assert result.find("junction[@type='internal']") is not None
    SumoMapConverter._validate_connector_geometry(result, manifest)


@pytest.mark.parametrize("damage", ["gap", "length", "missing_link"])
def test_geometry_validation_rejects_damaged_cache(tmp_path, damage):
    path, _, raw, manifest = _split_fixture(tmp_path)
    SumoMapConverter._restore_connector_geometry(raw, path, manifest)
    root = ET.parse(path).getroot()
    lane = root.find("edge[@id=':j_1']/lane")
    if damage == "gap":
        lane.set("shape", "1,1 2,1")
    elif damage == "length":
        lane.set("length", "0.2")
    else:
        root.remove(root.find("connection[@from=':j_1']"))
    with pytest.raises(RuntimeError):
        SumoMapConverter._validate_connector_geometry(root, manifest)


@pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO unavailable")
def test_cached_network_is_checked_again_before_use(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "simulation/road_networks/tokyo_shinjuku_lane_level.json"
    converter = SumoMapConverter()
    bundle = converter.convert(source, cache_root=tmp_path)
    tree = ET.parse(bundle.net_file)
    lane_id = bundle.connector_via_lane["connector::intersection::n1582703025::3"]
    lane = next(x for x in tree.getroot().findall(".//lane") if x.get("id") == lane_id)
    lane.set("shape", "-615.97,281.96 -619.52,284.51")
    tree.write(bundle.net_file)
    with pytest.raises(RuntimeError, match="discontinuous connector"):
        converter.convert(source, cache_root=tmp_path)


@pytest.mark.skipif(shutil.which("netconvert") is None, reason="SUMO unavailable")
def test_cached_network_rejects_divergent_vehicle_lane_lengths(tmp_path):
    import json

    root = Path(__file__).resolve().parents[1]
    source = root / "simulation/road_networks/amsterdam_centrum_lane_level.json"
    converter = SumoMapConverter()
    bundle = converter.convert(source, cache_root=tmp_path)
    manifest = json.loads(Path(bundle.manifest_file).read_text())
    bounded = set(manifest["bounded_vehicle_lanes"])
    edge_id, indexes = next(
        (edge_id, indexes)
        for edge_id, indexes in manifest["lane_by_edge_index"].items()
        if len(indexes) > 1 and any(
            f"{edge_id}_{index}" in bounded for index in indexes))
    tree = ET.parse(bundle.net_file)
    lane = tree.getroot().find(
        f"edge[@id='{edge_id}']/lane[@id='{edge_id}_{next(iter(indexes))}']")
    lane.set("length", str(float(lane.get("length")) + 1))
    tree.write(bundle.net_file)

    with pytest.raises(RuntimeError, match="inconsistent vehicle lane lengths"):
        converter.convert(source, cache_root=tmp_path)


@pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO unavailable")
def test_tokyo_crosswalk_has_no_geometry_jump_or_collision():
    from evaluation.experiments.telemetry import ExperimentTrackedEngine
    from evaluation.experiments.scene_catalog import Topology, _crosswalk
    from evaluation.experiments.time_window_calibration import _configure_case_actors
    from simulation.multi_sim_engine import MultiScenario

    root = Path(__file__).resolve().parents[1]
    # Keep the historical connector regression without retaining the retired
    # same-endpoint trip in the runnable benchmark catalog.
    topology = Topology("tokyo_shinjuku", root / "simulation/road_networks")
    scenario = _crosswalk(
        topology, "regression", "tokyo_connector_geometry",
        title="Tokyo connector geometry regression").scenario
    _configure_case_actors(scenario, {})
    scenario.update(total_time_s=60, physics_only_mode=True,
                    enable_driving_evaluation=False, stop_when_all_vehicles_terminal=True)
    engine = ExperimentTrackedEngine(MultiScenario.from_dict(scenario))
    result = engine.run({})
    assert result.vehicle_results["ego"].arrived
    assert not engine.traffic_mgr.collision_log
    frames = engine._vehicle_trajectory
    for a, b in zip(frames, frames[1:]):
        dt = b["time_s"] - a["time_s"]
        delta = math.hypot(b["pose_x_m"] - a["pose_x_m"], b["pose_y_m"] - a["pose_y_m"])
        assert delta <= max(a["speed_kmh"], b["speed_kmh"]) / 3.6 * dt + 0.15
