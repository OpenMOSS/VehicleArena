"""Movement lamps and scoring must use the authoritative lane-level clock."""
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.driving_evaluator import DrivingEvaluator, DrivingEvaluationConfig
from evaluation.driving_process_score import _legitimate_stop, DrivingProcessScoreConfig
from simulation.lane_level_runtime import LaneGeometryRuntime
from simulation.road_networks import load_road_network
from simulation.traffic_manager import TrafficCoordinator
from simulation.vehicle_state import VehicleState
from visualization.web3d_live import Web3DFrameEncoder
from visualization.web3d_scene import build_web3d_scene
from visualization.web3d_camera import Web3DCameraRenderer
from visualization.signal_layout import signal_stop_lines
from simulation.lane_level_runtime import pose_at


INCOMING = 'n1286462578_n314160089::lane_3'
TERMINAL = 'n1667167408_n1667167480::lane_3'
STRAIGHT = 'connector::intersection::n1667167449::4'
LEFT = 'connector::intersection::n1667167449::5'


@pytest.fixture
def manager():
    net = load_road_network('hangzhou_binjiang')
    net.load_scenario_events({})  # The obsolete clock exists but must not be read.
    return TrafficCoordinator(net)


def vehicle_on_lane(manager, lane_id, progress=.97):
    lane = manager._lane_geometry._lane_by_id[lane_id]
    vehicle = VehicleState(
        vehicle_id='ego', current_node=lane['start_node'],
        current_segment=lane['segment_id'], current_lane_id=lane_id,
        current_lane=lane['index'], edge_progress=progress,
        destination_node=lane['end_node'], is_llm=True,
        route_control_authority='llm_maneuver')
    vehicle.pose_x_m, vehicle.pose_y_m = lane['centerline_xy'][-1]
    manager.vehicles['ego'] = vehicle
    return vehicle


def assert_stop_agreement(vehicle, env, expected, awareness=None):
    awareness = awareness or {}
    assert _legitimate_stop(
        vehicle, env, awareness, DrivingProcessScoreConfig()) is expected
    evaluator = DrivingEvaluator.__new__(DrivingEvaluator)
    evaluator.config = DrivingEvaluationConfig()
    assert evaluator._has_legitimate_stop_reason(
        vehicle, env, awareness) is expected


@pytest.mark.parametrize('time_s', [38.8, 59.0, 71.1, 75.5])
def test_terminal_red_light_never_uses_legacy_clock(manager, monkeypatch, time_s):
    vehicle = vehicle_on_lane(manager, TERMINAL, .959041)
    assert manager._current_lane_terminates_at_destination(vehicle)
    assert not manager._prepare_default_continuation(vehicle)
    before = deepcopy(vars(vehicle))
    monkeypatch.setattr(manager.road_network, 'get_traffic_light',
                        lambda *a, **kw: pytest.fail('legacy signal clock read'))
    manager._physics_time = time_s
    env = manager._build_env_view(vehicle, round(time_s))
    assert env.traffic_lights_by_connector
    assert env.dist_to_light_m < 30
    assert_stop_agreement(vehicle, env, True)
    assert vars(vehicle) == before  # Observation never selects or brakes.


@pytest.mark.parametrize('selection,expected', [('', True), (STRAIGHT, False), (LEFT, True)])
def test_mixed_lamps_respect_selection_without_guessing(manager, selection, expected):
    vehicle = vehicle_on_lane(manager, INCOMING)
    vehicle.planned_connector_id = selection
    manager._physics_time = 7.6
    before = deepcopy(vars(vehicle))
    env = manager._build_env_view(vehicle, 8)
    if not selection:
        assert env.traffic_light is None  # Not a fabricated single red/green.
        assert env.traffic_lights_by_connector[STRAIGHT].signal == 'green'
        assert env.traffic_lights_by_connector[LEFT].signal == 'red'
    else:
        assert env.traffic_light.signal == ('red' if expected else 'green')
    assert_stop_agreement(vehicle, env, expected)
    assert vars(vehicle) == before


def test_unsignalized_or_entered_lane_does_not_inherit_a_node_lamp(manager, monkeypatch):
    lane = next(lane for lane in manager._lane_geometry.data['lanes']
                if not manager._lane_geometry.signal_groups_for_lane(lane['id']))
    vehicle = vehicle_on_lane(manager, lane['id'])
    monkeypatch.setattr(manager.road_network, 'get_traffic_light',
                        lambda *a, **kw: pytest.fail('legacy signal clock read'))
    env = manager._build_env_view(vehicle, 0)
    assert env.traffic_light is None and not env.traffic_lights_by_connector
    vehicle = vehicle_on_lane(manager, INCOMING)
    vehicle.active_connector_id = LEFT
    vehicle.planned_connector_id = LEFT
    env = manager._build_env_view(vehicle, 0)
    assert env.traffic_light is None and not env.traffic_lights_by_connector


def test_selected_unsignalized_movement_does_not_inherit_another_turns_red(manager, monkeypatch):
    vehicle = vehicle_on_lane(manager, INCOMING)
    vehicle.planned_connector_id = LEFT
    original = manager._lane_geometry.signal_state
    monkeypatch.setattr(manager._lane_geometry, 'signal_state',
                        lambda cid, t: None if cid == LEFT else original(cid, t))
    manager._physics_time = 38.8
    env = manager._build_env_view(vehicle, 39)
    assert env.traffic_light is None and not env.traffic_lights_by_connector
    assert_stop_agreement(vehicle, env, False)


def test_unselected_all_green_does_not_exempt_unexplained_stopping(manager, monkeypatch):
    vehicle = vehicle_on_lane(manager, INCOMING)
    monkeypatch.setattr(manager._lane_geometry, 'signal_state',
                        lambda cid, t: SimpleNamespace(
                            signal='green', remaining_seconds=10, pedestrian_green=False))
    env = manager._build_env_view(vehicle, 0)
    assert env.traffic_light.signal == 'green'
    assert_stop_agreement(vehicle, env, False)


@pytest.mark.parametrize(
    'on_connector,connector_remaining_m,downstream_gap_m,expected', [
        (True, 4.0, 6.0, True),
        (True, 9.0, 6.0, True),
        (True, 9.1, 6.0, False),
        (True, 30.0, 6.0, False),
        (False, 4.0, 6.0, False),
        (True, 4.0, None, False),
    ])
def test_connector_downstream_vehicle_is_a_legitimate_stop_reason(
        manager, on_connector, connector_remaining_m,
        downstream_gap_m, expected):
    vehicle = vehicle_on_lane(manager, INCOMING)
    vehicle.active_connector_id = STRAIGHT if on_connector else ''
    env = SimpleNamespace(
        traffic_light=None, traffic_lights_by_connector={},
        dist_to_light_m=float('inf'))
    awareness = {
        'downstream_gap_m': downstream_gap_m,
        'distance_to_connector_end_m': connector_remaining_m,
    }
    assert_stop_agreement(vehicle, env, expected, awareness)


@pytest.mark.parametrize('gap_m,expected', [(5.0, True), (15.0, True),
                                           (15.1, False), (None, False)])
def test_same_direction_connector_path_blocker_is_a_legitimate_stop_reason(
        manager, gap_m, expected):
    vehicle = vehicle_on_lane(manager, INCOMING)
    vehicle.active_connector_id = STRAIGHT
    env = SimpleNamespace(
        traffic_light=None, traffic_lights_by_connector={},
        dist_to_light_m=float('inf'))
    assert_stop_agreement(
        vehicle, env, expected,
        {'connector_path_blocker': {
            'vehicle_id': 'merging_peer', 'gap_m': gap_m,
        }})


def test_shared_lane_exports_all_direction_heads_and_live_states(manager):
    vehicle_on_lane(manager, INCOMING)
    scene = build_web3d_scene('hangzhou_binjiang',
                             junction_id='intersection::n1667167449',
                             radius_m=220, include_demo_actors=False)
    heads = [h for mast in scene['static']['signals'] for h in mast['heads']
             if h['lane_id'] == INCOMING]
    assert {h['turn']: h['connector_id'] for h in heads} == {
        'straight': STRAIGHT, 'left': LEFT}
    assert len({tuple(h['position_xz']) for h in heads}) == 2
    assert all(h['width_m'] >= 1.0 for h in heads)
    assert math.dist(heads[0]['position_xz'], heads[1]['position_xz']) == pytest.approx(1.25, abs=.002)
    assert all(h['connector_id'] in h['connector_ids'] for h in heads)
    engine = SimpleNamespace(traffic_mgr=manager, scenario=SimpleNamespace(
        road_network_id='hangzhou_binjiang', physics_step_s=.1))
    frame = Web3DFrameEncoder(focus_entity_id='ego',
        junction_id='intersection::n1667167449').encode(engine, 7.6, 76)
    assert frame['signals'][STRAIGHT] == 'green'
    assert frame['signals'][LEFT] == 'red'
    for mast in scene['static']['signals']:
        for head in mast['heads']:
            assert head['connector_id'] in frame['signals']


def test_parallel_connectors_merge_only_when_turn_and_full_phase_schedule_match():
    runtime = LaneGeometryRuntime.__new__(LaneGeometryRuntime)
    runtime._connectors_from = {'lane': [
        {'id': cid, 'turn': turn} for cid, turn in
        [('a', 'straight'), ('b', 'straight'), ('c', 'left'),
         ('d', 'straight'), ('free', 'right')]]}
    plan = {'node_id': 'j', 'phases': [
        {'connector_ids': ['a', 'b', 'c']}, {'connector_ids': ['d']}]}
    runtime._signal_plan_by_connector = dict.fromkeys(['a', 'b', 'c', 'd'], plan)
    groups = runtime.signal_groups_for_lane('lane')
    assert [(g['turn'], g['connector_ids']) for g in groups] == [
        ('left', ['c']), ('straight', ['a', 'b']), ('straight', ['d'])]
    assert runtime.signal_groups_for_lane('missing') == []


def test_four_direction_heads_fit_their_lane_without_overlap(manager):
    lane_id = 'n5790081082_n8900831904::lane_0'
    lane = manager._lane_geometry._lane_by_id[lane_id]
    scene = build_web3d_scene('hangzhou_binjiang',
                             center_world_xy=lane['centerline_xy'][-1],
                             radius_m=220, include_demo_actors=False)
    heads = [h for mast in scene['static']['signals'] for h in mast['heads']
             if h['lane_id'] == lane_id]
    assert {h['turn'] for h in heads} == {'uturn', 'left', 'straight', 'right'}
    assert len(heads) == 4
    span = max(math.dist(a['position_xz'], b['position_xz'])
               + (a['width_m'] + b['width_m']) / 2
               for a in heads for b in heads)
    assert span <= lane['width_m'] + .002
    for index, first in enumerate(heads):
        for second in heads[index + 1:]:
            assert math.dist(first['position_xz'], second['position_xz']) > (
                first['width_m'] + second['width_m']) / 2


def test_missing_authored_uturn_stop_line_still_has_a_visible_live_signal():
    manager = TrafficCoordinator(load_road_network('chengdu_chunxi'))
    lane_id = 'n3516900121_n5529709290::lane_0'
    cid = 'connector::n3516900121::0'
    data = manager._lane_geometry.data
    before = deepcopy(data)
    assert not any(s['lane_id'] == lane_id for s in data['stop_lines'])
    stops = [s for s in signal_stop_lines(data) if s['lane_id'] == lane_id]
    assert len(stops) == 1 and stops[0]['source'] == 'lane_endpoint_signal_display'
    lane = manager._lane_geometry._lane_by_id[lane_id]
    assert stops[0]['line_xy'] == [lane['left_boundary_xy'][-1], lane['right_boundary_xy'][-1]]
    scene = build_web3d_scene('chengdu_chunxi', junction_id='n3516900121',
                             include_demo_actors=False)
    head = next(h for m in scene['static']['signals'] for h in m['heads']
                if h['connector_id'] == cid)
    assert head['turn'] == 'uturn'
    vehicle_on_lane(manager, lane_id)
    engine = SimpleNamespace(traffic_mgr=manager, scenario=SimpleNamespace(
        road_network_id='chengdu_chunxi', physics_step_s=.1))
    frame = Web3DFrameEncoder(focus_entity_id='ego', junction_id='n3516900121').encode(engine, 0, 0)
    assert frame['signals'][cid] == 'green'
    manager._physics_time = 15.0
    env = manager._build_env_view(manager.vehicles['ego'], 15)
    assert env.traffic_light.signal == 'red'
    assert env.dist_to_light_m == pytest.approx(.03 * manager._lane_geometry._lane_lengths[lane_id])
    assert manager._lane_geometry.data == before  # No map or physics mutation.


def test_browser_shows_independent_direction_arrows_and_tracks_phase_changes(manager, tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip('Playwright Chromium is not installed')
    vehicle = vehicle_on_lane(manager, INCOMING)
    lane = manager._lane_geometry._lane_by_id[INCOMING]
    vehicle.pose_x_m, vehicle.pose_y_m, vehicle.yaw_rad = pose_at(
        lane['centerline_xy'], manager._lane_geometry._lane_lengths[INCOMING] - 8)
    engine = SimpleNamespace(traffic_mgr=manager, scenario=SimpleNamespace(
        road_network_id='hangzhou_binjiang', physics_step_s=.1))
    renderer = Web3DCameraRenderer()
    try:
        renderer._start()
        page = renderer._browser.new_page(viewport={'width': 1024, 'height': 960})
        def instrument(route):
            response = route.fetch()
            route.fulfill(response=response, body=response.text() + '''
              window.signalProbe = () => signalControllers.map(c => ({
                id: c.connectorId, turn: c.turn,
                active: Object.keys(c.bulbs).filter(k => c.bulbs[k].material.emissiveIntensity > 1),
                geometry: c.bulbs.red.geometry.type,
                side: c.bulbs.red.material.side,
                toneMapped: c.bulbs.red.material.toneMapped,
                scale: c.bulbs.red.scale.x,
                colors: Object.fromEntries(Object.entries(c.bulbs).map(([k, b]) => [k, {
                  emission: b.material.emissiveIntensity, color: b.material.color.getHex()}])),
                vertices: Array.from(c.bulbs.red.geometry.attributes.position.array)}));
            ''')
        page.route('**/app.js', instrument)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{renderer._server.server_address[1]}/?view=cockpit&capture=1',
                  wait_until='networkidle')
        page.wait_for_function("typeof window.signalProbe === 'function'")
        encoder = Web3DFrameEncoder(focus_entity_id='ego',
                                    junction_id='intersection::n1667167449')
        for time_s in (7.6, 64.0, 25.5):
            frame = encoder.encode(engine, time_s, round(time_s * 10))
            page.evaluate('f => window.vehicleArenaCaptureFrame(f)', frame)
            rendered = {x['id']: x for x in page.evaluate('window.signalProbe()')}
            for cid, turn in ((STRAIGHT, 'straight'), (LEFT, 'left')):
                assert rendered[cid]['active'] == [frame['signals'][cid]]
                assert rendered[cid]['turn'] == turn
                assert rendered[cid]['geometry'] == 'ShapeGeometry'
                assert rendered[cid]['side'] == 0  # THREE.FrontSide
                assert rendered[cid]['toneMapped'] is False
                vertices = rendered[cid]['vertices']
                assert (max(vertices[::3]) - min(vertices[::3])) * rendered[cid]['scale'] > .65
                for name, material in rendered[cid]['colors'].items():
                    if name != frame['signals'][cid]:
                        assert material == {'emission': 0, 'color': 0x080a0b}
            assert rendered[STRAIGHT]['vertices'] != rendered[LEFT]['vertices']
            page.screenshot(path=str(tmp_path / f'signals-{time_s}.png'))
        assert not errors
    finally:
        renderer.close()
