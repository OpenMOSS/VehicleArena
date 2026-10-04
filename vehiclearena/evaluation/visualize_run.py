"""
Visualize an agent simulation run on a road network and export as GIF.

Runs the ``multi_basic_rain`` scenario (beijing_zhongguancun, 2 vehicles)
with PerfectAgent, records per-tick positions, and renders an animated
GIF showing vehicles moving on the real OSM road network.

Usage:
    cd vehiclearena
    python evaluation/visualize_run.py
"""

import sys
import os
import copy
import io
import json
import math
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import imageio.v2 as imageio

from simulation.multi_sim_engine import MultiSimEngine, MultiScenario
from simulation.road_network import RoadNetwork
from simulation.road_networks import load_road_network
from simulation.sumo_traffic_manager import SumoTrafficManager
from simulation.ground_truth_rules import derive_ground_truth
from simulation.memory import SessionHistory
from vehiclearena import VehicleWorld
from utils import execute
from evaluation.check_utils import _FULL_INIT_PREFIX


# ── Scenarios ─────────────────────────────────────────────────────────
# Import all scenarios from test_multi_agent
from evaluation.test_multi_agent import (
    SCENARIO_BASIC_RAIN, SCENARIO_CONVOY, SCENARIO_COMPLEX,
)

SCENARIOS = {
    "basic":   SCENARIO_BASIC_RAIN,
    "convoy":  SCENARIO_CONVOY,
    "complex": SCENARIO_COMPLEX,
}

# Active scenario — set by main() via --scenario flag
SCENARIO_DICT = SCENARIO_COMPLEX


# ── Coordinate interpolation ────────────────────────────────────────

def _segment_polyline(road_network, segment_id):
    """Get the full polyline (list of [lat, lng]) for a segment."""
    seg = road_network.edges.get(segment_id)
    if seg is None:
        return []
    n1 = road_network.nodes.get(seg.from_node)
    n2 = road_network.nodes.get(seg.to_node)
    if n1 is None or n2 is None:
        return []
    if seg.geometry:
        return seg.geometry
    return [[n1.lat, n1.lng], [n2.lat, n2.lng]]


def _interp_polyline(polyline, progress):
    """Interpolate a position along a polyline at a given progress [0, 1]."""
    if not polyline:
        return None, None
    if len(polyline) == 1:
        return polyline[0][0], polyline[0][1]

    # Compute cumulative distances
    dists = [0.0]
    for i in range(1, len(polyline)):
        d = math.sqrt(
            (polyline[i][0] - polyline[i - 1][0]) ** 2 +
            (polyline[i][1] - polyline[i - 1][1]) ** 2)
        dists.append(dists[-1] + d)
    total = dists[-1]
    if total < 1e-12:
        return polyline[0][0], polyline[0][1]

    target = progress * total
    for i in range(1, len(dists)):
        if dists[i] >= target:
            frac = (target - dists[i - 1]) / (dists[i] - dists[i - 1] + 1e-15)
            lat = polyline[i - 1][0] + frac * (polyline[i][0] - polyline[i - 1][0])
            lng = polyline[i - 1][1] + frac * (polyline[i][1] - polyline[i - 1][1])
            return lat, lng
    return polyline[-1][0], polyline[-1][1]


def _partial_polyline_points(polyline, progress):
    """Return geometry points from start of polyline up to *progress* [0, 1]."""
    if not polyline or len(polyline) < 2:
        return list(polyline) if polyline else []
    if progress >= 1.0:
        return list(polyline)
    if progress <= 0.0:
        return [polyline[0]]

    dists = [0.0]
    for i in range(1, len(polyline)):
        d = math.sqrt((polyline[i][0] - polyline[i - 1][0]) ** 2 +
                       (polyline[i][1] - polyline[i - 1][1]) ** 2)
        dists.append(dists[-1] + d)
    total = dists[-1]
    if total < 1e-12:
        return [polyline[0]]

    target = progress * total
    result = [polyline[0]]
    for i in range(1, len(polyline)):
        if dists[i] <= target:
            result.append(polyline[i])
        else:
            frac = (target - dists[i - 1]) / (dists[i] - dists[i - 1] + 1e-15)
            lat = polyline[i - 1][0] + frac * (polyline[i][0] - polyline[i - 1][0])
            lng = polyline[i - 1][1] + frac * (polyline[i][1] - polyline[i - 1][1])
            result.append([lat, lng])
            break
    return result


def _oriented_polyline(road_network, seg_id, from_node_id):
    """Get segment polyline oriented so it starts from *from_node_id*."""
    seg = road_network.edges.get(seg_id)
    if not seg:
        return []
    polyline = _segment_polyline(road_network, seg_id)
    if not polyline:
        return []
    # Use coordinate distance to decide direction (more robust than node ID match)
    node = road_network.nodes.get(from_node_id)
    if node:
        d_start = (polyline[0][0] - node.lat) ** 2 + (polyline[0][1] - node.lng) ** 2
        d_end = (polyline[-1][0] - node.lat) ** 2 + (polyline[-1][1] - node.lng) ** 2
        if d_end < d_start:
            return list(reversed(polyline))
    return polyline


def _find_segment_between(road_network, node_a, node_b):
    """Find segment connecting two adjacent nodes; return (seg_id, reversed)."""
    for seg_id, seg in road_network.edges.items():
        if seg.from_node == node_a and seg.to_node == node_b:
            return seg_id, False
        if seg.from_node == node_b and seg.to_node == node_a:
            return seg_id, True
    return None, False


def orient_segment_chain(road_network, seg_ids, start_node_id):
    """Orient a chain of actually-traversed segments so they connect end-to-end.

    Returns a list of polylines (one per segment), each oriented so that
    consecutive polylines connect head-to-tail.
    """
    oriented = []
    prev_end = start_node_id
    for seg_id in seg_ids:
        seg = road_network.edges.get(seg_id)
        if not seg:
            oriented.append([])
            continue
        polyline = _segment_polyline(road_network, seg_id)
        if not polyline:
            oriented.append([])
            continue
        # Orient so polyline starts near prev_end node
        node = road_network.nodes.get(prev_end)
        if node:
            d_start = (polyline[0][0] - node.lat) ** 2 + (polyline[0][1] - node.lng) ** 2
            d_end = (polyline[-1][0] - node.lat) ** 2 + (polyline[-1][1] - node.lng) ** 2
            if d_end < d_start:
                polyline = list(reversed(polyline))
        oriented.append(polyline)
        # Update prev_end to the OTHER end of this segment
        if seg.from_node == prev_end:
            prev_end = seg.to_node
        else:
            prev_end = seg.from_node
    return oriented


def build_route_trajectory(road_network, route, reached_idx,
                           current_seg, current_progress):
    """Build (lat, lng) list following road geometry along a vehicle's route.

    Args:
        route: full list of route node IDs
        reached_idx: index of the last route node the vehicle has reached
        current_seg: segment the vehicle is currently on (may be empty)
        current_progress: edge_progress on current_seg [0, 1]
    """
    points = []

    def _append(pt):
        pos = (pt[0], pt[1])
        if not points or (abs(points[-1][0] - pos[0]) > 1e-10 or
                          abs(points[-1][1] - pos[1]) > 1e-10):
            points.append(pos)

    # Fully-traversed segments (route[0]→route[1], …, route[reached_idx-1]→route[reached_idx])
    for i in range(reached_idx):
        seg_id, is_rev = _find_segment_between(
            road_network, route[i], route[i + 1])
        if seg_id:
            polyline = _oriented_polyline(road_network, seg_id, route[i])
            for pt in polyline:
                _append(pt)
        else:
            node = road_network.nodes.get(route[i + 1])
            if node:
                _append([node.lat, node.lng])

    # Partial current segment
    if current_seg and reached_idx < len(route) - 1:
        seg = road_network.edges.get(current_seg)
        if seg:
            polyline = _oriented_polyline(
                road_network, current_seg, route[reached_idx])
            # If the segment from_node != route node, progress is inverted
            prog = current_progress
            if seg.from_node != route[reached_idx]:
                prog = 1.0 - current_progress
            pts = _partial_polyline_points(polyline, prog)
            for pt in pts:
                _append(pt)
    elif not current_seg:
        # Vehicle sitting at a node (arrived or waiting)
        node = road_network.nodes.get(route[reached_idx])
        if node:
            _append([node.lat, node.lng])

    return points


def get_vehicle_latlng(road_network, vs):
    """Project the authoritative lane-level pose back to latitude/longitude."""
    manager = getattr(vs, "_traffic_mgr", None)
    runtime = getattr(manager, "_lane_geometry", None)
    projection = getattr(runtime, "projection", {}) if runtime else {}
    origin_lat = projection.get("origin_lat")
    origin_lng = projection.get("origin_lng")
    if origin_lat is not None and origin_lng is not None:
        lat = float(origin_lat) + float(vs.pose_y_m) / 110_540.0
        lng = (
            float(origin_lng) + float(vs.pose_x_m)
            / (111_320.0 * math.cos(math.radians(float(origin_lat)))))
        return lat, lng
    node = road_network.nodes.get(vs.current_node)
    if node:
        return node.lat, node.lng
    return None, None


# ── Run simulation and record trajectory ────────────────────────────

def run_and_record():
    """Run simulation and return (road_network, frames).

    Each frame is a dict:
        {
            "tick": int, "time_min": float,
            "weather": str, "period": str,
            "vehicles": {
                vid: {"lat": float, "lng": float,
                      "speed": float, "arrived": bool, "node": str}
            }
        }
    """
    scenario = MultiScenario.from_dict(SCENARIO_DICT)
    road_network = load_road_network(scenario.road_network_id)
    road_network.load_scenario_events(
        scenario._grid_events,
        tick_interval_s=scenario.tick_interval_s)
    road_network.config.seconds_per_tick = 1  # traffic lights use 1-second resolution

    traffic_mgr = SumoTrafficManager(road_network)

    # Initialize vehicles
    vw_map = {}
    expect_vw_map = {}
    memory_map = {}
    prev_snap = {}
    for vcfg in scenario.vehicles:
        vid = vcfg.vehicle_id
        traffic_mgr.register_vehicle(
            vehicle_id=vid,
            start_node=vcfg.initial_node,
            destination=vcfg.destination_node,
            destination_name=vcfg.destination_name,
            start_lane=vcfg.initial_lane,
            auto_navigate=bool(vcfg.destination_node),
        )
        vw_map[vid] = VehicleWorld()
        expect_vw_map[vid] = VehicleWorld()
        if scenario.inits_code.strip():
            full_init = _FULL_INIT_PREFIX + scenario.inits_code
            execute(full_init, local_vars={'vw': vw_map[vid]}, global_vars=None)
            execute(full_init, local_vars={'vw': expect_vw_map[vid]}, global_vars=None)
        memory_map[vid] = SessionHistory()
        prev_snap[vid] = None

    # Weather/daynight lookup helpers
    def weather_at(t):
        w = "sunny"
        for kf in scenario.weather_keyframes:
            if kf.t <= t:
                w = kf.condition
        return w

    def period_at(t):
        p = "day"
        for kf in scenario.daynight_keyframes:
            if kf.t <= t:
                p = kf.period
        return p

    # Track physical segments traversed for trajectory rendering.
    seg_history = {}        # vid → [seg_id, ...] in order of traversal
    previous_segment = {}   # vid → last sampled physical segment
    start_nodes = {}        # vid → initial node
    for vcfg in scenario.vehicles:
        start_nodes[vcfg.vehicle_id] = vcfg.initial_node

    # Tick loop — advance physics in fine sub-steps for smooth animation
    frames = []
    interval = scenario.tick_interval_s
    sub_step = 10.0  # record a frame every 10 seconds
    t = 0.0
    tick_index = 0
    arrived_set = set()
    prev_nodes = {}           # vid → last known node (for intersection events)

    total_time_s = scenario.total_time_s
    current_sub = 0.0
    all_arrived_since = None  # track when all vehicles arrived

    while current_sub <= total_time_s:
        # Advance physics to current_sub
        traffic_mgr.recalculate_speeds(time_s=current_sub)
        traffic_mgr.advance_world_to(current_sub)

        # Record physical segment transitions.
        for vcfg in scenario.vehicles:
            vid = vcfg.vehicle_id
            vs = traffic_mgr.get_state(vid)
            if not vs or not vs.current_segment:
                continue
            if previous_segment.get(vid) != vs.current_segment:
                seg_history.setdefault(vid, []).append(vs.current_segment)
                previous_segment[vid] = vs.current_segment

        # Detect events at this time
        cur_weather = weather_at(current_sub)
        cur_period = period_at(current_sub)
        events_now = []

        # Simulation start
        if not frames:
            events_now.append(("sim_start",))

        # Weather change
        if frames and cur_weather != frames[-1]["weather"]:
            events_now.append(("weather", cur_weather))
        # Daynight change
        if frames and cur_period != frames[-1]["period"]:
            events_now.append(("daynight", cur_period))
        # Per-vehicle events: intersection arrival, traffic light, arrival
        for vcfg in scenario.vehicles:
            vid = vcfg.vehicle_id
            vs = traffic_mgr.get_state(vid)
            if not vs:
                continue

            # Intersection arrival (node changed)
            cur_node = vs.current_node
            if cur_node and cur_node != prev_nodes.get(vid):
                if prev_nodes.get(vid) is not None:  # skip initial position
                    # Check if signal node
                    node_obj = road_network.nodes.get(cur_node)
                    has_signal = node_obj and node_obj.signal
                    if has_signal:
                        from_node_id = prev_nodes.get(vid)
                        light = road_network.get_traffic_light(
                            cur_node, tick_index, from_node=from_node_id)
                        sig = light.signal if light else "?"
                        if node_obj.has_crosswalk:
                            if light and light.is_crosswalk_phase:
                                extras = "crosswalk ACTIVE"
                            else:
                                extras = f"{sig}, crosswalk"
                        else:
                            extras = sig
                        events_now.append((
                            "intersection", vid, extras))
                prev_nodes[vid] = cur_node

            # Vehicle arrival
            if vs.arrived and vid not in arrived_set:
                events_now.append(("arrived", vid))

        # Record frame
        frame = {
            "tick": tick_index,
            "time_s": current_sub,
            "weather": cur_weather,
            "period": cur_period,
            "events": events_now,
            "vehicles": {},
        }
        for vcfg in scenario.vehicles:
            vid = vcfg.vehicle_id
            vs = traffic_mgr.get_state(vid)
            if vs and vs.arrived:
                arrived_set.add(vid)

            lat, lng = get_vehicle_latlng(road_network, vs)
            frame["vehicles"][vid] = {
                "lat": lat,
                "lng": lng,
                "speed": vs.current_speed_kmh if vs else 0,
                "arrived": vid in arrived_set,
                "node": vs.current_node if vs else "",
                "segment": vs.current_segment if vs else "",
                "progress": vs.edge_progress if vs else 0,
            }
        frames.append(frame)

        # Stop once all vehicles have arrived
        all_vids = {vcfg.vehicle_id for vcfg in scenario.vehicles}
        if arrived_set >= all_vids:
            if all_arrived_since is None:
                all_arrived_since = current_sub
            break

        # At full tick boundaries, run GT + agent actions
        if abs(current_sub - t) < 0.01:
            for vcfg in scenario.vehicles:
                vid = vcfg.vehicle_id
                vw = vw_map[vid]
                vs = traffic_mgr.get_state(vid)

                gt_lines, cur_snap, _, _ = derive_ground_truth(
                    vw=vw,
                    prev_snapshot=prev_snap[vid],
                    vehicle_state=vs,
                )
                prev_snap[vid] = cur_snap

                gt_actions = []
                for line in gt_lines:
                    line = line.strip()
                    if line:
                        try:
                            execute(line, local_vars={'vw': vw}, global_vars=None)
                            gt_actions.append(line)
                        except Exception:
                            pass

                # Add GT actions as events on the most recent frame
                if gt_actions and frames:
                    for act in gt_actions:
                        frames[-1]["events"].append(("action", vid, act))

            t += interval
            tick_index += 1

        current_sub += sub_step

    print(f"Recorded {len(frames)} frames, "
          f"{len(scenario.vehicles)} vehicles")
    return road_network, frames, seg_history, start_nodes


# ── Rendering ───────────────────────────────────────────────────────

_COLOR_PALETTE = [
    "#2196F3", "#FF9800", "#4CAF50", "#9C27B0",
    "#F44336", "#00BCD4", "#795548", "#607D8B",
]


def _build_vehicle_styles(scenario_dict):
    """Auto-generate colors and labels from scenario vehicle list."""
    colors, labels = {}, {}
    for i, vcfg in enumerate(scenario_dict["vehicles"]):
        vid = vcfg["vehicle_id"]
        colors[vid] = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
        labels[vid] = vid.replace("_", " ").title()
    return colors, labels


VEHICLE_COLORS, VEHICLE_LABELS = _build_vehicle_styles(SCENARIO_DICT)

# Weather → background tint
WEATHER_BG = {
    "sunny": "#FFFDE7",
    "rainy": "#E3F2FD",
    "snowy": "#ECEFF1",
    "heavy_snow": "#CFD8DC",
    "foggy": "#F5F5F5",
}
PERIOD_ALPHA = {
    "morning": 1.0,
    "afternoon": 1.0,
    "dusk": 0.85,
    "night": 0.7,
    "dawn": 0.9,
    "day": 1.0,
}


def build_road_lines(road_network):
    """Pre-compute line segments for all roads (as lat/lng arrays)."""
    lines = []
    road_types = []
    for seg in road_network.edges.values():
        n1 = road_network.nodes.get(seg.from_node)
        n2 = road_network.nodes.get(seg.to_node)
        if n1 is None or n2 is None:
            continue
        if seg.geometry:
            pts = [(p[1], p[0]) for p in seg.geometry]  # lng, lat
        else:
            pts = [(n1.lng, n1.lat), (n2.lng, n2.lat)]
        if len(pts) >= 2:
            lines.append(pts)
            road_types.append(seg.road_type)
    return lines, road_types


ROAD_WIDTHS = {
    "motorway": 1.2, "trunk": 1.0, "primary": 1.0,
    "secondary": 0.8, "tertiary": 0.8, "residential": 0.6,
    "urban": 0.6,
}
ROAD_COLORS = {
    "motorway": "#BDBDBD", "trunk": "#BDBDBD", "primary": "#BDBDBD",
    "secondary": "#BDBDBD", "tertiary": "#CFD8DC", "residential": "#D7CCC8",
    "urban": "#C5CAE9",
}


def render_frame(road_network, road_lines, road_types, frame,
                 trajectory, bounds, fig_size=(10, 10), dpi=100):
    """Render a single frame and return as RGB numpy array."""
    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)

    bg = WEATHER_BG.get(frame["weather"], "#FFFFFF")
    alpha = PERIOD_ALPHA.get(frame["period"], 1.0)
    fig.patch.set_facecolor(bg)
    fig.patch.set_alpha(alpha)
    ax.set_facecolor(bg)

    lng_min, lng_max, lat_min, lat_max = bounds
    margin = 0.001
    ax.set_xlim(lng_min - margin, lng_max + margin)
    ax.set_ylim(lat_min - margin, lat_max + margin)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6)
    ax.ticklabel_format(useOffset=False, style='plain')

    # Draw road segments
    for pts, rtype in zip(road_lines, road_types):
        color = ROAD_COLORS.get(rtype, "#E0E0E0")
        width = ROAD_WIDTHS.get(rtype, 0.8)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=width, solid_capstyle='round',
                zorder=1)

    # Build reverse adjacency (incoming edges) for direction-aware rendering
    incoming = defaultdict(set)
    for src, dsts in road_network.adjacency.items():
        for dst in dsts:
            incoming[dst].add(src)

    # Draw signal nodes — cluster nearby signals (same intersection)
    signal_nodes = [(nid, n) for nid, n in road_network.nodes.items()
                    if n.signal]
    # Cluster signals within ~0.001 degree (~100m)
    clusters = []
    used = set()
    for i, (nid, n) in enumerate(signal_nodes):
        if nid in used:
            continue
        cluster = [(nid, n)]
        used.add(nid)
        for j, (nid2, n2) in enumerate(signal_nodes):
            if nid2 in used:
                continue
            if abs(n.lat - n2.lat) < 0.001 and abs(n.lng - n2.lng) < 0.001:
                cluster.append((nid2, n2))
                used.add(nid2)
        clusters.append(cluster)

    SIGNAL_COLORS = {"green": "#4CAF50", "yellow": "#FFC107",
                     "red": "#F44336"}
    tick = frame["tick"]
    for cluster in clusters:
        # Draw per-node: base dot + direction arms at each node's own position
        arm_len = 0.00018  # degrees, ~18m visual length
        for nid, node in cluster:
            nlng, nlat = node.lng, node.lat
            # Base dot (small, semi-transparent)
            ax.plot(nlng, nlat, 'o', color='#616161', markersize=2.5,
                    alpha=0.6, zorder=5, markeredgewidth=0)
            # Per-approach direction-colored line segments
            is_ped_phase = False
            for approach_id in incoming.get(nid, set()):
                nbr = road_network.nodes.get(approach_id)
                if not nbr:
                    continue
                light = road_network.get_traffic_light(
                    nid, tick, from_node=approach_id)
                if not light:
                    continue
                if light.is_crosswalk_phase:
                    is_ped_phase = True
                sig_color = SIGNAL_COLORS.get(light.signal, "#9E9E9E")
                edge = road_network.get_edge_by_nodes(approach_id, nid)
                if edge and len(edge.geometry) >= 2:
                    gp = edge.geometry[-2]
                    dx = gp[1] - nlng
                    dy = gp[0] - nlat
                else:
                    dx = nbr.lng - nlng
                    dy = nbr.lat - nlat
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 1e-9:
                    continue
                dx, dy = dx / dist, dy / dist
                ax.plot([nlng, nlng + dx * arm_len],
                        [nlat, nlat + dy * arm_len],
                        color=sig_color, linewidth=1.8, alpha=0.8,
                        solid_capstyle='round', zorder=6)
            if is_ped_phase:
                ax.plot(nlng, nlat, 'o', color='white', markersize=2,
                        alpha=0.9, zorder=7, markeredgewidth=0)

    # Draw crosswalk marks — use clusters to avoid duplicate bars
    for cluster in clusters:
        # Skip cluster if no crosswalk node
        cw_nodes = [(nid, n) for nid, n in cluster if n.has_crosswalk]
        if not cw_nodes:
            continue
        # Check pedestrian phase from first crosswalk node
        first_nid = cw_nodes[0][0]
        light = road_network.get_traffic_light(first_nid, tick)
        bar_color = "#555555"
        bar_alpha = 0.45
        if light and light.is_crosswalk_phase:
            bar_color = "#4CAF50"
            bar_alpha = 0.8
        # Collect unique approach directions across all nodes in cluster
        drawn_dirs = set()
        for nid, node in cw_nodes:
            for nbr_id in incoming.get(nid, set()):
                nbr = road_network.nodes.get(nbr_id)
                if not nbr:
                    continue
                # Use edge geometry for accurate direction at intersection
                edge = road_network.get_edge_by_nodes(nbr_id, nid)
                if edge and len(edge.geometry) >= 2:
                    gp = edge.geometry[-2]
                    dx = gp[1] - node.lng
                    dy = gp[0] - node.lat
                else:
                    dx = nbr.lng - node.lng
                    dy = nbr.lat - node.lat
                length = math.sqrt(dx * dx + dy * dy)
                if length < 1e-9:
                    continue
                ndx, ndy = dx / length, dy / length
                # Quantize direction to 10° bins to deduplicate similar approaches
                angle_bin = round(math.atan2(ndy, ndx) * 18 / math.pi)
                if angle_bin in drawn_dirs:
                    continue
                drawn_dirs.add(angle_bin)
                offset = 0.00025
                cx = node.lng + offset * ndx
                cy = node.lat + offset * ndy
                px, py = -ndy, ndx
                bar_half = 0.0001
                ax.plot([cx - bar_half * px, cx + bar_half * px],
                        [cy - bar_half * py, cy + bar_half * py],
                        color=bar_color, linewidth=2.0, alpha=bar_alpha,
                        zorder=4, solid_capstyle='butt')

    # Draw destination markers
    for vid, vdata in frame["vehicles"].items():
        # Find destination node
        for vcfg in SCENARIO_DICT["vehicles"]:
            if vcfg["vehicle_id"] == vid:
                dest_id = vcfg["destination_node"]
                dest_node = road_network.nodes.get(dest_id)
                if dest_node:
                    color = VEHICLE_COLORS.get(vid, "#666")
                    ax.plot(dest_node.lng, dest_node.lat, '*',
                            color=color, markersize=8, alpha=0.4,
                            zorder=6, markeredgewidth=0.3,
                            markeredgecolor='#666')

    # Draw trajectory lines
    for vid, color in VEHICLE_COLORS.items():
        traj = trajectory.get(vid, [])
        if len(traj) >= 2:
            lngs = [p[1] for p in traj]
            lats = [p[0] for p in traj]
            ax.plot(lngs, lats, color=color, linewidth=1.5,
                    alpha=0.3, zorder=7, solid_capstyle='round')

    # Draw vehicle positions — square=LLM, circle=SUMO background
    # Build LLM vehicle set from scenario
    llm_vids = set()
    for vcfg in SCENARIO_DICT.get("vehicles", []):
        if vcfg.get("is_evaluated"):
            llm_vids.add(vcfg["vehicle_id"])

    for vid, vdata in frame["vehicles"].items():
        if vdata["lat"] is None:
            continue
        color = VEHICLE_COLORS.get(vid, "#666")
        label = VEHICLE_LABELS.get(vid, vid)
        is_llm = vid in llm_vids
        marker = 's' if is_llm else 'o'
        size = 7
        alpha = 0.85 if is_llm else 0.6
        ax.plot(vdata["lng"], vdata["lat"], marker, color=color,
                markersize=size, alpha=alpha, zorder=10,
                markeredgewidth=0.6, markeredgecolor='white')
        spd = f'{vdata["speed"]:.0f}km/h'
        ann_text = f'{label}\n{"ARRIVED" if vdata["arrived"] else spd}'
        ax.annotate(
            ann_text,
            (vdata["lng"], vdata["lat"]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=6, fontweight='bold', color=color,
            alpha=0.85, zorder=11,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.65, edgecolor=color, linewidth=0.4),
        )


    # Title with status info
    weather_icon = {"sunny": "Clear", "rainy": "Rain",
                    "snowy": "Light snow", "heavy_snow": "Heavy snow",
                    "foggy": "Fog"}.get(
                        frame["weather"], frame["weather"])
    period_str = frame["period"].capitalize()
    title = (f'Tick {frame["tick"]}  |  t = {frame["time_min"]:.0f} min  |  '
             f'{weather_icon}  |  {period_str}')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    # Status bar — split into multiple lines to avoid overflow
    status_parts = []
    for vid in sorted(frame["vehicles"]):
        vd = frame["vehicles"][vid]
        lbl = VEHICLE_LABELS.get(vid, vid)
        if vd["arrived"]:
            st = "ARRIVED"
        else:
            st = f'{vd["speed"]:.0f}km/h'
        status_parts.append(f'{lbl}:{st}')
    # Arrange into rows of ~8 vehicles each
    row_size = 8
    rows = []
    for i in range(0, len(status_parts), row_size):
        rows.append('  |  '.join(status_parts[i:i + row_size]))
    ax.set_ylabel('')
    ax.set_xlabel('\n'.join(rows), fontsize=7, labelpad=6)

    n_rows = (len(frame["vehicles"]) + 7) // 8
    bot = 0.05 + n_rows * 0.03
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=bot)

    # Render to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = imageio.imread(buf)
    plt.close(fig)
    return img


def render_scenario_dict(scenario_dict: dict, output_path: str,
                         verbose: bool = True) -> str:
    """Render a GIF for an arbitrary scenario dict.

    Args:
        scenario_dict: Dict accepted by MultiScenario.from_dict().
        output_path: Path to write the GIF file.
        verbose: Print progress.

    Returns:
        The output_path on success.
    """
    global SCENARIO_DICT, VEHICLE_COLORS, VEHICLE_LABELS
    SCENARIO_DICT = scenario_dict
    VEHICLE_COLORS, VEHICLE_LABELS = _build_vehicle_styles(scenario_dict)

    if verbose:
        print(f"  Rendering: {scenario_dict.get('name', scenario_dict['scenario_id'])}")

    road_network, frames, seg_history, start_nodes = run_and_record()
    road_lines, road_types = build_road_lines(road_network)

    # Compute bounds from vehicle trajectories
    all_vlats, all_vlngs = [], []
    for f in frames:
        for vd in f["vehicles"].values():
            if vd["lat"] is not None:
                all_vlats.append(vd["lat"])
                all_vlngs.append(vd["lng"])
    if all_vlats:
        pad = 0.003
        bounds = (
            min(all_vlngs) - pad, max(all_vlngs) + pad,
            min(all_vlats) - pad, max(all_vlats) + pad,
        )
    else:
        lats = [n.lat for n in road_network.nodes.values()]
        lngs = [n.lng for n in road_network.nodes.values()]
        bounds = (min(lngs), max(lngs), min(lats), max(lats))

    # Precompute oriented polylines
    oriented_chains = {}
    for vid, segs in seg_history.items():
        oriented_chains[vid] = orient_segment_chain(
            road_network, segs, start_nodes[vid])
    seg_index = {}
    for vid, segs in seg_history.items():
        seg_index[vid] = {sid: idx for idx, sid in enumerate(segs)}

    images = []

    for i, frame in enumerate(frames):
        trajectory = {}
        for vid in seg_history:
            chain = oriented_chains[vid]
            vd = frame["vehicles"].get(vid, {})
            cur_seg = vd.get("segment", "")
            idx_map = seg_index[vid]
            if vd.get("arrived"):
                draw_up_to = len(chain)
            elif cur_seg in idx_map:
                draw_up_to = idx_map[cur_seg]
            else:
                draw_up_to = len(chain)

            points = []
            for j in range(draw_up_to):
                for pt in chain[j]:
                    pos = (pt[0], pt[1])
                    if not points or (
                            abs(points[-1][0] - pos[0]) > 1e-10 or
                            abs(points[-1][1] - pos[1]) > 1e-10):
                        points.append(pos)
            if vd.get("lat") is not None and not vd.get("arrived"):
                car_pos = (vd["lat"], vd["lng"])
                if not points or (
                        abs(points[-1][0] - car_pos[0]) > 1e-7 or
                        abs(points[-1][1] - car_pos[1]) > 1e-7):
                    points.append(car_pos)
            trajectory[vid] = points

        img = render_frame(road_network, road_lines, road_types,
                           frame, trajectory, bounds)
        images.append(img)

    # Save GIF
    durations = []
    for i, frame in enumerate(frames):
        if i == 0 or i == len(frames) - 1:
            durations.append(2000)
        else:
            durations.append(600)

    from PIL import Image
    pil_frames = [Image.fromarray(img) for img in images]
    pil_frames[0].save(
        output_path, save_all=True, append_images=pil_frames[1:],
        duration=durations, loop=0, optimize=False,
    )
    traffic_mgr.close()
    if verbose:
        file_size_kb = os.path.getsize(output_path) / 1024
        print(f"  GIF saved: {output_path} ({len(images)} frames, {file_size_kb:.0f} KB)")
    return output_path


def main():
    global SCENARIO_DICT, VEHICLE_COLORS, VEHICLE_LABELS

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS) + ["all"],
                        default="complex",
                        help="Which scenario to visualize (default: complex)")
    args = parser.parse_args()

    if args.scenario == "all":
        for name in SCENARIOS:
            print(f"\n{'=' * 60}")
            print(f"  Scenario: {name}")
            print(f"{'=' * 60}")
            SCENARIO_DICT = SCENARIOS[name]
            VEHICLE_COLORS, VEHICLE_LABELS = _build_vehicle_styles(SCENARIO_DICT)
            _run_one_scenario(name)
        print("\nAll scenarios done!")
        return

    SCENARIO_DICT = SCENARIOS[args.scenario]
    VEHICLE_COLORS, VEHICLE_LABELS = _build_vehicle_styles(SCENARIO_DICT)
    _run_one_scenario(args.scenario)


def _run_one_scenario(scenario_name):
    print("=" * 60)
    print(f"VehicleArena — {SCENARIO_DICT['name']}")
    print("=" * 60)

    # 1. Run simulation
    print("\n[1/3] Running simulation...")
    road_network, frames, seg_history, start_nodes = run_and_record()

    # 2. Pre-compute road lines and bounds
    print("[2/3] Pre-computing road geometry...")
    road_lines, road_types = build_road_lines(road_network)

    # Compute bounds from all nodes
    lats = [n.lat for n in road_network.nodes.values()]
    lngs = [n.lng for n in road_network.nodes.values()]
    bounds = (min(lngs), max(lngs), min(lats), max(lats))

    # Zoom to area of interest (vehicle trajectory bounding box + padding)
    all_vlats, all_vlngs = [], []
    for f in frames:
        for vd in f["vehicles"].values():
            if vd["lat"] is not None:
                all_vlats.append(vd["lat"])
                all_vlngs.append(vd["lng"])
    if all_vlats:
        pad = 0.003
        bounds = (
            min(all_vlngs) - pad, max(all_vlngs) + pad,
            min(all_vlats) - pad, max(all_vlats) + pad,
        )

    # 3. Render frames
    print(f"[3/3] Rendering {len(frames)} frames...")
    images = []

    # Precompute oriented polylines from actual traversed segments
    oriented_chains = {}
    for vid, segs in seg_history.items():
        oriented_chains[vid] = orient_segment_chain(
            road_network, segs, start_nodes[vid])

    # Build segment index lookup: seg_id → index in seg_history
    seg_index = {}
    for vid, segs in seg_history.items():
        seg_index[vid] = {sid: idx for idx, sid in enumerate(segs)}

    for i, frame in enumerate(frames):
        # Build trajectory: draw all segments BEFORE the current one
        trajectory = {}
        for vid in seg_history:
            chain = oriented_chains[vid]
            vd = frame["vehicles"].get(vid, {})
            cur_seg = vd.get("segment", "")

            # Find how many completed segments to draw
            idx_map = seg_index[vid]
            if vd.get("arrived"):
                draw_up_to = len(chain)  # arrived — draw all
            elif cur_seg in idx_map:
                draw_up_to = idx_map[cur_seg]  # segments before current
            else:
                draw_up_to = len(chain)  # at node between segments

            points = []
            for j in range(draw_up_to):
                for pt in chain[j]:
                    pos = (pt[0], pt[1])
                    if not points or (
                            abs(points[-1][0] - pos[0]) > 1e-10 or
                            abs(points[-1][1] - pos[1]) > 1e-10):
                        points.append(pos)

            # Connect trajectory to car's actual position
            if vd.get("lat") is not None and not vd.get("arrived"):
                car_pos = (vd["lat"], vd["lng"])
                if not points or (
                        abs(points[-1][0] - car_pos[0]) > 1e-7 or
                        abs(points[-1][1] - car_pos[1]) > 1e-7):
                    points.append(car_pos)

            trajectory[vid] = points

        img = render_frame(road_network, road_lines, road_types,
                           frame, trajectory, bounds)
        images.append(img)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Frame {i + 1}/{len(frames)}")

    # 4. Save GIF
    output_path = os.path.join(os.path.dirname(__file__),
                               f"simulation_{scenario_name}.gif")
    print(f"\nSaving GIF to {output_path}...")
    durations = []
    for i, frame in enumerate(frames):
        if i == 0 or i == len(frames) - 1:
            durations.append(2000)
        else:
            durations.append(600)
    # Use PIL for reliable per-frame duration (in milliseconds)
    from PIL import Image
    pil_frames = [Image.fromarray(img) for img in images]
    pil_frames[0].save(
        output_path, save_all=True, append_images=pil_frames[1:],
        duration=durations, loop=0, optimize=False,
    )

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"Done! {len(images)} frames, {file_size_kb:.0f} KB")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
