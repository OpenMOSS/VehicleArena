"""
OSM Import — Convert OpenStreetMap data to RoadNetwork.

Uses osmnx to download and process OSM road networks, then converts
them to VehicleArena's RoadNetwork format.

Prerequisites:
    pip install osmnx>=1.9.0

Usage:
    from simulation.osm_import import import_from_osm, import_from_bbox

    # By place name
    net = import_from_osm("Zhongguancun, Haidian, Beijing, China")

    # By bounding box (north, south, east, west)
    net = import_from_bbox(39.99, 39.97, 116.33, 116.30)

    # Save for reuse
    net.save_json("road_networks/beijing_zhongguancun.json")
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# OSM highway tag → VehicleArena road_type mapping
_HIGHWAY_MAP = {
    "motorway": "motorway",
    "motorway_link": "motorway",
    "trunk": "trunk",
    "trunk_link": "trunk",
    "primary": "primary",
    "primary_link": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
    "residential": "residential",
    "living_street": "residential",
    "unclassified": "urban",
    "service": "service",
}

# Default speed limits by road type (km/h)
_DEFAULT_SPEEDS = {
    "motorway": 120,
    "trunk": 80,
    "primary": 60,
    "secondary": 50,
    "tertiary": 40,
    "residential": 30,
    "urban": 50,
    "service": 20,
}

# Default lane counts by road type
_DEFAULT_LANES = {
    "motorway": 6,
    "trunk": 4,
    "primary": 4,
    "secondary": 4,
    "tertiary": 2,
    "residential": 2,
    "urban": 2,
    "service": 1,
}


def _first_tag(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _parse_int_tag(value) -> Optional[int]:
    value = _first_tag(value)
    if value is None or value == "":
        return None
    match = re.search(r"-?\d+", str(value))
    return int(match.group()) if match else None


def _parse_float_tag(value) -> Optional[float]:
    value = _first_tag(value)
    if value is None or value == "":
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def _parse_bool_tag(value) -> bool:
    value = str(_first_tag(value) or "").strip().lower()
    return value in {"1", "true", "yes"}


def _parse_speed_kmh(value, fallback: int) -> int:
    """Parse the common OSM maxspeed forms without treating mph as km/h."""
    value = _first_tag(value)
    if value is None:
        return fallback
    text = str(value).strip().lower()
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return fallback
    speed = float(match.group())
    if "mph" in text:
        speed *= 1.609344
    return max(1, int(round(speed)))


def _parse_turn_lanes(value) -> list[str]:
    value = _first_tag(value)
    if not value:
        return []
    return [item.strip() for item in str(value).split("|")]


def import_from_osm(
    place: str,
    network_type: str = "drive",
    simplify: bool = True,
    max_nodes: int = 500,
) -> "RoadNetwork":
    """Import a road network by place name via OSM Nominatim geocoding.

    Args:
        place: Place name string (e.g. "Zhongguancun, Beijing, China").
        network_type: "drive" (default), "walk", "bike", or "all".
        simplify: If True, merge intermediate nodes on straight roads.
        max_nodes: If the network exceeds this many nodes, sample a subgraph
                   around the centroid. Set to 0 for no limit.

    Returns:
        RoadNetwork instance.
    """
    import osmnx as ox

    logger.info(f"Downloading OSM network for '{place}' ...")
    G = ox.graph_from_place(place, network_type=network_type, simplify=simplify)

    if max_nodes > 0 and len(G.nodes) > max_nodes:
        G = _truncate_graph(G, max_nodes)

    return _convert_graph(G)


def import_from_bbox(
    north: float,
    south: float,
    east: float,
    west: float,
    network_type: str = "drive",
    simplify: bool = True,
    max_nodes: int = 500,
) -> "RoadNetwork":
    """Import a road network by bounding box.

    Args:
        north, south, east, west: WGS84 bounding box coordinates.
        network_type: "drive" (default), "walk", "bike", or "all".
        simplify: If True, merge intermediate nodes.
        max_nodes: Node limit (0 = no limit).

    Returns:
        RoadNetwork instance.
    """
    import osmnx as ox

    logger.info(f"Downloading OSM network for bbox "
                f"({north:.4f},{south:.4f},{east:.4f},{west:.4f}) ...")
    G = ox.graph_from_bbox(north, south, east, west,
                           network_type=network_type, simplify=simplify)

    if max_nodes > 0 and len(G.nodes) > max_nodes:
        G = _truncate_graph(G, max_nodes)

    return _convert_graph(G)


def import_from_point(
    lat: float,
    lng: float,
    dist_meters: int = 1500,
    network_type: str = "drive",
    simplify: bool = True,
    max_nodes: int = 500,
) -> "RoadNetwork":
    """Import a road network centered on a point.

    Args:
        lat, lng: Center point coordinates.
        dist_meters: Radius in meters around center.
        network_type: "drive" (default), "walk", "bike", or "all".
        simplify: If True, merge intermediate nodes.
        max_nodes: Node limit (0 = no limit).

    Returns:
        RoadNetwork instance.
    """
    import osmnx as ox

    logger.info(f"Downloading OSM network around ({lat:.4f},{lng:.4f}) "
                f"r={dist_meters}m ...")
    G = ox.graph_from_point((lat, lng), dist=dist_meters,
                            network_type=network_type, simplify=simplify)

    if max_nodes > 0 and len(G.nodes) > max_nodes:
        G = _truncate_graph(G, max_nodes)

    return _convert_graph(G)


# ── Internal conversion ──────────────────────────────────────

def _truncate_graph(G, max_nodes: int):
    """Truncate graph to max_nodes by keeping the most central subgraph."""
    import osmnx as ox
    import networkx as nx

    logger.info(f"Truncating graph from {len(G.nodes)} to ~{max_nodes} nodes")

    # Find centroid
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    centroid_lat = nodes_gdf["y"].mean()
    centroid_lng = nodes_gdf["x"].mean()

    # Find the nearest node to centroid
    center_node = ox.nearest_nodes(G, centroid_lng, centroid_lat)

    # BFS from center to collect max_nodes nodes
    visited = set()
    queue = [center_node]
    while queue and len(visited) < max_nodes:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                queue.append(neighbor)

    return G.subgraph(visited).copy()


def _convert_graph(G) -> "RoadNetwork":
    """Convert an osmnx MultiDiGraph to RoadNetwork."""
    from simulation.road_network import RoadNetwork, RoadNode, RoadSegment

    net = RoadNetwork()

    # Convert nodes
    for osm_id, data in G.nodes(data=True):
        node_id = f"n{osm_id}"
        street_name = data.get("street_name", "")
        if not street_name:
            # Try to infer from incident edges
            street_name = _infer_street_name(G, osm_id)

        net.add_node(RoadNode(
            id=node_id,
            lat=data.get("y", 0.0),
            lng=data.get("x", 0.0),
            name=data.get("name", ""),
            street_name=street_name,
            signal="traffic_signals" in str(data.get("highway", "")),
            osm_id=osm_id,
        ))

    # Convert edges (aggregate parallel edges between same node pair)
    pair_records = defaultdict(list)
    for edge_u, edge_v, edge_key, edge_data in G.edges(
            keys=True, data=True):
        pair_records[(min(edge_u, edge_v), max(edge_u, edge_v))].append(
            (edge_u, edge_v, edge_key, edge_data))
    pair_counts = Counter({
        pair: len(records) for pair, records in pair_records.items()})
    seen_pairs = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        pair = (min(u, v), max(u, v))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        from_id = f"n{u}"
        to_id = f"n{v}"
        edge_id = RoadNetwork.make_edge_id(from_id, to_id)

        # Extract attributes
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0]
        road_type = _HIGHWAY_MAP.get(highway, "urban")

        # Speed limit
        speed_limit = _parse_speed_kmh(
            data.get("maxspeed"),
            _DEFAULT_SPEEDS.get(road_type, 50))

        # Lane count
        lanes = (
            _parse_int_tag(data.get("lanes"))
            or _DEFAULT_LANES.get(road_type, 2))
        lanes_forward = _parse_int_tag(data.get("lanes:forward"))
        lanes_backward = _parse_int_tag(data.get("lanes:backward"))

        # Distance
        length = data.get("length", 500.0)

        # Oneway
        raw_oneway = str(_first_tag(
            data.get("oneway", False))).strip().lower()
        oneway_reversed = raw_oneway == "-1"
        oneway = raw_oneway in {"1", "true", "yes", "-1"}

        # Geometry
        geometry = []
        if "geometry" in data:
            try:
                coords = list(data["geometry"].coords)
                geometry = [(lat, lng) for lng, lat in coords]
            except Exception:
                pass
        if oneway_reversed:
            from_id, to_id = to_id, from_id
            geometry.reverse()

        # When OSM represents two separated one-way carriageways with the
        # same endpoint pair, the compact topology source cannot retain
        # them as parallel graph edges. Preserve both travel directions and
        # their directional lane counts in one semantic road corridor rather
        # than silently dropping whichever directed edge appeared second.
        directed_records = []
        for record_u, record_v, _record_key, record_data in pair_records[pair]:
            record_oneway = str(_first_tag(
                record_data.get("oneway", False))).strip().lower()
            if record_oneway not in {"1", "true", "yes", "-1"}:
                continue
            if record_oneway == "-1":
                record_u, record_v = record_v, record_u
            record_lanes = (
                _parse_int_tag(record_data.get("lanes"))
                or _DEFAULT_LANES.get(
                    _HIGHWAY_MAP.get(
                        _first_tag(record_data.get(
                            "highway", "unclassified")), "urban"), 2))
            directed_records.append((
                f"n{record_u}", f"n{record_v}",
                max(1, record_lanes), record_data))
        directed_pairs = {
            (record_from, record_to)
            for record_from, record_to, _, _ in directed_records}
        if len(directed_pairs) > 1:
            forward_records = [
                record for record in directed_records
                if record[0] == from_id and record[1] == to_id]
            backward_records = [
                record for record in directed_records
                if record[0] == to_id and record[1] == from_id]
            if forward_records and backward_records:
                lanes_forward = sum(record[2] for record in forward_records)
                lanes_backward = sum(record[2] for record in backward_records)
                lanes = lanes_forward + lanes_backward
                oneway = False
                logger.warning(
                    "Collapsed %d parallel directed OSM edges for %s into "
                    "one bidirectional corridor with directional lanes",
                    len(directed_records), edge_id)
        elif len(directed_records) > 1 and oneway:
            same_direction = [
                record for record in directed_records
                if record[0] == from_id and record[1] == to_id]
            if len(same_direction) == len(directed_records):
                lanes = sum(record[2] for record in directed_records)
                lanes_forward = lanes

        # Street name
        name = data.get("name", "")
        if isinstance(name, list):
            name = name[0] if name else ""
        turn_lanes_forward = _parse_turn_lanes(
            data.get("turn:lanes:forward")
            or data.get("turn:lanes"))
        turn_lanes_backward = _parse_turn_lanes(
            data.get("turn:lanes:backward"))
        if len(directed_pairs) > 1:
            forward_data = next((
                record[3] for record in directed_records
                if record[0] == from_id and record[1] == to_id
            ), {})
            backward_data = next((
                record[3] for record in directed_records
                if record[0] == to_id and record[1] == from_id
            ), {})
            turn_lanes_forward = _parse_turn_lanes(
                forward_data.get("turn:lanes")
                or forward_data.get("turn:lanes:forward"))
            turn_lanes_backward = _parse_turn_lanes(
                backward_data.get("turn:lanes")
                or backward_data.get("turn:lanes:forward"))

        net.add_segment(RoadSegment(
            id=edge_id,
            from_node=from_id,
            to_node=to_id,
            name=name,
            road_type=road_type,
            lanes=lanes,
            lanes_forward=lanes_forward,
            lanes_backward=lanes_backward,
            turn_lanes_forward=turn_lanes_forward,
            turn_lanes_backward=turn_lanes_backward,
            lane_width_meters=(
                _parse_float_tag(data.get("width:lanes"))
                or (
                    (_parse_float_tag(data.get("width")) or 0.0)
                    / max(1, lanes)
                )
                or 3.5),
            speed_limit=speed_limit,
            distance_meters=float(length),
            oneway=oneway,
            geometry=geometry,
            osm_way_id=data.get("osmid", 0),
            layer=_parse_int_tag(data.get("layer")) or 0,
            bridge=_parse_bool_tag(data.get("bridge")),
            tunnel=_parse_bool_tag(data.get("tunnel")),
            source_edge_count=pair_counts[pair],
        ))

    logger.info(f"Imported RoadNetwork: {net.node_count} nodes, "
                f"{net.edge_count} segments")
    return net


def _infer_street_name(G, node_id) -> str:
    """Try to get a street name from edges incident to a node."""
    for _, _, data in G.edges(node_id, data=True):
        name = data.get("name", "")
        if name:
            return name[0] if isinstance(name, list) else name
    return ""
