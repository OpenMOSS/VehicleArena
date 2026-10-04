"""Compile VehicleArena lane maps into deterministic SUMO networks.

The lane JSON remains the authoring source of truth.  SUMO ``.net.xml`` is a
runtime cache produced through ``netconvert`` and is never edited by hand.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SUMO_MAP_FORMAT = "vehiclearena-sumo-map-v1"
SUMO_MAP_COMPILER_REVISION = "16"


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points, points[1:])
    )


def _shape(points: Sequence[Sequence[float]]) -> str:
    return " ".join(f"{float(x):.3f},{float(y):.3f}" for x, y in points)


def _parse_shape(value: str) -> List[List[float]]:
    return [list(map(float, point.split(",")))[:2] for point in value.split()]


def _project_offset(points: Sequence[Sequence[float]], point: Sequence[float]) -> float:
    """Closest along-polyline offset, including on bent approach lanes."""
    best = (float("inf"), 0.0)
    offset = 0.0
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        fraction = max(0.0, min(1.0, (
            (point[0] - a[0]) * dx + (point[1] - a[1]) * dy
        ) / (length * length))) if length else 0.0
        distance = math.hypot(
            point[0] - a[0] - fraction * dx,
            point[1] - a[1] - fraction * dy)
        best = min(best, (distance, offset + fraction * length))
        offset += length
    return best[1]


def _polyline_part(points, start: float, end: float) -> List[List[float]]:
    """Slice by geometric distance without replacing bends by a chord."""
    result = []
    offset = 0.0
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        if length and offset <= end and offset + length >= start:
            for position in (max(start, offset), min(end, offset + length)):
                fraction = (position - offset) / length
                point = [a[i] + fraction * (b[i] - a[i]) for i in (0, 1)]
                if not result or math.dist(result[-1], point) > 1e-8:
                    result.append(point)
        offset += length
    return result


def _clip_lane_extensions(authored, compiled):
    """Remove netconvert extensions beyond either end of an authored lane.

    A junction border can extend a lane past its connector mouth. Connecting
    that endpoint back to the authored mouth introduces a reversal, which
    flips SUMO's lateral offset. Bound the external lane first; projecting
    both ends onto the entire connector would also cut legitimate U-turns.
    """
    points = []
    for point in authored:
        if not points or math.dist(points[-1], point) > 1e-8:
            points.append(list(point))

    def extends(endpoint, origin, tangent, sign):
        length = math.hypot(*tangent)
        offset = sum((endpoint[i] - origin[i]) * tangent[i]
                     for i in (0, 1)) / length
        # Smaller endpoint differences are already merged by the connector
        # join's 2 cm tolerance. Do not rewrite lanes for XML rounding alone.
        return (math.dist(endpoint, origin) >= 0.02 - 1e-9
                and sign * offset > 1e-8)

    start, end = compiled[0], compiled[-1]
    clip_start = extends(start, points[0],
                         [points[1][i] - points[0][i] for i in (0, 1)], -1)
    clip_end = extends(end, points[-1],
                       [points[-1][i] - points[-2][i] for i in (0, 1)], 1)
    if not (clip_start or clip_end):
        return None
    start = points[0] if clip_start else start
    end = points[-1] if clip_end else end
    lo, hi = _project_offset(points, start), _project_offset(points, end)
    if hi - lo <= 0.001:
        # Very short lanes may have been moved wholly outside their source
        # interval by the junction border. Keep the original nonzero lane.
        return points
    result = _polyline_part(points, lo, hi)
    result[0], result[-1] = list(start), list(end)
    return result


def pedestrian_node_pair_key(start_node: str, end_node: str) -> str:
    """Stable JSON-manifest key for one directed pedestrian road leg."""
    return f"{start_node}\x1f{end_node}"


def _mean_shape(lanes: Sequence[Mapping[str, Any]]) -> List[List[float]]:
    """Return a carriageway centreline without inventing new topology."""
    lines = [lane["centerline_xy"] for lane in lanes]
    point_counts = {len(line) for line in lines}
    if len(point_counts) == 1:
        return [
            [
                sum(float(line[index][0]) for line in lines) / len(lines),
                sum(float(line[index][1]) for line in lines) / len(lines),
            ]
            for index in range(len(lines[0]))
        ]
    # Generated lanes normally share vertices.  A surveyed map may not; its
    # median lane is a safer geometric reference than resampling corners.
    return [list(point) for point in lines[len(lines) // 2]]


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


@dataclass(frozen=True)
class SumoMapBundle:
    """Compiled network plus stable VehicleArena-to-SUMO identifiers."""

    source_path: str
    cache_dir: str
    net_file: str
    manifest_file: str
    source_sha256: str
    edge_by_lane: Dict[str, str]
    lane_index_by_lane: Dict[str, int]
    lane_by_edge_index: Dict[str, Dict[int, str]]
    connector_via_lane: Dict[str, str]
    connector_link_index: Dict[str, int]
    connector_tls: Dict[str, str]
    tls_link_count: Dict[str, int]
    sidewalk_lane_by_edge: Dict[str, str]
    pedestrian_edge_by_node_pair: Dict[str, str]
    pedestrian_approach_edge_by_id: Dict[str, str]
    crosswalk_edge_by_id: Dict[str, str]
    crosswalk_link_index: Dict[str, int]
    crosswalk_tls: Dict[str, str]

    @classmethod
    def load(cls, manifest_file: str | os.PathLike[str]) -> "SumoMapBundle":
        path = Path(manifest_file)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format") != SUMO_MAP_FORMAT:
            raise ValueError(f"unsupported SUMO map manifest: {path}")
        return cls(
            source_path=raw["source_path"],
            cache_dir=str(path.parent),
            net_file=str(path.parent / raw["net_file"]),
            manifest_file=str(path),
            source_sha256=raw["source_sha256"],
            edge_by_lane=dict(raw["edge_by_lane"]),
            lane_index_by_lane={
                key: int(value)
                for key, value in raw["lane_index_by_lane"].items()
            },
            lane_by_edge_index={
                edge: {int(index): lane_id for index, lane_id in values.items()}
                for edge, values in raw["lane_by_edge_index"].items()
            },
            connector_via_lane=dict(raw["connector_via_lane"]),
            connector_link_index={
                key: int(value)
                for key, value in raw["connector_link_index"].items()
            },
            connector_tls=dict(raw["connector_tls"]),
            tls_link_count={
                key: int(value)
                for key, value in raw["tls_link_count"].items()
            },
            sidewalk_lane_by_edge=dict(raw.get(
                "sidewalk_lane_by_edge", {})),
            pedestrian_edge_by_node_pair=dict(raw.get(
                "pedestrian_edge_by_node_pair", {})),
            pedestrian_approach_edge_by_id=dict(raw.get(
                "pedestrian_approach_edge_by_id", {})),
            crosswalk_edge_by_id=dict(raw.get(
                "crosswalk_edge_by_id", {})),
            crosswalk_link_index={
                key: int(value)
                for key, value in raw.get(
                    "crosswalk_link_index", {}).items()
            },
            crosswalk_tls=dict(raw.get("crosswalk_tls", {})),
        )


class SumoMapConverter:
    """Build a SUMO cache from one ``vehiclearena-lane-level-v0.3`` map."""

    def __init__(self, *, netconvert_binary: str = "netconvert"):
        binary = shutil.which(netconvert_binary)
        if not binary:
            raise RuntimeError(
                "netconvert is unavailable; install SUMO before selecting "
                "the SUMO physics engine")
        self.netconvert_binary = binary

    def convert(
        self,
        source_path: str | os.PathLike[str],
        *,
        cache_root: str | os.PathLike[str] | None = None,
        force: bool = False,
    ) -> SumoMapBundle:
        source = Path(source_path).resolve()
        source_bytes = source.read_bytes()
        digest = hashlib.sha256(
            source_bytes + SUMO_MAP_FORMAT.encode("ascii")
            + SUMO_MAP_COMPILER_REVISION.encode("ascii")
        ).hexdigest()
        root = Path(
            cache_root
            or os.environ.get("VEHICLEARENA_SUMO_CACHE", "")
            or (Path(tempfile.gettempdir()) / "vehiclearena-sumo")
        )
        cache_dir = root / f"{source.stem}-{digest[:12]}"
        manifest_path = cache_dir / "manifest.json"
        if manifest_path.exists() and not force:
            bundle = SumoMapBundle.load(manifest_path)
            if Path(bundle.net_file).exists() and bundle.source_sha256 == digest:
                self._validate_network(
                    Path(bundle.net_file),
                    json.loads(manifest_path.read_text(encoding="utf-8")))
                return bundle

        raw = json.loads(source_bytes)
        if raw.get("schema") != "vehiclearena-lane-level-v0.3":
            raise ValueError(
                f"{source} is not a vehiclearena-lane-level-v0.3 map")
        staging = Path(tempfile.mkdtemp(
            prefix=f".{source.stem}-", dir=str(root.mkdir(
                parents=True, exist_ok=True) or root)
        ))
        try:
            manifest = self._write_plain_xml(raw, source, digest, staging)
            net_file = staging / "network.net.xml"
            command = [
                self.netconvert_binary,
                "--node-files", str(staging / "nodes.nod.xml"),
                "--edge-files", str(staging / "edges.edg.xml"),
                "--connection-files", str(staging / "connections.con.xml"),
                "--output-file", str(net_file),
                "--no-turnarounds", "true",
                "--junctions.corner-detail", "5",
                "--junctions.internal-link-detail", "12",
                "--geometry.remove", "false",
                "--plain.extend-edge-shape", "false",
                # Keep short approaches; custom connector clipping is repaired
                # deterministically after the final topology/TLS compilation.
                "--junctions.endpoint-shape", "true",
                "--offset.disable-normalization", "true",
            ]
            completed = self._run_netconvert(
                command, net_file, source, "initial")
            # First let netconvert discover every controlled link, including
            # implicit links in composite junctions. Compile VehicleArena's
            # phases against those final indexes, then rebuild
            # deterministically.
            actual_links, actual_tls, tls_link_count = (
                self._resolve_controlled_links(
                    raw, net_file, manifest)
            )
            actual_via_lanes = self._resolve_connector_via_lanes(
                raw, net_file, manifest)
            pedestrian_topology = self._resolve_pedestrian_topology(
                raw, net_file, manifest)
            manifest["connector_link_index"] = actual_links
            manifest["connector_tls"] = actual_tls
            manifest["tls_link_count"] = tls_link_count
            manifest["connector_via_lane"] = actual_via_lanes
            manifest.update(pedestrian_topology)
            self._write_signal_programs(
                raw, actual_links, actual_tls, tls_link_count,
                pedestrian_topology["crosswalk_link_index"],
                pedestrian_topology["crosswalk_tls"],
                staging / "signals.tll.xml")
            command_with_signals = command + [
                "--tllogic-files", str(staging / "signals.tll.xml")]
            completed = self._run_netconvert(
                command_with_signals, net_file, source,
                "signal-link rebuild")
            verified_links, verified_tls, verified_link_count = (
                self._resolve_controlled_links(raw, net_file, manifest))
            verified_via_lanes = self._resolve_connector_via_lanes(
                raw, net_file, manifest)
            verified_pedestrian_topology = self._resolve_pedestrian_topology(
                raw, net_file, manifest)
            if (verified_links != actual_links
                    or verified_tls != actual_tls
                    or verified_link_count != tls_link_count
                    or verified_via_lanes != actual_via_lanes
                    or verified_pedestrian_topology != pedestrian_topology):
                raise RuntimeError(
                    "SUMO connection mapping was not deterministic")
            manifest["netconvert_stdout"] = completed.stdout.strip()
            manifest["netconvert_stderr"] = completed.stderr.strip()
            self._restore_connector_geometry(raw, net_file, manifest)
            manifest_path_staging = staging / "manifest.json"
            manifest_path_staging.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._validate_network(net_file, manifest)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            staging.rename(cache_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return SumoMapBundle.load(manifest_path)

    @staticmethod
    def _connector_chains(root: ET.Element, manifest: Mapping[str, Any]) -> dict:
        """Resolve entire via chains, not just the first internal lane.

        netconvert may split a turn at an internal pedestrian/vehicle yield
        point. Its successor has the same external target and an extra via.
        """
        links = {}
        external = {}
        for connection in root.findall("connection"):
            source = f"{connection.get('from')}_{connection.get('fromLane')}"
            target = f"{connection.get('to')}_{connection.get('toLane')}"
            links[(source, target)] = connection.get("via") or target
            if not connection.get("from", "").startswith(":"):
                external[connection.get("via")] = (source, target)
        chains = {}
        for connector_id, first in manifest["connector_via_lane"].items():
            if first not in external:
                raise RuntimeError(f"missing connector entry: {connector_id}")
            source, target = external[first]
            chain = []
            current = first
            while current != target:
                if current in chain or not current.startswith(":"):
                    raise RuntimeError(f"invalid connector chain: {connector_id}")
                chain.append(current)
                successor = links.get((current, target))
                if successor is None:
                    raise RuntimeError(f"incomplete connector chain: {connector_id}")
                current = successor
            chains[connector_id] = (source, chain, target)
        return chains

    @classmethod
    def _restore_connector_geometry(cls, raw, net_file: Path, manifest: dict) -> None:
        """Restore authored curves while retaining SUMO topology and yield nodes.

        SUMO clips custom connections against carriageway-wide node borders.
        A bend may cross that border twice, losing half its geometry. Never
        repair this by drawing a straight line from the truncated end. Restore
        the full authored curve, including approach pieces trimmed by SUMO,
        and retain each internal split at its projected geometric location.
        Request matrices, signal links, permissions and speed caps are untouched;
        SUMO computes conflict distances from these final shapes when loading.
        """
        root = ET.parse(net_file).getroot()
        lanes = {lane.get("id"): lane for lane in root.findall(".//lane")}
        authored_lanes = {lane["id"]: lane for lane in raw["lanes"]}
        chains = cls._connector_chains(root, manifest)
        external_lanes = {}
        for connector in raw["connectors"]:
            source, _, target = chains[connector["id"]]
            external_lanes[source] = connector["from_lane"]
            external_lanes[target] = connector["to_lane"]
        bounded_lanes = []
        for lane_id, authored_id in external_lanes.items():
            lane = lanes[lane_id]
            bounded = _clip_lane_extensions(
                authored_lanes[authored_id]["centerline_xy"],
                _parse_shape(lane.get("shape")))
            if bounded is not None:
                lane.set("shape", _shape(bounded))
                lane.set("length", f"{_polyline_length(bounded):.3f}")
                bounded_lanes.append(lane_id)
        manifest["bounded_vehicle_lanes"] = sorted(bounded_lanes)
        # SUMO carries the same longitudinal position across lanes on an edge.
        # A clipped lane therefore cannot get its own driving length: doing so
        # moves a vehicle along the road when it changes lanes, even at rest.
        for edge_id, lane_indexes in manifest.get("lane_by_edge_index", {}).items():
            vehicle_lanes = [lanes[f"{edge_id}_{index}"] for index in lane_indexes]
            if len(vehicle_lanes) < 2:
                continue
            if len({lane.get("length") for lane in vehicle_lanes}) == 1:
                continue
            shared_length = sum(
                _polyline_length(_parse_shape(lane.get("shape")))
                for lane in vehicle_lanes
            ) / len(vehicle_lanes)
            for lane in vehicle_lanes:
                lane.set("length", f"{shared_length:.3f}")
        assignments = {}
        geometry = {}
        for connector in raw["connectors"]:
            cid = connector["id"]
            source, chain, target = chains[cid]
            before = authored_lanes[connector["from_lane"]]["centerline_xy"]
            after = authored_lanes[connector["to_lane"]]["centerline_xy"]
            start = _parse_shape(lanes[source].get("shape"))[-1]
            end = _parse_shape(lanes[target].get("shape"))[0]
            pieces = (
                [start],
                _polyline_part(before, _project_offset(before, start),
                               _polyline_length(before)),
                connector["centerline_xy"],
                _polyline_part(after, 0.0, _project_offset(after, end)),
                [end],
            )
            full = []
            for piece in pieces:
                for point in piece:
                    if not full or math.dist(full[-1], point) > 0.02:
                        full.append(list(point))
            # Exact joins take precedence over centimetre rounding in XML.
            full[0], full[-1] = start, end
            length = _polyline_length(full)
            boundaries = [0.0] + [
                _project_offset(full, _parse_shape(lanes[lane_id].get("shape"))[-1])
                for lane_id in chain[:-1]
            ] + [length]
            split_method = "projected"
            if any(b - a <= 0.001 for a, b in zip(boundaries, boundaries[1:])):
                # A severely clipped loop may leave its old split outside the
                # real curve. Preserve the ordered, nonzero via stages using
                # their relative lengths, rather than deleting a yield node
                # or collapsing it onto the entry/exit boundary.
                weights = [_polyline_length(_parse_shape(lanes[x].get("shape")))
                           for x in chain]
                if min(weights) <= 0 or length <= 0.001 * len(chain):
                    raise RuntimeError(f"invalid internal yield geometry: {cid}")
                boundaries = [0.0]
                for weight in weights:
                    boundaries.append(boundaries[-1] + length * weight / sum(weights))
                boundaries[-1] = length
                split_method = "relative_length"
            for lane_id, lo, hi in zip(chain, boundaries, boundaries[1:]):
                points = _polyline_part(full, lo, hi)
                value = (_shape(points), f"{_polyline_length(points):.3f}")
                if lane_id in assignments and assignments[lane_id] != value:
                    raise RuntimeError(f"conflicting shared internal lane: {lane_id}")
                assignments[lane_id] = value
            geometry[cid] = {"lanes": chain, "length_m": length,
                             "split_method": split_method}
        for lane_id, (shape, length) in assignments.items():
            lanes[lane_id].set("shape", shape)
            lanes[lane_id].set("length", length)
        # Internal junction coordinates identify an existing yield point.
        # Move that marker with the split; preserve its incLanes/intLanes.
        for junction in root.findall("junction"):
            if junction.get("type") == "internal" and junction.get("id") in assignments:
                point = _parse_shape(assignments[junction.get("id")][0])[0]
                junction.set("x", f"{point[0]:.3f}")
                junction.set("y", f"{point[1]:.3f}")
        manifest["connector_geometry"] = geometry
        _write_xml(root, net_file)

    @classmethod
    def _validate_connector_geometry(cls, root, manifest) -> None:
        lanes = {lane.get("id"): lane for lane in root.findall(".//lane")}
        for cid, (source, chain, target) in cls._connector_chains(root, manifest).items():
            ids = [source, *chain, target]
            for before, after in zip(ids, ids[1:]):
                gap = math.dist(_parse_shape(lanes[before].get("shape"))[-1],
                                _parse_shape(lanes[after].get("shape"))[0])
                if gap > 0.02:
                    raise RuntimeError(f"discontinuous connector {cid}: {before} -> {after}, gap={gap:.3f}m")
            length = 0.0
            for lane_id in chain:
                lane = lanes[lane_id]
                actual = _polyline_length(_parse_shape(lane.get("shape")))
                if actual <= 0 or abs(actual - float(lane.get("length"))) > 0.02:
                    raise RuntimeError(f"invalid connector length: {cid}/{lane_id}")
                length += actual
            expected = manifest.get("connector_geometry", {}).get(cid)
            if expected is None or abs(length - expected["length_m"]) > 0.05:
                raise RuntimeError(f"connector geometry lost authored path: {cid}")

    @staticmethod
    def _run_netconvert(
        command: Sequence[str], net_file: Path, source: Path, phase: str,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False)
        if completed.returncode != 0 or not net_file.exists():
            raise RuntimeError(
                f"netconvert {phase} failed for {source.name}: "
                f"{completed.stderr.strip()}")
        return completed

    @staticmethod
    def _resolve_controlled_links(
        raw: Mapping[str, Any], net_file: Path,
        manifest: Mapping[str, Any],
    ) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, int]]:
        """Map authored connector IDs to netconvert's final TLS indexes."""
        root = ET.parse(net_file).getroot()
        by_signature = {}
        for connection in root.findall("connection"):
            if (not connection.get("tl")
                    or connection.get("from", "").startswith(":")):
                continue
            signature = (
                connection.get("from"), connection.get("to"),
                int(connection.get("fromLane", "0")),
                int(connection.get("toLane", "0")),
            )
            by_signature[signature] = connection
        lane_by_id = {item["id"]: item for item in raw.get("lanes", [])}
        indexes = manifest["lane_index_by_lane"]
        edge_by_lane = manifest["edge_by_lane"]
        plans = {item["node_id"] for item in raw.get("signal_plans", [])}
        connector_links: Dict[str, int] = {}
        connector_tls: Dict[str, str] = {}
        for connector in raw.get("connectors", []):
            if (not connector.get("signal_controlled")
                    or connector.get("node_id") not in plans):
                continue
            source = lane_by_id[connector["from_lane"]]
            target = lane_by_id[connector["to_lane"]]
            signature = (
                edge_by_lane[source["id"]],
                edge_by_lane[target["id"]],
                int(indexes[source["id"]]),
                int(indexes[target["id"]]),
            )
            compiled = by_signature.get(signature)
            if compiled is None:
                raise RuntimeError(
                    "netconvert dropped controlled connector "
                    f"{connector['id']}")
            connector_links[connector["id"]] = int(
                compiled.get("linkIndex", "-1"))
            connector_tls[connector["id"]] = str(compiled.get("tl"))
        tls_link_count = {}
        for logic in root.findall("tlLogic"):
            phases = logic.findall("phase")
            if phases:
                tls_link_count[str(logic.get("id"))] = len(
                    phases[0].get("state", ""))
        missing_tls = set(connector_tls.values()) - set(tls_link_count)
        if missing_tls:
            raise RuntimeError(
                f"SUMO omitted traffic-light programs: {sorted(missing_tls)}")
        return connector_links, connector_tls, tls_link_count

    @staticmethod
    def _resolve_connector_via_lanes(
        raw: Mapping[str, Any], net_file: Path,
        manifest: Mapping[str, Any],
    ) -> Dict[str, str]:
        """Resolve every authored connector to its SUMO internal lane."""
        root = ET.parse(net_file).getroot()
        by_signature = {}
        for connection in root.findall("connection"):
            if connection.get("from", "").startswith(":"):
                continue
            signature = (
                connection.get("from"), connection.get("to"),
                int(connection.get("fromLane", "0")),
                int(connection.get("toLane", "0")),
            )
            by_signature[signature] = connection
        lane_by_id = {item["id"]: item for item in raw.get("lanes", [])}
        indexes = manifest["lane_index_by_lane"]
        edge_by_lane = manifest["edge_by_lane"]
        result: Dict[str, str] = {}
        for connector in raw.get("connectors", []):
            source = lane_by_id[connector["from_lane"]]
            target = lane_by_id[connector["to_lane"]]
            signature = (
                edge_by_lane[source["id"]], edge_by_lane[target["id"]],
                int(indexes[source["id"]]), int(indexes[target["id"]]),
            )
            compiled = by_signature.get(signature)
            via_lane = compiled.get("via", "") if compiled is not None else ""
            if not via_lane:
                raise RuntimeError(
                    "netconvert did not create an internal lane for connector "
                    f"{connector['id']}")
            result[connector["id"]] = via_lane
        return result

    @staticmethod
    def _resolve_pedestrian_topology(
        raw: Mapping[str, Any], net_file: Path,
        manifest: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Resolve generated sidewalks and authored crossings after netconvert."""
        root = ET.parse(net_file).getroot()
        sidewalk_lane_by_edge: Dict[str, str] = {}
        crossing_edges = []
        for edge in root.findall("edge"):
            if edge.get("function") == "crossing":
                lane = edge.find("lane")
                shape = [] if lane is None else [
                    tuple(float(value) for value in point.split(","))
                    for point in lane.get("shape", "").split()
                ]
                crossing_edges.append({
                    "id": str(edge.get("id")),
                    "crossing_edges": frozenset(
                        edge.get("crossingEdges", "").split()),
                    "shape": shape,
                })
                continue
            if edge.get("function") == "internal":
                continue
            for lane in edge.findall("lane"):
                allowed = set(lane.get("allow", "").split())
                disallowed = set(lane.get("disallow", "").split())
                if "pedestrian" in allowed and "pedestrian" not in disallowed:
                    sidewalk_lane_by_edge[str(edge.get("id"))] = str(
                        lane.get("id"))
                    break
        expected_edges = set(manifest["lane_by_edge_index"])
        sidewalk_lane_by_edge = {
            edge_id: lane_id
            for edge_id, lane_id in sidewalk_lane_by_edge.items()
            if edge_id in expected_edges
        }
        if set(sidewalk_lane_by_edge) != expected_edges:
            missing = sorted(expected_edges - set(sidewalk_lane_by_edge))
            raise RuntimeError(
                "SUMO conversion missing sidewalk lanes: "
                f"{missing[:5]}")

        connections_to_crossing: Dict[str, ET.Element] = {}
        for connection in root.findall("connection"):
            target = str(connection.get("to", ""))
            if target and connection.get("tl"):
                connections_to_crossing[target] = connection

        crosswalk_edge_by_id: Dict[str, str] = {}
        crosswalk_link_index: Dict[str, int] = {}
        crosswalk_tls: Dict[str, str] = {}
        input_edges = manifest.get("crosswalk_input_edges", {})
        for crosswalk in raw.get("crosswalks", []):
            crosswalk_id = str(crosswalk["id"])
            node_id = str(crosswalk["node_id"])
            expected = frozenset(input_edges.get(crosswalk_id, []))
            candidates = [
                item for item in crossing_edges
                if item["crossing_edges"] == expected
                and item["id"].startswith(f":{node_id}_c")
            ]
            if not candidates:
                raise RuntimeError(
                    "netconvert dropped authored pedestrian crossing "
                    f"{crosswalk_id} at {node_id}")
            center = crosswalk.get("center_xy", [0.0, 0.0])

            def distance_sq(item: Mapping[str, Any]) -> float:
                points = item["shape"]
                if not points:
                    return float("inf")
                midpoint = points[len(points) // 2]
                return ((midpoint[0] - float(center[0])) ** 2
                        + (midpoint[1] - float(center[1])) ** 2)

            compiled = min(candidates, key=distance_sq)
            crossing_edge_id = str(compiled["id"])
            crosswalk_edge_by_id[crosswalk_id] = crossing_edge_id
            controlled = connections_to_crossing.get(crossing_edge_id)
            if controlled is not None:
                crosswalk_link_index[crosswalk_id] = int(
                    controlled.get("linkIndex", "-1"))
                crosswalk_tls[crosswalk_id] = str(controlled.get("tl"))
        return {
            "sidewalk_lane_by_edge": sidewalk_lane_by_edge,
            "crosswalk_edge_by_id": crosswalk_edge_by_id,
            "crosswalk_link_index": crosswalk_link_index,
            "crosswalk_tls": crosswalk_tls,
        }

    @staticmethod
    def _write_signal_programs(
        raw: Mapping[str, Any], connector_links: Mapping[str, int],
        connector_tls: Mapping[str, str],
        tls_link_count: Mapping[str, int],
        crosswalk_links: Mapping[str, int],
        crosswalk_tls: Mapping[str, str],
        path: Path,
    ) -> None:
        """Write phases using the final SUMO link-index ordering."""
        plans = {
            item["node_id"]: item for item in raw.get("signal_plans", [])
        }
        by_tls: Dict[str, Dict[int, str]] = {}
        for connector_id, index in connector_links.items():
            tls = connector_tls[connector_id]
            by_tls.setdefault(tls, {})[int(index)] = connector_id
        crossing_indexes: Dict[str, set[int]] = {}
        for crosswalk_id, index in crosswalk_links.items():
            tls = crosswalk_tls[crosswalk_id]
            crossing_indexes.setdefault(tls, set()).add(int(index))
        root = ET.Element("tlLogics")
        for node_id, plan in sorted(plans.items()):
            indexed = by_tls.get(node_id, {})
            if not indexed:
                continue
            link_count = int(tls_link_count[node_id])
            connector_order = [indexed.get(index, "")
                               for index in range(link_count)]
            logic = ET.SubElement(root, "tlLogic", {
                "id": node_id, "type": "static",
                "programID": "vehiclearena", "offset": "0",
            })
            for phase in plan.get("phases", []):
                permitted = set(phase.get("connector_ids", []))
                green_s = float(phase.get("green_s", 0.0))
                yellow_s = float(phase.get("yellow_s", 0.0))
                all_red_s = float(phase.get("all_red_s", 0.0))
                if green_s > 0:
                    crossing_links = crossing_indexes.get(node_id, set())
                    ET.SubElement(logic, "phase", {
                        "duration": f"{green_s:.3f}",
                        "state": "".join(
                            ("G" if phase.get("pedestrian_green") else "r")
                            if index in crossing_links
                            else "g" if (connector_id
                                         and connector_id in permitted)
                            else "r"
                            for index, connector_id
                            in enumerate(connector_order)),
                        "name": str(phase.get("label", "green")),
                    })
                if yellow_s > 0:
                    crossing_links = crossing_indexes.get(node_id, set())
                    ET.SubElement(logic, "phase", {
                        "duration": f"{yellow_s:.3f}",
                        "state": "".join(
                            "r" if index in crossing_links
                            else "y" if (connector_id
                                         and connector_id in permitted)
                            else "r"
                            for index, connector_id
                            in enumerate(connector_order)),
                        "name": f"{phase.get('label', 'phase')}:yellow",
                    })
                if all_red_s > 0:
                    ET.SubElement(logic, "phase", {
                        "duration": f"{all_red_s:.3f}",
                        "state": "r" * link_count,
                        "name": f"{phase.get('label', 'phase')}:all_red",
                    })
        _write_xml(root, path)

    @staticmethod
    def _write_plain_xml(
        raw: Mapping[str, Any], source_path: Path, digest: str, output: Path,
    ) -> Dict[str, Any]:
        lanes = list(raw.get("lanes", []))
        lane_by_id = {lane["id"]: lane for lane in lanes}
        groups: Dict[Tuple[str, str], List[dict]] = {}
        for lane in lanes:
            groups.setdefault(
                (lane["segment_id"], lane["direction"]), []).append(lane)
        ordered_groups = sorted(groups.items())

        edge_for_group: Dict[Tuple[str, str], str] = {}
        edge_by_lane: Dict[str, str] = {}
        lane_index_by_lane: Dict[str, int] = {}
        lane_by_edge_index: Dict[str, Dict[int, str]] = {}
        pedestrian_edge_by_node_pair: Dict[str, str] = {}
        for ordinal, (group_key, group_lanes) in enumerate(ordered_groups):
            edge_id = f"va_e{ordinal:06d}"
            edge_for_group[group_key] = edge_id
            lane_by_edge_index[edge_id] = {}
            for lane in group_lanes:
                # SUMO lane 0 is the sidewalk. Vehicle lanes retain their
                # relative order at indexes 1..N.
                lane_index = 1 + int(lane.get(
                    "directional_index", lane.get("index", 0)))
                edge_by_lane[lane["id"]] = edge_id
                lane_index_by_lane[lane["id"]] = lane_index
                lane_by_edge_index[edge_id][lane_index] = lane["id"]
            representative = group_lanes[0]
            pedestrian_edge_by_node_pair[pedestrian_node_pair_key(
                str(representative["start_node"]),
                str(representative["end_node"]),
            )] = edge_id

        intersection_xy = {
            item["id"]: item["center_xy"]
            for item in raw.get("intersections", [])
        }
        node_xy = {
            item["id"]: item["xy"] for item in raw.get("nodes_xy", [])
        }
        endpoints: Dict[str, List[Sequence[float]]] = {}
        for lane in lanes:
            endpoints.setdefault(lane["start_junction"], []).append(
                lane["centerline_xy"][0])
            endpoints.setdefault(lane["end_junction"], []).append(
                lane["centerline_xy"][-1])
        signal_nodes = {
            item["node_id"] for item in raw.get("signal_plans", [])
        }
        ordered_crosswalks = sorted(
            raw.get("crosswalks", []), key=lambda item: item["id"])
        ordered_pedestrian_approaches = sorted(
            raw.get("pedestrian_approaches", []),
            key=lambda item: item["id"],
        )
        nodes_root = ET.Element("nodes")
        used_nodes = sorted({
            lane[side]
            for lane in lanes
            for side in ("start_junction", "end_junction")
        })
        for node_id in used_nodes:
            points = endpoints.get(node_id, [])
            xy = intersection_xy.get(node_id) or node_xy.get(node_id)
            if xy is None and points:
                xy = [
                    sum(float(point[0]) for point in points) / len(points),
                    sum(float(point[1]) for point in points) / len(points),
                ]
            if xy is None:
                raise ValueError(f"no metric coordinate for junction {node_id}")
            attrs = {
                "id": node_id,
                "x": f"{float(xy[0]):.3f}",
                "y": f"{float(xy[1]):.3f}",
                "type": (
                    "traffic_light" if node_id in signal_nodes
                    else "priority"),
            }
            ET.SubElement(nodes_root, "node", attrs)
        for crosswalk in ordered_crosswalks:
            line = crosswalk.get("centerline_xy", [])
            if len(line) < 2:
                raise ValueError(
                    f"crosswalk {crosswalk['id']} has no usable centerline")
        pedestrian_approach_edge_by_id: Dict[str, str] = {}
        for ordinal, approach in enumerate(ordered_pedestrian_approaches):
            approach_id = str(approach["id"])
            node_id = str(approach["node_id"])
            line = approach.get("centerline_xy", [])
            if node_id not in used_nodes:
                raise ValueError(
                    f"pedestrian approach {approach_id} references unknown "
                    f"junction {node_id}")
            if len(line) < 2:
                raise ValueError(
                    f"pedestrian approach {approach_id} has no usable "
                    "centerline")
            outer_node_id = f"va_pn{ordinal:06d}"
            ET.SubElement(nodes_root, "node", {
                "id": outer_node_id,
                "x": f"{float(line[-1][0]):.3f}",
                "y": f"{float(line[-1][1]):.3f}",
                "type": "dead_end",
            })
            pedestrian_approach_edge_by_id[approach_id] = (
                f"va_p{ordinal:06d}")
        _write_xml(nodes_root, output / "nodes.nod.xml")

        edges_root = ET.Element("edges")
        for group_key, group_lanes in ordered_groups:
            group_lanes.sort(key=lambda item: int(item.get(
                "directional_index", item.get("index", 0))))
            indexes = [int(item.get(
                "directional_index", item.get("index", 0)))
                for item in group_lanes]
            if indexes != list(range(len(group_lanes))):
                raise ValueError(
                    f"non-contiguous directional lane indexes for {group_key}")
            start_nodes = {item["start_junction"] for item in group_lanes}
            end_nodes = {item["end_junction"] for item in group_lanes}
            if len(start_nodes) != 1 or len(end_nodes) != 1:
                raise ValueError(f"inconsistent lane endpoints for {group_key}")
            group_speed_mps = min(
                float(item.get("speed_limit_kmh", 50.0))
                for item in group_lanes) / 3.6
            group_width_m = sum(
                float(item.get("width_m", 3.5))
                for item in group_lanes) / len(group_lanes)
            edge = ET.SubElement(edges_root, "edge", {
                "id": edge_for_group[group_key],
                "from": next(iter(start_nodes)),
                "to": next(iter(end_nodes)),
                "numLanes": str(len(group_lanes) + 1),
                "speed": f"{group_speed_mps:.6f}",
                "width": f"{group_width_m:.3f}",
                "shape": _shape(_mean_shape(group_lanes)),
                "spreadType": "center",
                "allow": "passenger bus truck delivery emergency",
            })
            ET.SubElement(edge, "lane", {
                "index": "0",
                "speed": "2.500000",
                "width": "2.500",
                "allow": "pedestrian",
            })
            for lane in group_lanes:
                ET.SubElement(edge, "lane", {
                    "index": str(lane_index_by_lane[lane["id"]]),
                    "speed": f"{float(lane.get('speed_limit_kmh', 50.0)) / 3.6:.6f}",
                    "width": f"{float(lane.get('width_m', 3.5)):.3f}",
                    "allow": "passenger bus truck delivery emergency",
                    # Preserve the authored high-definition lane centreline.
                    # Without a lane-level shape netconvert recreates lanes
                    # from one averaged edge line, producing a second map
                    # geometry for physics and a visible pose jump at t=0.
                    "shape": _shape(lane["centerline_xy"]),
                })
        for ordinal, approach in enumerate(ordered_pedestrian_approaches):
            approach_id = str(approach["id"])
            edge = ET.SubElement(edges_root, "edge", {
                "id": pedestrian_approach_edge_by_id[approach_id],
                "from": str(approach["node_id"]),
                "to": f"va_pn{ordinal:06d}",
                "numLanes": "1",
                "speed": "2.500000",
                "width": f"{float(approach.get('width_m', 2.5)):.3f}",
                "shape": _shape(approach["centerline_xy"]),
                "spreadType": "center",
                "allow": "pedestrian",
            })
            ET.SubElement(edge, "lane", {
                "index": "0",
                "speed": "2.500000",
                "width": f"{float(approach.get('width_m', 2.5)):.3f}",
                "allow": "pedestrian",
            })
        _write_xml(edges_root, output / "edges.edg.xml")

        plans = {
            item["node_id"]: item for item in raw.get("signal_plans", [])
        }
        signal_connectors: Dict[str, List[dict]] = {}
        for connector in raw.get("connectors", []):
            if connector.get("signal_controlled") and connector["node_id"] in plans:
                signal_connectors.setdefault(
                    connector["node_id"], []).append(connector)
        for values in signal_connectors.values():
            values.sort(key=lambda item: item["id"])

        connections_root = ET.Element("connections")
        valid_connectors = 0
        for connector in sorted(
                raw.get("connectors", []), key=lambda item: item["id"]):
            source_lane = lane_by_id.get(connector["from_lane"])
            target = lane_by_id.get(connector["to_lane"])
            if source_lane is None or target is None:
                continue
            attrs = {
                "from": edge_by_lane[source_lane["id"]],
                "to": edge_by_lane[target["id"]],
                "fromLane": str(lane_index_by_lane[source_lane["id"]]),
                "toLane": str(lane_index_by_lane[target["id"]]),
                "dir": {
                    "left": "l", "right": "r", "straight": "s",
                    "uturn": "t",
                }.get(str(connector.get("turn", "straight")), "s"),
                "keepClear": "true",
                # The junction connector is already a surveyed/generated
                # first-class lane in VehicleArena. Let SUMO execute that
                # same curve instead of synthesising a different spline.
                "shape": _shape(connector["centerline_xy"]),
            }
            ET.SubElement(connections_root, "connection", attrs)
            valid_connectors += 1

        crosswalk_input_edges: Dict[str, List[str]] = {}
        for crosswalk in ordered_crosswalks:
            node_id = str(crosswalk["node_id"])
            segment_ids = {
                str(value) for value in crosswalk.get(
                    "crossed_road_segment_ids",
                    [crosswalk["road_segment_id"]],
                )
            }
            if not segment_ids:
                raise ValueError(
                    f"crosswalk {crosswalk['id']} has no crossed road "
                    "segments")
            crossed_edges = []
            resolved_segment_ids = set()
            for group_key, group_lanes in ordered_groups:
                if group_key[0] not in segment_ids:
                    continue
                representative = group_lanes[0]
                if node_id not in {
                        representative["start_junction"],
                        representative["end_junction"]}:
                    continue
                crossed_edges.append(edge_for_group[group_key])
                resolved_segment_ids.add(group_key[0])
            missing_segments = segment_ids - resolved_segment_ids
            if missing_segments:
                raise ValueError(
                    f"crosswalk {crosswalk['id']} has non-incident crossed "
                    f"road segments: {sorted(missing_segments)}")
            if not crossed_edges:
                raise ValueError(
                    f"crosswalk {crosswalk['id']} has no incident SUMO edge")
            crossed_edges.sort()
            crosswalk_input_edges[str(crosswalk["id"])] = crossed_edges
            ET.SubElement(connections_root, "crossing", {
                "node": node_id,
                "edges": " ".join(crossed_edges),
                "priority": "true",
                "width": f"{float(crosswalk.get('width_m', 4.0)):.3f}",
                "shape": _shape(crosswalk["centerline_xy"]),
            })
        _write_xml(connections_root, output / "connections.con.xml")

        return {
            "format": SUMO_MAP_FORMAT,
            "source_path": str(source_path),
            "source_sha256": digest,
            "net_file": "network.net.xml",
            "counts": {
                "junctions": len(used_nodes),
                "edges": len(ordered_groups),
                "lanes": len(lanes) + len(ordered_groups),
                "vehicle_lanes": len(lanes),
                "sidewalks": len(ordered_groups),
                "connectors": valid_connectors,
                "crosswalks": len(crosswalk_input_edges),
                "pedestrian_approaches": len(
                    pedestrian_approach_edge_by_id),
                "traffic_lights": len(signal_connectors),
            },
            "edge_by_lane": edge_by_lane,
            "lane_index_by_lane": lane_index_by_lane,
            "lane_by_edge_index": {
                edge: {str(index): lane_id for index, lane_id in values.items()}
                for edge, values in lane_by_edge_index.items()
            },
            "pedestrian_edge_by_node_pair": pedestrian_edge_by_node_pair,
            "crosswalk_input_edges": crosswalk_input_edges,
            "pedestrian_approach_edge_by_id": (
                pedestrian_approach_edge_by_id),
            # Filled from the final netconvert output. Internal-lane IDs are
            # generated by SUMO and cannot be predicted from PlainXML safely.
            "connector_via_lane": {},
            "connector_link_index": {},
            "connector_tls": {},
            # Filled from SUMO's first-pass, automatically generated signal
            # program. Composite junctions can contain controlled links that
            # are not one of the map's authored driving connectors.
            "tls_link_count": {},
            "sidewalk_lane_by_edge": {},
            "crosswalk_edge_by_id": {},
            "crosswalk_link_index": {},
            "crosswalk_tls": {},
        }

    @staticmethod
    def _validate_network(net_file: Path, manifest: Mapping[str, Any]) -> None:
        root = ET.parse(net_file).getroot()
        SumoMapConverter._validate_connector_geometry(root, manifest)
        for edge_id, indexes in manifest["lane_by_edge_index"].items():
            edge = root.find(f"edge[@id='{edge_id}']")
            if edge is None:
                continue  # The missing-edge check below reports this case.
            lengths = set()
            for index in indexes:
                lane_id = f"{edge_id}_{index}"
                lane = edge.find(f"lane[@id='{lane_id}']")
                if lane is None:
                    raise RuntimeError(f"SUMO conversion missing vehicle lane: {lane_id}")
                lengths.add(lane.get("length"))
            if len(lengths) != 1:
                raise RuntimeError(f"inconsistent vehicle lane lengths: {edge_id}")
        authored_edge_ids = set(manifest["lane_by_edge_index"])
        external_edges = [
            item for item in root.findall("edge")
            if item.get("id") in authored_edge_ids
        ]
        lanes = [lane for edge in external_edges for lane in edge.findall("lane")]
        expected = manifest["counts"]
        if len(external_edges) != int(expected["edges"]):
            raise RuntimeError(
                "SUMO conversion lost road edges: "
                f"{len(external_edges)} != {expected['edges']}")
        if len(lanes) != int(expected["lanes"]):
            raise RuntimeError(
                "SUMO conversion lost lanes: "
                f"{len(lanes)} != {expected['lanes']}")
        ids = {edge.get("id") for edge in external_edges}
        missing = set(manifest["lane_by_edge_index"]) - ids
        if missing:
            raise RuntimeError(
                f"SUMO conversion missing mapped edges: {sorted(missing)[:5]}")
        if len(manifest["connector_via_lane"]) != int(expected["connectors"]):
            raise RuntimeError(
                "SUMO conversion lost connector internal lanes: "
                f"{len(manifest['connector_via_lane'])} != "
                f"{expected['connectors']}")
        if len(manifest.get("sidewalk_lane_by_edge", {})) != int(
                expected["sidewalks"]):
            raise RuntimeError("SUMO conversion lost sidewalk lanes")
        approach_edge_ids = set(manifest.get(
            "pedestrian_approach_edge_by_id", {}).values())
        if len(approach_edge_ids) != int(expected.get(
                "pedestrian_approaches", 0)):
            raise RuntimeError(
                "SUMO conversion has an invalid pedestrian approach mapping")
        compiled_approaches = {
            str(edge.get("id")): edge
            for edge in root.findall("edge")
            if edge.get("id") in approach_edge_ids
        }
        if set(compiled_approaches) != approach_edge_ids:
            raise RuntimeError(
                "SUMO conversion lost authored pedestrian approaches")
        for edge_id, edge in compiled_approaches.items():
            lanes = edge.findall("lane")
            if len(lanes) != 1 or "pedestrian" not in set(
                    lanes[0].get("allow", "").split()):
                raise RuntimeError(
                    "SUMO pedestrian approach is not walkable: "
                    f"{edge_id}")
        if len(manifest.get("crosswalk_edge_by_id", {})) != int(
                expected["crosswalks"]):
            raise RuntimeError("SUMO conversion lost pedestrian crossings")
        if len(set(manifest.get("crosswalk_edge_by_id", {}).values())) != int(
                expected["crosswalks"]):
            raise RuntimeError(
                "SUMO conversion reused one crossing for multiple authored "
                "crosswalks")


def route_edges_for_vehicle(vehicle: Any, bundle: SumoMapBundle) -> List[str]:
    """Translate the vehicle's lane route into consecutive SUMO edges."""
    lane_ids: List[str] = []
    if getattr(vehicle, "current_lane_id", ""):
        lane_ids.append(vehicle.current_lane_id)
    actions = getattr(vehicle, "lane_route_actions", [])
    action_index = max(0, int(getattr(
        vehicle, "lane_route_action_index", 0)))
    for action in actions[action_index:]:
        target = action.get("to_lane_id")
        if target:
            lane_ids.append(str(target))
    edges: List[str] = []
    for lane_id in lane_ids:
        edge = bundle.edge_by_lane.get(lane_id)
        if edge and (not edges or edges[-1] != edge):
            edges.append(edge)
    return edges
