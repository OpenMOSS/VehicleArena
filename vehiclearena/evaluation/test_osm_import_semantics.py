"""OSM conversion regression tests without downloading external data."""

from __future__ import annotations

import json
import os
import sys
import types

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
simulation_package = types.ModuleType("simulation")
simulation_package.__path__ = [os.path.join(ROOT, "simulation")]
sys.modules["simulation"] = simulation_package

from simulation.osm_import import _convert_graph
from simulation.road_network import RoadNetwork


class _Nodes:
    def __init__(self, records):
        self.records = records

    def __call__(self, data=False):
        return list(self.records.items()) if data else list(self.records)


class _Edges:
    def __init__(self, records):
        self.records = records

    def __call__(self, node_id=None, keys=False, data=False):
        records = self.records
        if node_id is not None:
            records = [
                item for item in records
                if item[0] == node_id or item[1] == node_id]
        if keys and data:
            return list(records)
        if data:
            return [(u, v, attrs) for u, v, _key, attrs in records]
        if keys:
            return [(u, v, key) for u, v, key, _attrs in records]
        return [(u, v) for u, v, _key, _attrs in records]


class _Graph:
    def __init__(self, nodes, edges):
        self.nodes = _Nodes(nodes)
        self.edges = _Edges(edges)


def main() -> None:
    graph = _Graph(
        {
            1: {"x": 116.0, "y": 39.0},
            2: {"x": 116.001, "y": 39.0},
            3: {"x": 116.002, "y": 39.0},
        },
        [
            (1, 2, 0, {
                "highway": "primary", "oneway": "yes",
                "lanes": "2", "maxspeed": "30 mph",
                "turn:lanes": "through|right", "length": 100.0,
                "name": "Parallel Road", "osmid": 100,
            }),
            (2, 1, 0, {
                "highway": "primary", "oneway": "yes",
                "lanes": "1", "maxspeed": "50",
                "turn:lanes": "left", "length": 100.0,
                "name": "Parallel Road", "osmid": 101,
            }),
            (2, 3, 0, {
                "highway": "secondary", "oneway": "-1",
                "lanes": "1", "maxspeed": "40", "length": 80.0,
                "layer": "-1", "tunnel": "yes",
                "width": "3.2", "name": "Reverse Tunnel",
                "osmid": 102,
            }),
        ],
    )

    network = _convert_graph(graph)
    parallel = network.get_segment("n1_n2")
    assert parallel and not parallel.oneway
    assert parallel.lanes_forward == 2
    assert parallel.lanes_backward == 1
    assert parallel.lanes == 3
    assert len(parallel.forward_lanes()) == 2
    assert len(parallel.backward_lanes()) == 1
    assert parallel.speed_limit == 48
    assert parallel.turn_lanes_forward == ["through", "right"]
    assert parallel.turn_lanes_backward == ["left"]
    assert parallel.source_edge_count == 2

    reverse = network.get_segment("n2_n3")
    assert reverse and reverse.oneway
    assert reverse.from_node == "n3" and reverse.to_node == "n2"
    assert reverse.layer == -1 and reverse.tunnel
    assert abs(reverse.lane_width_meters - 3.2) < 1e-9

    restored = RoadNetwork.from_dict(network.to_dict())
    restored_parallel = restored.get_segment("n1_n2")
    assert restored_parallel.lanes_forward == 2
    assert restored_parallel.turn_lanes_backward == ["left"]

    print(json.dumps({
        "status": "PASS",
        "mph_to_kmh": parallel.speed_limit,
        "parallel_edges_preserved_as": {
            "forward_lanes": parallel.lanes_forward,
            "backward_lanes": parallel.lanes_backward,
        },
        "reversed_oneway": {
            "from": reverse.from_node,
            "to": reverse.to_node,
            "layer": reverse.layer,
            "tunnel": reverse.tunnel,
        },
        "roundtrip_preserved": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
