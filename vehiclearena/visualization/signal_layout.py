"""Read-only signal placement, including mapped movements without stop records."""


def signal_stop_lines(data: dict) -> list[dict]:
    """Keep authored lines, deriving missing controlled approaches at lane end.

    The physical stop-progress fallback is 1.0 when no stop line is authored.
    In particular, inferred U-turn connectors can have a real signal phase but
    no stop_lines entry. Do not silently omit their lamps or edit the map.
    """
    lines = list(data.get("stop_lines", []))
    covered = {line["lane_id"] for line in lines}
    controlled = {
        cid for plan in data.get("signal_plans", [])
        for phase in plan["phases"] for cid in phase["connector_ids"]
    }
    lanes = {lane["id"]: lane for lane in data["lanes"]}
    for connector in data["connectors"]:
        lane_id = connector["from_lane"]
        if connector["id"] not in controlled or lane_id in covered:
            continue
        lane = lanes[lane_id]
        left = lane.get("left_boundary_xy", [])
        right = lane.get("right_boundary_xy", [])
        if not left or not right:
            raise ValueError(f"Cannot place signal for lane {lane_id}: missing boundaries")
        lines.append({
            "id": f"derived_signal_stop::{lane_id}",
            "lane_id": lane_id, "node_id": connector["node_id"],
            "line_xy": [list(left[-1]), list(right[-1])],
            "source": "lane_endpoint_signal_display",
        })
        covered.add(lane_id)
    return lines
