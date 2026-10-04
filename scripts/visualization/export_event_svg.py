"""Export red-light event scenes as zoomable SVG files.

SVG text renders with the *viewer's* fonts, so Chinese labels display
correctly in VSCode preview or a browser even though this host has no fonts.

Usage:
    python scripts/visualization/export_event_svg.py TRACE.json.gz [-o out_dir]
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
from collections import Counter
from pathlib import Path

NETDIR = Path(__file__).resolve().parents[2] / "vehiclearena/simulation/road_networks"
RADIUS_M = 100.0
WINDOW_S = 12.0
W, H = 1100, 980
MARGIN = 70

SIGNAL_COLORS = {"green": "#3a3", "yellow": "#dc3", "red": "#d43",
                 "unsignalized": "#bbb"}


def centerline(rec):
    if rec.get("centerline_xy"):
        return [tuple(p) for p in rec["centerline_xy"]]
    left, right = rec["left_boundary_xy"], rec["right_boundary_xy"]
    return [((l[0] + r[0]) / 2, (l[1] + r[1]) / 2) for l, r in zip(left, right)]


def signal_lookup(net):
    plan_by_conn = {}
    for plan in net.get("signal_plans", []):
        cycle = sum(ph["green_s"] + ph["yellow_s"] + ph["all_red_s"]
                    for ph in plan["phases"])
        for ph in plan["phases"]:
            for cid in ph["connector_ids"]:
                plan_by_conn[cid] = {"cycle_s": cycle, "phases": plan["phases"]}

    def signal_state(cid, t):
        plan = plan_by_conn.get(cid)
        if not plan:
            return "unsignalized"
        ct = t % plan["cycle_s"]
        cursor = 0.0
        for ph in plan["phases"]:
            d = ph["green_s"] + ph["yellow_s"] + ph["all_red_s"]
            if ct < cursor + d:
                el = ct - cursor
                if cid in ph["connector_ids"]:
                    if el < ph["green_s"]:
                        return "green"
                    if el < ph["green_s"] + ph["yellow_s"]:
                        return "yellow"
                return "red"
            cursor += d
        return "red"

    def runs(cid, t_center, half=30.0):
        out = []
        t = t_center - half
        while t < t_center + half:
            sig = signal_state(cid, t)
            lo = t
            hi = min(t + 0.05, t_center + half)
            # coarse then fine boundary search
            step = 0.5
            probe = lo
            while probe + step < t_center + half \
                    and signal_state(cid, probe + step) == sig:
                probe += step
            a, b = probe, min(probe + step, t_center + half)
            while b - a > 0.05:
                m = (a + b) / 2
                if signal_state(cid, m) == sig:
                    a = m
                else:
                    b = m
            out.append((lo - t_center, b - t_center, sig))
            t = b
        return out

    return signal_state, runs


class Svg:
    def __init__(self, pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.x0, self.x1 = min(xs) - 15, max(xs) + 15
        self.y0, self.y1 = min(ys) - 15, max(ys) + 15
        sx = (W - 2 * MARGIN) / max(self.x1 - self.x0, 1)
        sy = (H - 2 * MARGIN - 120) / max(self.y1 - self.y0, 1)
        self.s = min(sx, sy)
        self.parts = []

    def px(self, p):
        return (MARGIN + (p[0] - self.x0) * self.s,
                MARGIN + (self.y1 - p[1]) * self.s)

    def poly(self, points, color, width, dash=None, opacity=1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                       (self.px(p) for p in points))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"{dash_attr} opacity="{opacity}" '
            f'stroke-linecap="round"/>')

    def circle(self, p, r, color, fill="none", width=3):
        x, y = self.px(p)
        self.parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" '
            f'stroke="{color}" stroke-width="{width}"/>')

    def text(self, p, s, color="#000", size=14, anchor="start"):
        x, y = self.px(p) if isinstance(p, tuple) else p
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor}" '
            f'font-family="sans-serif">{html.escape(s)}</text>')

    def raw(self, s):
        self.parts.append(s)

    def save(self, path, title):
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
            f'height="{H}" viewBox="0 0 {W} {H}">'
            f'<rect width="{W}" height="{H}" fill="white"/>'
            + "".join(self.parts) + "</svg>", encoding="utf-8")


def export(trace_path, out_dir):
    with gzip.open(trace_path, "rt") as f:
        p = json.load(f)
    vname = p["variant"]["variant_id"]
    net = json.loads((NETDIR / (
        p["variant"]["scenario"]["road_network_id"]
        + "_lane_level.json")).read_text())
    signal_state, runs = signal_lookup(net)
    lanes = {l["id"]: l for l in net["lanes"]}
    conns = {c["id"]: c for c in net["connectors"]}
    traj = {}
    for r in p["trajectories"]["vehicles"]:
        traj.setdefault(r["vehicle_id"], []).append(r)
    for rows in traj.values():
        rows.sort(key=lambda r: r["time_s"])

    made = []
    for e in p.get("physics_events", []):
        if not (e.get("type") == "connector_entered"
                and e["details"].get("signal") == "red"):
            continue
        vid, t_ev = e["vehicle_id"], e["time_s"]
        reported = e["details"]["connector_id"]
        rows = [r for r in traj.get(vid, [])
                if t_ev - WINDOW_S <= r["time_s"] <= t_ev + WINDOW_S]
        if not rows:
            continue
        before = [r["current_lane_id"] for r in rows
                  if r["time_s"] <= t_ev - 0.5]
        approach = Counter(before).most_common(1)[0][0] if before else None
        landing, seen = None, False
        for r in rows:
            if r["time_s"] < t_ev:
                continue
            if r["active_connector_id"]:
                seen = True
            elif seen and r["current_lane_id"] != approach:
                landing = r["current_lane_id"]
                break
        physical = None
        for cid, c in conns.items():
            if c["from_lane"] == approach and landing and c["to_lane"] == landing:
                physical = cid
                break

        cx = sum(r["pose_x_m"] for r in rows) / len(rows)
        cy = sum(r["pose_y_m"] for r in rows) / len(rows)
        near_lanes = [l for l in lanes.values()
                      if any(abs(pt[0] - cx) < RADIUS_M
                             and abs(pt[1] - cy) < RADIUS_M
                             for pt in centerline(l))]
        near_conns = [c for c in conns.values()
                      if any(abs(pt[0] - cx) < RADIUS_M
                             and abs(pt[1] - cy) < RADIUS_M
                             for pt in centerline(c))]
        pts = [pt for l in near_lanes for pt in centerline(l)]
        pts += [pt for c in near_conns for pt in centerline(c)]
        pts += [(r["pose_x_m"], r["pose_y_m"]) for r in rows]
        svg = Svg(pts)

        for l in near_lanes:
            svg.poly(centerline(l), "#d8d8d8", 2)
        for c in near_conns:
            svg.poly(centerline(c), "#ead9c2", 2)
        if approach in lanes:
            svg.poly(centerline(lanes[approach]), "#333", 4)
        if landing in lanes:
            svg.poly(centerline(lanes[landing]), "#333", 4)
        if physical and physical in conns:
            svg.poly(centerline(conns[physical]), "#0a0", 6)
        if reported in conns:
            rep_pts = centerline(conns[reported])
            if all(svg.x0 < pt[0] < svg.x1 and svg.y0 < pt[1] < svg.y1
                   for pt in rep_pts):
                svg.poly(rep_pts, "#d00", 6)

        # trajectory with time gradient blue -> red
        t0, t1 = rows[0]["time_s"], rows[-1]["time_s"]
        for a, b in zip(rows, rows[1:]):
            frac = (a["time_s"] - t0) / max(t1 - t0, 1e-9)
            color = f"#{int(255*frac):02x}28{int(255*(1-frac)):02x}"
            svg.poly([(a["pose_x_m"], a["pose_y_m"]),
                      (b["pose_x_m"], b["pose_y_m"])], color, 4)
        ev = min(rows, key=lambda r: abs(r["time_s"] - t_ev))
        svg.circle((ev["pose_x_m"], ev["pose_y_m"]), 10, "#d00")
        svg.text((ev["pose_x_m"] + 2, ev["pose_y_m"] + 2),
                 f"{vid} t={t_ev:.1f}s", "#d00")

        # title block
        svg.text((MARGIN, 30), f"{vname}", "#000", 18)
        svg.text((MARGIN, 52),
                 f"connector_entered recorded RED · vehicle={vid} · "
                 f"t={t_ev:.1f}s", "#666", 14)

        # signal timeline strips (bottom)
        y = H - 95
        for label, cid, color in (
                ("记录 REPORTED", reported, "#c00"),
                ("物理 PHYSICAL", physical, "#080")):
            if not cid:
                svg.text((MARGIN, y + 12), f"{label}: 未找到（无灯路口?)",
                         color, 14)
                y += 48
                continue
            node = conns.get(cid, {}).get("node_id", "?")
            turn = conns.get(cid, {}).get("turn", "?")
            sig = signal_state(cid, t_ev)
            svg.text((MARGIN, y + 12),
                     f"{label}: {cid} ({turn}, node {node}) "
                     f"@t=事件时刻 → {sig}", color, 14)
            x0, w_total = MARGIN + 30, W - 2 * MARGIN - 60
            for a, b, s in runs(cid, t_ev):
                xa = x0 + (a + 30) / 60 * w_total
                xb = x0 + (b + 30) / 60 * w_total
                svg.raw(f'<rect x="{xa:.1f}" y="{y + 20}" '
                        f'width="{xb - xa:.1f}" height="14" '
                        f'fill="{SIGNAL_COLORS.get(s, "#ccc")}"/>')
            svg.raw(f'<rect x="{x0:.1f}" y="{y + 20}" width="{w_total}" '
                    f'height="14" fill="none" stroke="#888"/>')
            svg.raw(f'<line x1="{x0 + w_total/2:.1f}" y1="{y + 16}" '
                    f'x2="{x0 + w_total/2:.1f}" y2="{y + 38}" '
                    f'stroke="#000" stroke-width="2"/>')
            y += 52

        out = out_dir / f"{vname}__{vid}_t{int(round(t_ev))}.svg"
        svg.save(out, vname)
        made.append(out)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path)
    args = ap.parse_args()
    out_dir = args.out_dir or (
        args.trace.parent / "analysis" / "event_svg")
    out_dir.mkdir(parents=True, exist_ok=True)
    made = export(args.trace, out_dir)
    for m in made:
        print(m)
    print(f"{len(made)} SVG files")


if __name__ == "__main__":
    main()
