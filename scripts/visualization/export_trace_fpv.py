"""Render a first-person (cockpit) MP4 from a recorded trace.

Simplified 2.5D perspective: lane/connector/crosswalk ground polygons,
vehicle boxes, and signal heads showing the *authored phase-table* state
(the same state the attribution audit cross-checks). No system fonts needed.

Usage:
    python scripts/visualization/export_trace_fpv.py TRACE.json.gz [-o out.mp4]
        [--vehicle ego] [--snapshot 286.9] [--sim-step 0.1] [--fps 10]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

NETDIR = Path(__file__).resolve().parents[2] / "vehiclearena/simulation/road_networks"
W, H = 1280, 720
CAM_HEIGHT = 1.5
HORIZON_FRAC = 0.42
FOV_DEG = 65.0
NEAR_M = 0.5
VIEW_M = 100.0
SIGNAL_HEAD_Z = 5.2

F_PX = (W / 2) / math.tan(math.radians(FOV_DEG) / 2)
HORIZON_Y = H * HORIZON_FRAC


def load(trace_path):
    with gzip.open(trace_path, "rt") as f:
        p = json.load(f)
    network = p["variant"]["scenario"]["road_network_id"]
    net = json.loads((NETDIR / f"{network}_lane_level.json").read_text())

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
            return None
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

    lanes = [{"id": l["id"],
              "left": [tuple(q) for q in l["left_boundary_xy"]],
              "right": [tuple(q) for q in l["right_boundary_xy"]]}
             for l in net["lanes"]]
    conns = [{"id": c["id"],
              "left": [tuple(q) for q in c["left_boundary_xy"]],
              "right": [tuple(q) for q in c["right_boundary_xy"]],
              "sig": bool(c.get("signal_controlled")),
              "entry": tuple(c["left_boundary_xy"][0])}
             for c in net["connectors"]]
    crosswalks = [[tuple(q) for q in cw.get("polygon_xy", [])]
                  for cw in net.get("crosswalks", []) if cw.get("polygon_xy")]

    traj = defaultdict(list)
    for r in p["trajectories"]["vehicles"]:
        traj[r["vehicle_id"]].append(r)
    for rows in traj.values():
        rows.sort(key=lambda r: r["time_s"])
    red_events = [
        (e["time_s"], e["vehicle_id"]) for e in p.get("physics_events", [])
        if e.get("type") == "connector_entered"
        and e["details"].get("signal") == "red"]
    return p, lanes, conns, crosswalks, traj, red_events, signal_state


def frame_at(rows, t):
    if not rows:
        return None
    if t <= rows[0]["time_s"]:
        a = b = rows[0]
    elif t >= rows[-1]["time_s"]:
        a = b = rows[-1]
    else:
        lo, hi = 0, len(rows) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if rows[mid]["time_s"] <= t:
                lo = mid
            else:
                hi = mid
        a, b = rows[lo], rows[hi]
    k = ((t - a["time_s"]) / max(b["time_s"] - a["time_s"], 1e-9)
         if a is not b else 0.0)
    return {
        "x": a["pose_x_m"] + (b["pose_x_m"] - a["pose_x_m"]) * k,
        "y": a["pose_y_m"] + (b["pose_y_m"] - a["pose_y_m"]) * k,
        "yaw": a["yaw_rad"] + (b["yaw_rad"] - a["yaw_rad"]) * k,
        "v": a["speed_kmh"],
    }


class Camera:
    def __init__(self, x, y, yaw):
        self.x, self.y = x, y
        self.fx, self.fy = math.cos(yaw), math.sin(yaw)
        self.lx, self.ly = -math.sin(yaw), math.cos(yaw)

    def project(self, wx, wy, wz=0.0):
        """Return (sx, sy, fwd) or None if behind near plane."""
        dx, dy = wx - self.x, wy - self.y
        fwd = dx * self.fx + dy * self.fy
        left = dx * self.lx + dy * self.ly
        if fwd < NEAR_M:
            return None
        sx = W / 2 + F_PX * left / fwd
        sy = HORIZON_Y + F_PX * (CAM_HEIGHT - wz) / fwd
        return sx, sy, fwd

    def project_clipped(self, pts3):
        """Project a polygon, clipping points against the near plane."""
        out = []
        n = len(pts3)
        for i in range(n):
            ax, ay, az = pts3[i]
            bx, by, bz = pts3[(i + 1) % n]
            for (px_, py_, pz_) in ((ax, ay, az),):
                pr = self.project(px_, py_, pz_)
                if pr:
                    out.append(pr[:2])
            # edge crossing the near plane -> add intersection
            fa = (ax - self.x) * self.fx + (ay - self.y) * self.fy
            fb = (bx - self.x) * self.fx + (by - self.y) * self.fy
            if (fa < NEAR_M) != (fb < NEAR_M):
                k = (NEAR_M - fa) / (fb - fa)
                ix, iy, iz = ax + (bx - ax) * k, ay + (by - ay) * k, \
                    az + (bz - az) * k
                pr = self.project(ix, iy, iz)
                if pr:
                    out.append(pr[:2])
        return out


def ground_polygon(left, right):
    pts = [(x, y, 0.0) for x, y in left]
    pts += [(x, y, 0.0) for x, y in reversed(right)]
    return pts


def render_frame(t, cam_pose, lanes, conns, crosswalks, traj, red_events,
                 signal_state, follow):
    img = Image.new("RGB", (W, H), (135, 160, 185))          # sky
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, int(HORIZON_Y), W, H], fill=(150, 150, 148))  # ground
    cam = Camera(*cam_pose)

    # gather + sort far-to-near by nearest point distance
    def dist2(points):
        return min((x - cam.x) ** 2 + (y - cam.y) ** 2 for x, y in points)

    surfaces = []
    for l in lanes:
        if dist2(l["left"]) < VIEW_M ** 2:
            surfaces.append((dist2(l["left"]), ground_polygon(
                l["left"], l["right"]), (82, 82, 82)))
    for c in conns:
        if dist2(c["left"]) < VIEW_M ** 2:
            surfaces.append((dist2(c["left"]), ground_polygon(
                c["left"], c["right"]), (96, 94, 88)))
    for cw in crosswalks:
        if dist2(cw) < VIEW_M ** 2:
            surfaces.append((dist2(cw),
                             [(x, y, 0.02) for x, y in cw],
                             (200, 200, 200)))
    surfaces.sort(key=lambda s: -s[0])
    for _, poly, color in surfaces:
        pts = cam.project_clipped(poly)
        if len(pts) >= 3:
            dr.polygon(pts, fill=color)

    # lane boundary lines for depth legibility
    for l in lanes:
        if dist2(l["left"]) < VIEW_M ** 2:
            for boundary in (l["left"], l["right"]):
                pts = cam.project_clipped(
                    [(x, y, 0.03) for x, y in boundary])
                if len(pts) >= 2:
                    dr.line(pts, fill=(210, 210, 205), width=2)

    # vehicles far-to-near as 3d boxes (silhouette of 8 corners)
    boxes = []
    for vid, rows in traj.items():
        if vid == follow:
            continue
        f = frame_at(rows, t)
        if not f:
            continue
        d2 = (f["x"] - cam.x) ** 2 + (f["y"] - cam.y) ** 2
        if d2 < VIEW_M ** 2:
            boxes.append((d2, f))
    boxes.sort(key=lambda b: -b[0])
    for d2, f in boxes:
        L, Wd = 2.3, 0.95
        ca, sa = math.cos(f["yaw"]), math.sin(f["yaw"])
        corners = []
        for rx, ry in ((L, Wd), (L, -Wd), (-L, -Wd), (-L, Wd)):
            corners.append((f["x"] + rx * ca - ry * sa,
                            f["y"] + rx * sa + ry * ca))
        pts3 = [(x, y, 0.0) for x, y in corners] + \
               [(x, y, 1.6) for x, y in corners]
        proj = [cam.project(x, y, z) for x, y, z in pts3]
        proj = [q[:2] for q in proj if q]
        if len(proj) >= 3:
            xs = [q[0] for q in proj]
            ys = [q[1] for q in proj]
            dr.rectangle([min(xs), min(ys), max(xs), max(ys)],
                         fill=(60, 60, 65), outline=(20, 20, 20))

    # signal heads above connector entries (authored phase-table state)
    for c in conns:
        if not c["sig"]:
            continue
        ex, ey = c["entry"]
        if (ex - cam.x) ** 2 + (ey - cam.y) ** 2 > 80 ** 2:
            continue
        sig = signal_state(c["id"], t)
        if sig is None:
            continue
        pole_top = cam.project(ex, ey, SIGNAL_HEAD_Z)
        pole_base = cam.project(ex, ey, 0.0)
        if not pole_top or not pole_base:
            continue
        dr.line([pole_base[:2], pole_top[:2]], fill=(40, 40, 40), width=2)
        color = {"green": (40, 220, 40), "yellow": (240, 210, 40),
                 "red": (230, 40, 40)}[sig]
        r = max(3.0, 220.0 / pole_top[2])
        dr.ellipse([pole_top[0] - r, pole_top[1] - r,
                    pole_top[0] + r, pole_top[1] + r],
                   fill=color, outline=(20, 20, 20))

    # HUD (ASCII only)
    dr.rectangle([0, 0, 660, 60], fill=(255, 255, 255))
    dr.text((10, 8), "first-person replay (simplified geometry)",
            fill=(0, 0, 0))
    dr.text((10, 30), f"t = {t:6.1f} s", fill=(0, 0, 0))
    f = frame_at(traj.get(follow, []), t)
    if f:
        dr.text((200, 30), f"{follow} v = {f['v']:5.1f} km/h",
                fill=(0, 0, 0))
    active = [(te, v) for te, v in red_events if abs(t - te) < 3.0]
    if active:
        dr.rectangle([0, H - 42, W, H], fill=(255, 225, 225))
        dr.text((10, H - 32),
                f"RED-LIGHT RECORDED: {active[0][1]} @ {active[0][0]:.1f}s "
                f"(attribution under audit)", fill=(180, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--vehicle", default="ego")
    ap.add_argument("--sim-step", type=float, default=0.1)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--snapshot", type=float, default=None,
                    help="render a single PNG at this sim time and exit")
    args = ap.parse_args()

    (p, lanes, conns, crosswalks, traj,
     red_events, signal_state) = load(args.trace)
    if args.vehicle not in traj:
        raise SystemExit(f"vehicle {args.vehicle!r} not in trace")

    if args.snapshot is not None:
        f = frame_at(traj[args.vehicle], args.snapshot)
        img = render_frame(args.snapshot, (f["x"], f["y"], f["yaw"]),
                           lanes, conns, crosswalks, traj, red_events,
                           signal_state, args.vehicle)
        out = args.output or Path(f"/tmp/fpv_{args.snapshot:.0f}.png")
        img.save(out)
        print(out)
        return

    total_t = float(p["variant"]["scenario"].get("total_time_s")
                    or max(r[-1]["time_s"] for r in traj.values()))
    out = args.output or args.trace.with_suffix("").with_suffix(".fpv.mp4")
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    t, n = 0.0, 0
    while t <= total_t:
        f = frame_at(traj[args.vehicle], t)
        img = render_frame(t, (f["x"], f["y"], f["yaw"]),
                           lanes, conns, crosswalks, traj, red_events,
                           signal_state, args.vehicle)
        proc.stdin.write(img.tobytes())
        t += args.sim_step
        n += 1
    proc.stdin.close()
    proc.wait()
    print(f"{out}  ({n} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
