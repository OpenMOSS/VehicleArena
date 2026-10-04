"""Render a recorded trace into an MP4 bird's-eye video (headless, no fonts).

Usage:
    python scripts/visualization/export_trace_video.py TRACE.json.gz [-o out.mp4]
        [--follow ego] [--sim-step 0.2] [--fps 10]

Frames are drawn with PIL (no system fonts needed; HUD text is ASCII).
Encoding uses imageio-ffmpeg's bundled ffmpeg (libx264).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

NETDIR = Path(__file__).resolve().parents[2] / "vehiclearena/simulation/road_networks"
W, H = 1280, 960


def centerline(rec):
    if rec.get("centerline_xy"):
        return [tuple(p) for p in rec["centerline_xy"]]
    left, right = rec["left_boundary_xy"], rec["right_boundary_xy"]
    return [((l[0] + r[0]) / 2, (l[1] + r[1]) / 2) for l, r in zip(left, right)]


def load(trace_path):
    with gzip.open(trace_path, "rt") as f:
        p = json.load(f)
    network = p["variant"]["scenario"]["road_network_id"]
    net = json.loads((NETDIR / f"{network}_lane_level.json").read_text())
    lanes = [centerline(l) for l in net["lanes"]]
    conns = [centerline(c) for c in net["connectors"]]
    traj = defaultdict(list)
    for r in p["trajectories"]["vehicles"]:
        traj[r["vehicle_id"]].append(r)
    for rows in traj.values():
        rows.sort(key=lambda r: r["time_s"])
    red_events = [
        (e["time_s"], e["vehicle_id"]) for e in p.get("physics_events", [])
        if e.get("type") == "connector_entered"
        and e["details"].get("signal") == "red"]
    return p, lanes, conns, traj, red_events


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


def render_video(trace_path, out_path, follow, sim_step, fps, view_half_m):
    p, lanes, conns, traj, red_events = load(trace_path)
    total_t = float(p["variant"]["scenario"].get("total_time_s")
                    or max(r[-1]["time_s"] for r in traj.values()))
    meta = p["variant"]["variant_id"]

    def camera(t):
        if follow and follow in traj:
            f = frame_at(traj[follow], t)
            if f:
                return f["x"], f["y"]
        xs = [pt[0] for ln in lanes for pt in ln]
        ys = [pt[1] for ln in lanes for pt in ln]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    ffmpeg = None
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             "-movflags", "+faststart", str(out_path)],
            stdin=subprocess.PIPE)
        t = 0.0
        n = 0
        while t <= total_t:
            cx, cy = camera(t)
            scale = (W / 2) / view_half_m

            def px(x, y):
                return (W / 2 + (x - cx) * scale, H / 2 - (y - cy) * scale)

            img = Image.new("RGB", (W, H), (250, 250, 250))
            dr = ImageDraw.Draw(img)
            # draw lanes in view
            for ln in lanes:
                pts = [px(*pt) for pt in ln
                       if abs(pt[0] - cx) < view_half_m * 1.3
                       and abs(pt[1] - cy) < view_half_m * 1.3]
                if len(pts) >= 2:
                    dr.line(pts, fill=(215, 215, 215), width=3)
            for cn in conns:
                pts = [px(*pt) for pt in cn
                       if abs(pt[0] - cx) < view_half_m * 1.3
                       and abs(pt[1] - cy) < view_half_m * 1.3]
                if len(pts) >= 2:
                    dr.line(pts, fill=(232, 220, 200), width=3)
            for vid, rows in traj.items():
                f = frame_at(rows, t)
                if not f:
                    continue
                qx, qy = px(f["x"], f["y"])
                if not (-50 < qx < W + 50 and -50 < qy < H + 50):
                    continue
                L, Wd = 4.6 * scale, 1.9 * scale
                ca, sa = math.cos(-f["yaw"]), math.sin(-f["yaw"])
                rect = [(-L / 2, -Wd / 2), (L / 2, -Wd / 2),
                        (L / 2, Wd / 2), (-L / 2, Wd / 2)]
                poly = [(qx + rx * ca - ry * sa, qy + rx * sa + ry * ca)
                        for rx, ry in rect]
                color = (40, 90, 220) if vid == "ego" else (120, 120, 120)
                if any(abs(t - te) < 3.0 and v == vid for te, v in red_events):
                    color = (220, 30, 30)
                dr.polygon(poly, fill=color)
            # HUD (ASCII only: no fonts on this host)
            dr.rectangle([0, 0, 560, 62], fill=(255, 255, 255))
            dr.text((10, 8), f"{meta}", fill=(0, 0, 0))
            dr.text((10, 30), f"t = {t:6.1f} s / {total_t:.0f} s",
                    fill=(0, 0, 0))
            ego_f = frame_at(traj.get(follow, []), t) if follow else None
            if ego_f:
                dr.text((300, 30), f"ego v = {ego_f['v']:5.1f} km/h",
                        fill=(0, 0, 0))
            active = [(te, v) for te, v in red_events if abs(t - te) < 3.0]
            if active:
                dr.rectangle([0, H - 46, W, H], fill=(255, 230, 230))
                dr.text((10, H - 36),
                        f"RED-LIGHT RECORDED: {active[0][1]} @ "
                        f"{active[0][0]:.1f}s  (attribution under audit)",
                        fill=(180, 0, 0))
            proc.stdin.write(img.tobytes())
            t += sim_step
            n += 1
        proc.stdin.close()
        proc.wait()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--follow", default="ego",
                    help="vehicle to follow; empty string = fixed full-map view")
    ap.add_argument("--sim-step", type=float, default=0.2,
                    help="simulated seconds per video frame")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--view-half-m", type=float, default=130.0,
                    help="camera half-width in metres when following")
    args = ap.parse_args()
    out = args.output or args.trace.with_suffix("").with_suffix(".mp4")
    n = render_video(args.trace, out, args.follow or None,
                     args.sim_step, args.fps, args.view_half_m)
    print(f"{out}  ({n} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
