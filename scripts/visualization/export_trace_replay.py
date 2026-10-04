"""Export a recorded experiment trace as a self-contained HTML replay page.

Usage:
    python scripts/visualization/export_trace_replay.py TRACE.json.gz [-o replay.html]

The page needs no server: map geometry, trajectories and physics events are
embedded as JSON. Built for debugging attribution/evaluation issues: events
are listed with signal colors, clicking an event jumps the timeline and
selects the vehicle, and red-light entries show a reported-vs-physical
connector comparison with signal timelines.
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
SAMPLE_S = 0.5
EVENT_FINE_HALF_WINDOW_S = 6.0
SIGNAL_STRIP_HALF_WINDOW_S = 30.0


def _centerline(record):
    if record.get("centerline_xy"):
        return [tuple(p) for p in record["centerline_xy"]]
    left, right = record["left_boundary_xy"], record["right_boundary_xy"]
    return [((l[0] + r[0]) / 2, (l[1] + r[1]) / 2) for l, r in zip(left, right)]


def _thin(points, min_step=2.0):
    if len(points) <= 2:
        return points
    out = [points[0]]
    for p in points[1:-1]:
        if math.dist(out[-1], p) >= min_step:
            out.append(p)
    out.append(points[-1])
    return out


def _build_signal_lookup(net):
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

    def strip(cid, t_center):
        """Run-length signal colors over +-SIGNAL_STRIP_HALF_WINDOW_S."""
        t0 = t_center - SIGNAL_STRIP_HALF_WINDOW_S
        t1 = t_center + SIGNAL_STRIP_HALF_WINDOW_S
        runs = []
        t = t0
        while t < t1:
            sig = signal_state(cid, t)
            # find next change by bisecting forward in 0.1 steps
            t_next = min(t1, t + 0.1)
            probe = t
            while t_next < t1:
                if signal_state(cid, t_next) != sig:
                    break
                probe = t_next
                t_next = min(t1, t_next + 0.5)
            runs.append([round(t - t_center, 2), sig])
            t = probe + 0.001
            # refine boundary
            lo, hi = probe, min(t1, probe + 0.6)
            while hi - lo > 0.05:
                mid = (lo + hi) / 2
                if signal_state(cid, mid) == sig:
                    lo = mid
                else:
                    hi = mid
            t = hi
        return runs

    return signal_state, strip


def build_payload(trace_path: Path) -> dict:
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt") as f:
        p = json.load(f)
    network = p["variant"]["scenario"]["road_network_id"]
    net = json.loads((NETDIR / f"{network}_lane_level.json").read_text())
    signal_state, strip = _build_signal_lookup(net)

    lanes = {}
    conns = {}
    for l in net["lanes"]:
        lanes[l["id"]] = [[round(x, 1), round(y, 1)]
                          for x, y in _thin(_centerline(l))]
    for c in net["connectors"]:
        conns[c["id"]] = {
            "pts": [[round(x, 1), round(y, 1)]
                    for x, y in _thin(_centerline(c))],
            "from": c["from_lane"], "to": c["to_lane"],
            "turn": c["turn"], "node": c["node_id"],
            "sig": bool(c.get("signal_controlled")),
        }

    traj = p["trajectories"]["vehicles"]
    by_vid = {}
    for r in traj:
        by_vid.setdefault(r["vehicle_id"], []).append(r)
    lane_ids = sorted(lanes)
    lane_idx = {lid: i for i, lid in enumerate(lane_ids)}
    conn_ids = sorted(conns)
    conn_idx = {cid: i for i, cid in enumerate(conn_ids)}

    vehicles = {}
    for vid, rows in by_vid.items():
        rows.sort(key=lambda r: r["time_s"])
        frames = []
        next_sample = 0.0
        for i, r in enumerate(rows):
            presence_change = i and r.get("present_in_physics_world", True) != rows[i-1].get("present_in_physics_world", True)
            if r["time_s"] + 1e-9 < next_sample and i != len(rows)-1 and not presence_change:
                continue
            next_sample = r["time_s"] + SAMPLE_S
            frames.append([
                round(r["time_s"], 1),
                round(r["pose_x_m"], 1), round(r["pose_y_m"], 1),
                round(r["yaw_rad"], 2),
                round(r["speed_kmh"], 1),
                lane_idx.get(r["current_lane_id"], -1),
                conn_idx.get(r["active_connector_id"], -1)
                if r["active_connector_id"] else -1,
                r.get("present_in_physics_world", True),
            ])
        vehicles[vid] = frames

    # Keep authored geometry: centerlines alone do not describe the road surface.
    environment = {
        "lanes": [{k: l.get(k) for k in (
            "id", "left_boundary_xy", "right_boundary_xy", "speed_limit_kmh")}
            for l in net["lanes"]],
        "connectors": [{k: c.get(k) for k in (
            "id", "left_boundary_xy", "right_boundary_xy")}
            for c in net["connectors"]],
        "stop_lines": net.get("stop_lines", []),
        "crosswalks": net.get("crosswalks", []),
        "intersections": net.get("intersections", []),
        "signal_plans": net.get("signal_plans", []),
    }
    pedestrian_rows = {}
    for r in p["trajectories"].get("pedestrians", []):
        pedestrian_rows.setdefault(r["ped_id"], []).append(r)
    pedestrians = {}
    for pid, rows in pedestrian_rows.items():
        pedestrians[pid] = [[r["time_s"], r["pose_x_m"], r["pose_y_m"],
                             r.get("spawned", True) and not r.get("arrived", False)
                             and r["pose_x_m"] is not None and r["pose_y_m"] is not None]
                            for r in sorted(rows, key=lambda r: r["time_s"])]
    vehicle_dimensions = {
        vid: [rows[0].get("length_m", 4.6), rows[0].get("width_m", 1.9)]
        for vid, rows in by_vid.items() if rows}

    events = []
    for e in p.get("physics_events", []):
        item = {"t": e["time_s"], "type": e["type"], "vid": e["vehicle_id"],
                "details": e.get("details", {})}
        if (e["type"] == "connector_entered"
                and e["details"].get("signal") == "red"):
            item["analysis"] = _analyse_red_event(
                e, by_vid.get(e["vehicle_id"], []), conns, signal_state, strip)
        events.append(item)

    penalties = []
    evaluation_events = []
    evaluations = {}
    for vid, vres in p["result"].get("vehicles", {}).items():
        de = vres.get("driving_evaluation")
        if not de:
            continue
        evaluations[vid] = {
            "dimension_scores": de.get("dimension_scores", {}),
            "hard_safety_passed": de.get("hard_safety_passed"),
            "trajectory_quality_score": de.get("trajectory_quality_score"),
            "single_vehicle_layer_score_100": de.get(
                "single_vehicle_layer_score_100"),
            "deduction_total": (de.get("driving_process") or {}).get("deduction_total"),
            "deductions_available": isinstance((de.get("driving_process") or {}).get("deductions"), list),
        }
        penalties.extend(_recorded_deductions(vid, de))
        for ev in de.get("events", []):
            evaluation_events.append({
                "t": ev.get("time_s"), "vid": vid,
                "type": ev.get("type"),
                "reason": _penalty_reason(ev),
                "details": ev,
            })
        for ep in de.get("decision_episodes", []):
            if ep.get("reasonable", True):
                continue
            evaluation_events.append({
                "t": ep.get("start_time_s"), "vid": vid,
                "type": "unreasonable_" + str(ep.get("type")),
                "reason": str(ep.get("reason", "")),
                "details": ep.get("details", {}),
            })
    penalties.sort(key=lambda x: (x["t"] is None, x["t"] or 0))
    evaluation_events.sort(key=lambda x: (x["t"] is None, x["t"] or 0))

    meta = {
        "variant_id": p["variant"]["variant_id"],
        "network": network,
        "total_time_s": p["variant"]["scenario"].get("total_time_s"),
        "vehicle_count": len(vehicles),
    }
    return {"meta": meta, "lanes": lanes, "connectors": conns,
            "lane_ids": lane_ids, "conn_ids": conn_ids,
            "vehicles": vehicles, "events": events,
            "penalties": penalties, "evaluations": evaluations,
            "evaluation_events": evaluation_events,
            "environment": environment, "pedestrians": pedestrians,
            "vehicle_dimensions": vehicle_dimensions}


def _recorded_deductions(vid, report):
    """Use the scoring ledger, not safety evidence, as the source of point losses."""
    names = {
        "unsignaled_turn": "转向灯未按要求提前开启",
        "unsignaled_lane_change": "变道前未按要求提前开启转向灯",
        "turn_signal_not_cancelled": "转向灯未及时关闭",
        "hard_acceleration": "急加速",
        "high_longitudinal_jerk": "纵向加速度变化过快",
        "unnecessary_hard_brake": "不必要的急刹车",
        "red_light_violation": "闯红灯（评分记录）",
        "leader_response_late": "对前车风险响应过晚",
        "vehicle_near_miss": "与车辆近失",
    }
    items = []
    for deduction in (report.get("driving_process") or {}).get("deductions", []) or []:
        t = deduction.get("start_time_s")
        end = deduction.get("end_time_s")
        # Some aggregate penalties store 0/0 without an actual event timestamp.
        if t == end == 0 and "count" in deduction.get("evidence", {}):
            t = end = None
        items.append({"t": t, "end_t": end, "vid": vid,
                      "type": deduction.get("type", "unknown"),
                      "points": deduction.get("points"),
                      "reason": names.get(deduction.get("type"), deduction.get("type", "未知扣分")),
                      "details": deduction})
    return items


def _penalty_reason(event: dict) -> str:
    kind = event.get("type", "")
    if kind == "near_miss":
        return f"近失: {event.get('hazard', '?')}"
    if kind == "red_light_entry":
        return f"闯红灯(记录): {event.get('connector_id', '?')}"
    if kind == "unsafe_lane_change":
        return (f"不安全变道 → lane {event.get('target_lane')} "
                f"(gap {event.get('nearest_gap_m')} m)")
    return json.dumps(event, ensure_ascii=False)[:160]


def _analyse_red_event(event, rows, conns, signal_state, strip):
    t_ev = event["time_s"]
    reported = event["details"]["connector_id"]
    before = [r["current_lane_id"] for r in rows if r["time_s"] <= t_ev - 0.5]
    approach = Counter(before).most_common(1)[0][0] if before else None
    landing, seen = None, False
    for r in sorted(rows, key=lambda r: r["time_s"]):
        if r["time_s"] < t_ev:
            continue
        if r["active_connector_id"]:
            seen = True
        elif seen and r["current_lane_id"] != approach:
            landing = r["current_lane_id"]
            break
    physical = None
    for cid, c in conns.items():
        if c["from"] == approach and landing is not None and c["to"] == landing:
            physical = cid
            break
    fine = [{
        "t": round(r["time_s"], 1),
        "lane": r["current_lane_id"],
        "conn": r["active_connector_id"] or "",
        "v": round(r["speed_kmh"], 1),
        "prog": round(r["edge_progress"], 3),
    } for r in rows
        if abs(r["time_s"] - t_ev) <= EVENT_FINE_HALF_WINDOW_S]
    out = {
        "reported": reported,
        "reported_node": conns.get(reported, {}).get("node"),
        "reported_turn": conns.get(reported, {}).get("turn"),
        "reported_from": conns.get(reported, {}).get("from"),
        "approach_lane": approach,
        "landing_lane": landing,
        "physical": physical,
        "physical_turn": conns.get(physical, {}).get("turn") if physical else None,
        "lane_match": bool(conns.get(reported)) and conns[reported]["from"] == approach,
        "fine": fine,
        "strips": {
            "reported": strip(reported, t_ev),
            "physical": strip(physical, t_ev) if physical else [],
        },
    }
    return out


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
 body { margin:0; font:13px/1.4 -apple-system, "Segoe UI", sans-serif;
        display:flex; height:100vh; overflow:hidden; }
 #side { width:340px; min-width:340px; overflow-y:auto; border-right:1px solid #ccc;
         padding:8px; background:#fafafa; }
 #main { flex:1; position:relative; }
 canvas { display:block; width:100%; height:100%; cursor:grab; }
 #bar { position:absolute; left:0; right:0; bottom:0; background:rgba(255,255,255,.95);
        border-top:1px solid #ccc; padding:6px 10px; display:flex; gap:10px;
        align-items:center; }
 #time { width:100%; }
 .ev { padding:3px 6px; margin:2px 0; border-radius:4px; cursor:pointer;
       border-left:6px solid #999; background:#fff; }
 .ev:hover { background:#eef; }
 .ev.red { border-color:#d00; } .ev.green { border-color:#0a0; }
 .ev.yellow { border-color:#da0; } .ev.unsignalized { border-color:#bbb; }
 .ev.arrived { border-color:#36c; }
 .ev.penalty { border-color:#a0a; background:#fdf5ff; }
 #info { position:absolute; top:8px; right:8px; background:rgba(255,255,255,.95);
         border:1px solid #ccc; border-radius:6px; padding:8px; max-width:420px;
         font-size:12px; white-space:pre-wrap; display:none; }
 #cmp { background:#fff; border:1px solid #d00; border-radius:6px; padding:8px;
        margin-top:8px; display:none; }
 .strip { position:relative; height:14px; border:1px solid #999; margin:2px 0 8px; }
 .strip div { position:absolute; top:0; bottom:0; }
 .marker { position:absolute; top:-3px; bottom:-3px; width:2px; background:#000; }
 table { border-collapse:collapse; font-size:11px; width:100%; }
 td, th { border:1px solid #ddd; padding:1px 4px; }
 h3 { margin:10px 0 4px; }
 __VIEW_CSS__
</style>
</head>
<body>
<div id="side">
  <h3 id="title"></h3>
  <div id="stats"></div>
  <div id="scores"></div>
  <h3>扣分项（点击跳转）</h3>
  <div id="penalties"></div>
  <details><summary>评估记录（辅助定位，不重复计分）</summary><div id="evaluation-events"></div></details>
  <div id="cmp"></div>
  <h3>事件（点击跳转）</h3>
  <div id="events"></div>
</div>
<div id="main">
  <div id="views">
    <section id="map-panel"><canvas id="cv" aria-label="上帝视角"></canvas>
      <div class="view-label">上帝视角 <span>拖动平移 · 滚轮缩放 · 点击车辆或信号灯</span></div>
      <div id="legend">蓝 / 橙：本车 / 选中车辆　灰：其他车辆　紫：行人<br>红黄绿箭头：转向信号　白横线：停止线　条纹：斑马线</div>
    </section>
    <section id="fp-panel"><canvas id="fp" aria-label="第一视角回放"></canvas>
      <div class="view-label">第一视角 <span id="fp-status"></span></div>
      <div class="view-note">轨迹重建画面 · 信号按地图配时还原</div>
    </section>
  </div>
  <div id="toolbar">
    <label>观察车辆 <select id="vehicle-select"></select></label>
    <label><input id="follow" type="checkbox" checked> 跟随车辆</label>
    <label><input id="labels" type="checkbox" checked> 环境标注</label>
    <label><input id="route" type="checkbox"> 行驶轨迹</label>
    <button id="fit">全图</button><button id="focus">定位车辆</button>
    <button id="record">录制第一视角</button><span id="record-status"></span>
  </div>
  <div id="info"></div>
  <div id="bar">
    <button id="play">▶</button>
    <input id="time" type="range" min="0" max="1000" value="0">
    <span id="clock"></span>
    <span>速度:</span>
    <span id="speeds"></span>
  </div>
</div>
<script>
const DATA = __DATA__;
const lanes = DATA.lanes, conns = DATA.connectors;
const laneIds = DATA.lane_ids, connIds = DATA.conn_ids;
const vehicles = DATA.vehicles;
const vids = Object.keys(vehicles);
let tMin = Infinity, tMax = -Infinity;
for (const v of vids) {
  const f = vehicles[v];
  if (!f.length) continue;
  tMin = Math.min(tMin, f[0][0]); tMax = Math.max(tMax, f[f.length-1][0]);
}
let tNow = tMin, playing = false, speed = 1;
let view = null; // {cx, cy, s}
let selected = vids.includes('ego') ? 'ego' : vids[0];

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
function resize() { cv.width = cv.clientWidth * devicePixelRatio;
  cv.height = cv.clientHeight * devicePixelRatio; }
addEventListener('resize', resize); resize();

// bounds
let bx0=1e18, by0=1e18, bx1=-1e18, by1=-1e18;
for (const id in lanes) for (const p of lanes[id]) {
  bx0=Math.min(bx0,p[0]); by0=Math.min(by0,p[1]);
  bx1=Math.max(bx1,p[0]); by1=Math.max(by1,p[1]); }
view = { cx:(bx0+bx1)/2, cy:(by0+by1)/2,
  s: Math.min(cv.width/(bx1-bx0+80), cv.height/(by1-by0+80)) };

function w2s(x, y) {
  return [cv.width/2 + (x-view.cx)*view.s,
          cv.height/2 - (y-view.cy)*view.s];
}
// pan & zoom
let drag = null;
cv.addEventListener('mousedown', e => drag = {x:e.clientX, y:e.clientY});
addEventListener('mouseup', () => drag = null);
addEventListener('mousemove', e => {
  if (!drag) return;
  view.cx -= (e.clientX-drag.x)*devicePixelRatio/view.s;
  view.cy += (e.clientY-drag.y)*devicePixelRatio/view.s;
  drag = {x:e.clientX, y:e.clientY};
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.15 : 1/1.15;
  view.s *= f;
}, {passive:false});

function frameAt(vid, t) {
  const f = vehicles[vid];
  if (!f || !f.length || t < f[0][0] || t > f[f.length-1][0]) return null;
  let lo = 0, hi = f.length - 1;
  if (t <= f[0][0]) return f[0][7]===false?null:f[0];
  if (t >= f[hi][0]) return f[hi][7]===false?null:f[hi];
  while (hi - lo > 1) { const m = (lo+hi)>>1;
    if (f[m][0] <= t) lo = m; else hi = m; }
  const a = f[lo], b = f[hi], k = (t-a[0])/(b[0]-a[0]||1);
  if (a[7] === false) return null;
  const dyaw = Math.atan2(Math.sin(b[3]-a[3]), Math.cos(b[3]-a[3]));
  return [t, a[1]+(b[1]-a[1])*k, a[2]+(b[2]-a[2])*k,
          a[3]+dyaw*k, a[4]+(b[4]-a[4])*k, a[5], a[6]];
}

__VIEW_JS__

function draw() {
  updateFollow();
  ctx.clearRect(0, 0, cv.width, cv.height);
  drawEnvironment();
  for (const vid of vids) {
    const f = frameAt(vid, tNow);
    if (!f) continue;
    const [x,y,yaw] = [f[1],f[2],f[3]];
    const q = w2s(x,y);
    const dim = DATA.vehicle_dimensions[vid] || [4.6,1.9];
    const L = Math.max(6,dim[0]*view.s), Wd = Math.max(3,dim[1]*view.s);
    ctx.save(); ctx.translate(q[0],q[1]); ctx.rotate(-yaw);
    ctx.fillStyle = vid===selected ? '#f80' : (vid==='ego' ? '#25d' : '#777');
    ctx.fillRect(-L/2, -Wd/2, L, Wd);
    ctx.fillStyle = '#bfdbfe'; ctx.fillRect(L*.12, -Wd*.38, L*.18, Wd*.76);
    ctx.restore();
  }
  // event flash markers
  for (const e of DATA.events) {
    if (Math.abs(e.t - tNow) > 0.6) continue;
    const f = frameAt(e.vid, e.t); if (!f) continue;
    const q = w2s(f[1], f[2]);
    ctx.beginPath(); ctx.arc(q[0], q[1], 10+4*Math.abs(Math.sin(tNow*6)), 0, 7);
    ctx.strokeStyle = e.details.signal==='red' ? '#d00' : '#36c';
    ctx.lineWidth = 3; ctx.stroke();
  }
  // penalty flash markers (magenta)
  for (const e of DATA.penalties) {
    if (e.t == null || Math.abs(e.t - tNow) > 0.6) continue;
    const f = frameAt(e.vid, e.t); if (!f) continue;
    const q = w2s(f[1], f[2]);
    ctx.beginPath(); ctx.arc(q[0], q[1], 14+4*Math.abs(Math.sin(tNow*6)), 0, 7);
    ctx.strokeStyle = '#a0a'; ctx.lineWidth = 3; ctx.stroke();
  }
  document.getElementById('clock').textContent =
    tNow.toFixed(1) + ' s / ' + tMax.toFixed(1) + ' s';
  document.getElementById('time').value = 1000*(tNow-tMin)/(tMax-tMin||1);
  showInfo();
  drawFirstPerson();
}

let last = performance.now();
function loop(now) {
  const dt = (now-last)/1000; last = now;
  if (playing) { tNow += dt*speed; if (tNow>=tMax){tNow=tMax;playing=false;
    if (recorder?.state==='recording') recorder.stop();
    document.getElementById('play').textContent='▶';} }
  draw(); requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

document.getElementById('play').onclick = () => {
  if (!playing && tNow >= tMax) tNow = tMin;
  playing = !playing;
  document.getElementById('play').textContent = playing ? '⏸' : '▶';
};
document.getElementById('time').oninput = e => {
  tNow = tMin + (tMax-tMin)*e.target.value/1000;
};
(function() {
  const box = document.getElementById('speeds');
  for (const s of [1, 2, 4, 8, 16]) {
    const b = document.createElement('button');
    b.textContent = s + 'x';
    b.style.marginLeft = '3px';
    b.onclick = () => { speed = s;
      for (const o of box.children) o.style.fontWeight = '';
      b.style.fontWeight = 'bold'; };
    if (s === 1) b.style.fontWeight = 'bold';
    box.appendChild(b);
  }
})();

cv.addEventListener('click', e => {
  if (panDistance > 4) return;
  const rect = cv.getBoundingClientRect();
  const mx = (e.clientX-rect.left)*devicePixelRatio;
  const my = (e.clientY-rect.top)*devicePixelRatio;
  let best = null, bd = 1e9;
  for (const vid of vids) {
    const f = frameAt(vid, tNow); if (!f) continue;
    const q = w2s(f[1], f[2]);
    const d = Math.hypot(q[0]-mx, q[1]-my);
    if (d < bd) { bd = d; best = vid; }
  }
  if (bd < 15*devicePixelRatio) { selected = best; return; }
  inspectSignal(mx, my);
});

function showInfo() {
  const el = document.getElementById('info');
  if (!selected) { el.style.display='none'; return; }
  const f = frameAt(selected, tNow);
  if (!f) { el.style.display='none'; return; }
  el.style.display='block';
  el.textContent = selected + '  t=' + tNow.toFixed(1) + 's\\n'
    + 'lane: ' + (laneIds[f[5]]||'?') + '\\n'
    + 'connector(recorded): ' + (f[6]>=0?connIds[f[6]]:'-') + '\\n'
    + 'speed: ' + f[4].toFixed(1) + ' km/h';
}

// sidebar
document.getElementById('title').textContent = DATA.meta.variant_id;
document.getElementById('stats').textContent =
  DATA.meta.network + ' · ' + DATA.meta.vehicle_count + ' 车 · ' +
  DATA.events.length + ' 事件 · ' + DATA.penalties.length + ' 扣分项';

// evaluation scores summary
(function() {
  const el = document.getElementById('scores');
  let h = '';
  for (const vid in DATA.evaluations) {
    const ev = DATA.evaluations[vid];
    const ds = ev.dimension_scores || {};
    h += '<b>' + vid + '</b>'
      + '  过程分=' + (ev.single_vehicle_layer_score_100 ?? 'n/a')
      + '  轨迹质量=' + (ev.trajectory_quality_score ?? 'n/a')
      + '  硬安全=' + (ev.hard_safety_passed ? '✓' : '✗') + '<br>'
      + '<span style="color:#666">safety=' + ds.safety
      + ' compliance=' + ds.compliance
      + ' comfort=' + ds.comfort
      + ' efficiency=' + ds.efficiency + '</span><br>';
    if (ev.deduction_total != null) h += '累计扣分=' + ev.deduction_total + ' 分<br>';
  }
  el.innerHTML = h;
})();

// penalty list
(function() {
  const box = document.getElementById('penalties');
  if (!DATA.penalties.length) {
    const reports = Object.values(DATA.evaluations);
    box.textContent = reports.length && reports.every(ev=>ev.deductions_available)
      ? '评分记录中无扣分项' : '原记录未提供完整扣分明细'; return;
  }
  for (const pen of DATA.penalties) {
    const div = document.createElement('div');
    div.className = 'ev penalty';
    const timeText = pen.t == null ? '汇总项（未记录具体时刻）' : pen.t.toFixed(1)
      + (pen.end_t != null && pen.end_t > pen.t ? '–'+pen.end_t.toFixed(1) : '') + 's';
    div.textContent = timeText + '  ' + pen.vid + '  '
      + (pen.points == null ? '分值未记录' : '扣 '+pen.points+' 分') + '  — ' + pen.reason;
    div.title = JSON.stringify(pen.details, null, 1);
    div.onclick = () => {
      if (pen.t == null) return;
      tNow = pen.t; selected = pen.vid;
      const f = frameAt(pen.vid, pen.t);
      if (f) { view.cx = f[1]; view.cy = f[2];
        view.s = Math.max(view.s, Math.min(cv.width, cv.height)/160); }
      showCmp(null);
    };
    box.appendChild(div);
  }
})();

for (const ev of DATA.evaluation_events) {
  const div = document.createElement('div'); div.className = 'ev';
  div.textContent = (ev.t == null ? '?' : ev.t.toFixed(1)+'s') + '  ' + ev.vid + '  ' + ev.reason;
  div.title = JSON.stringify(ev.details,null,1);
  div.onclick = () => { if(ev.t == null)return; tNow=ev.t;selected=ev.vid;focusVehicle();showCmp(null); };
  document.getElementById('evaluation-events').appendChild(div);
}

const evBox = document.getElementById('events');
const sigRank = {red:0, yellow:1, green:2};
const sortedEv = DATA.events.slice().sort((a,b) =>
  (sigRank[a.details.signal]??3)-(sigRank[b.details.signal]??3) || a.t-b.t);
for (const e of sortedEv) {
  const div = document.createElement('div');
  div.className = 'ev ' + (e.details.signal || e.type);
  div.textContent = e.t.toFixed(1)+'s  '+e.vid+'  '+e.type+
    (e.details.signal?'  ['+e.details.signal+']':'')+
    (e.details.turn?' '+e.details.turn:'');
  div.onclick = () => { tNow = e.t; selected = e.vid;
    const f = frameAt(e.vid, e.t);
    if (f) { view.cx=f[1]; view.cy=f[2];
      view.s = Math.max(view.s, Math.min(cv.width, cv.height)/160); }
    showCmp(e.analysis||null);
  };
  evBox.appendChild(div);
}

function stripHtml(runs, w) {
  if (!runs || !runs.length) return '<i>unsignalized / unknown</i>';
  const span = 2*w;
  let h = '<div class="strip">';
  for (let i=0;i<runs.length;i++) {
    const start = runs[i][0];
    const end = i+1<runs.length ? runs[i+1][0] : w;
    const c = {green:'#4c4', yellow:'#dd4', red:'#e55'}[runs[i][1]] || '#ccc';
    h += '<div style="left:'+(100*(start+w)/span)+'%;width:'+
         (100*(end-start)/span)+'%;background:'+c+'"></div>';
  }
  h += '<div class="marker" style="left:50%"></div></div>';
  return h;
}

function showCmp(a) {
  const el = document.getElementById('cmp');
  if (!a) { el.style.display='none'; return; }
  el.style.display='block';
  let h = '<b>红灯事件分析</b><br>'
    + '入口车道: ' + (a.approach_lane||'?') + '<br>'
    + '落点车道: ' + (a.landing_lane||'?') + '<br>'
    + 'from_lane 匹配: ' + (a.lane_match ? '✓（但转向可能错）' : '✗') + '<br><br>'
    + '<b style="color:#c00">记录: '+a.reported+'</b> ('+(a.reported_turn||'?')
    + ', node '+(a.reported_node||'?')+')'
    + stripHtml(a.strips.reported, 30)
    + '<b style="color:#080">物理: '+(a.physical||'未找到(可能无灯路口)')
    + '</b> ('+(a.physical_turn||'?')+')'
    + stripHtml(a.strips.physical, 30);
  h += '<details><summary>±6s 细粒度轨迹</summary><table>'
    + '<tr><th>t</th><th>lane</th><th>connector</th><th>v</th><th>prog</th></tr>';
  for (const r of a.fine)
    h += '<tr><td>'+r.t+'</td><td>'+r.lane+'</td><td>'+r.conn+'</td><td>'
      +r.v+'</td><td>'+r.prog+'</td></tr>';
  h += '</table></details>';
  el.innerHTML = h;
}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    payload = build_payload(args.trace)
    if not any(payload["vehicles"].values()):
        parser.error("trace contains no vehicle frames")
    out = args.output or args.trace.with_suffix("").with_suffix(".replay.html")
    assets = Path(__file__).with_name("replay_assets")
    page = (HTML_TEMPLATE
            .replace("__VIEW_CSS__", (assets / "views.css").read_text())
            .replace("__VIEW_JS__", (assets / "views.js").read_text())
            .replace("__TITLE__", html.escape(payload["meta"]["variant_id"]))
            .replace("__DATA__", json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
