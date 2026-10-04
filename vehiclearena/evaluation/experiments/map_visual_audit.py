"""Render and audit one vehicle GIF for every map-coverage network."""

from __future__ import annotations

import copy
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

from simulation.multi_sim_engine import MultiScenario, MultiSimEngine
from visualization.lane_world_renderer import (
    LaneWorldRenderer, Viewport, WorldRecording,
)


VISUAL_AUDIT_SCHEMA = "vehiclearena-map-visual-audit-v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


class _VisualAuditEngine(MultiSimEngine):
    """Capture the authoritative world after every requested physics step."""

    def __init__(self, scenario: MultiScenario, *, frame_interval_s: float,
                 width: int, height: int):
        super().__init__(scenario)
        self._audit_interval_s = float(frame_interval_s)
        self._audit_width = int(width)
        self._audit_height = int(height)
        self._audit_renderer = None
        self._audit_frames = []
        self._audit_next_time_s = 0.0

    def _make_renderer(self) -> LaneWorldRenderer:
        runtime = self.traffic_mgr._lane_geometry
        poses = []
        for vehicle in self.traffic_mgr.vehicles.values():
            pose = runtime.vehicle_pose(vehicle)
            if pose:
                poses.append(pose[:2])
        if not poses:
            raise ValueError("visual audit scene has no drawable vehicle")
        min_x = min(point[0] for point in poses)
        max_x = max(point[0] for point in poses)
        min_y = min(point[1] for point in poses)
        max_y = max(point[1] for point in poses)
        center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        radius = max(
            65.0,
            max(max_x - min_x, max_y - min_y) / 2.0 + 45.0)
        return LaneWorldRenderer(
            self.traffic_mgr, Viewport.around(center, radius),
            width=self._audit_width, height=self._audit_height,
            vehicle_marker_radius_px=5,
            title=self.scenario.road_network_id)

    def _log_per_substep(self, physics_time, tick_index, trigger_events):
        if physics_time + 1e-7 < self._audit_next_time_s:
            return
        if self._audit_renderer is None:
            self._audit_renderer = self._make_renderer()
        self._audit_frames.append(
            self._audit_renderer.capture(float(physics_time)))
        self._audit_next_time_s = round(
            float(physics_time) + self._audit_interval_s, 6)

    def recording(self) -> WorldRecording:
        if self._audit_renderer is None or not self._audit_frames:
            raise ValueError("visual audit captured no frames")
        return WorldRecording(
            network_id=self.scenario.road_network_id,
            viewport=self._audit_renderer.viewport,
            frame_interval_s=self._audit_interval_s,
            frames=self._audit_frames)


def _polygon_area(points: List[List[float]]) -> float:
    return abs(sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])) / 2.0)


def _decode_gif(path: Path) -> dict:
    images = []
    durations = []
    with Image.open(path) as source:
        size = source.size
        frame_count = int(getattr(source, "n_frames", 1))
        for index in range(frame_count):
            source.seek(index)
            images.append(source.convert("RGB"))
            durations.append(int(source.info.get("duration", 0)))
    changed_pixels = []
    for first, second in zip(images, images[1:]):
        difference = ImageChops.difference(first, second)
        changed_pixels.append(sum(
            1 for value in difference.convert("L").get_flattened_data()
            if value))
    variance = sum(ImageStat.Stat(images[0]).var) if images else 0.0
    return {
        "frame_count": len(images),
        "size_px": list(size),
        "durations_ms": durations,
        "changed_pixels": changed_pixels,
        "first_frame_variance": round(float(variance), 3),
    }


def _audit_recording(recording: WorldRecording, gif_info: dict,
                     expected_vehicle_ids: Iterable[str]) -> dict:
    failures = []
    expected_ids = sorted(expected_vehicle_ids)
    frame_checks = []
    tracks: Dict[str, List[List[float]]] = {item: [] for item in expected_ids}
    viewport = recording.viewport
    for index, frame in enumerate(recording.frames):
        ids = sorted(vehicle["id"] for vehicle in frame.vehicles)
        frame_failures = []
        if ids != expected_ids:
            frame_failures.append(
                f"vehicle_ids={ids}, expected={expected_ids}")
        if frame.collisions:
            frame_failures.append("collision_present")
        for vehicle in frame.vehicles:
            pose = vehicle["pose"]
            corners = vehicle["corners"]
            if not all(math.isfinite(float(value)) for value in pose):
                frame_failures.append(f"{vehicle['id']}: non_finite_pose")
            if _polygon_area(corners) <= 0.5:
                frame_failures.append(f"{vehicle['id']}: invalid_body_domain")
            if not all(
                    viewport.min_x <= point[0] <= viewport.max_x
                    and viewport.min_y <= point[1] <= viewport.max_y
                    for point in corners):
                frame_failures.append(f"{vehicle['id']}: body_cropped")
            tracks.setdefault(vehicle["id"], []).append(pose[:2])
        if frame_failures:
            failures.extend(
                f"frame_{index}: {item}" for item in frame_failures)
        frame_checks.append({
            "index": index,
            "time_s": frame.time_s,
            "vehicle_count": len(frame.vehicles),
            "collision_count": len(frame.collisions),
            "status": "passed" if not frame_failures else "failed",
        })
    motion = {
        vehicle_id: round(math.dist(points[0], points[-1]), 4)
        if len(points) >= 2 else 0.0
        for vehicle_id, points in tracks.items()}
    if max(motion.values(), default=0.0) < 0.25:
        failures.append("no_visible_vehicle_motion")
    if gif_info["frame_count"] != len(recording.frames):
        failures.append("gif_frame_count_mismatch")
    if gif_info["first_frame_variance"] < 10.0:
        failures.append("blank_or_nearly_uniform_frame")
    if (gif_info["changed_pixels"]
            and any(value <= 0 for value in gif_info["changed_pixels"])):
        failures.append("duplicate_adjacent_rendered_frame")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "frame_count": len(recording.frames),
        "frame_checks": frame_checks,
        "vehicle_motion_m": motion,
        "gif_decode": gif_info,
    }


def _render_one(payload: dict) -> dict:
    logging.disable(logging.CRITICAL)
    scenario_path = Path(payload["scenario_path"])
    output_dir = Path(payload["output_dir"])
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    network_id = scenario["road_network_id"]
    scenario["total_time_s"] = float(payload["duration_s"])
    scenario["tick_interval_s"] = float(payload["duration_s"])
    scenario["physics_step_s"] = 0.1
    for vehicle in scenario.get("vehicles", []):
        vehicle["agent_config"] = {"type": "sumo"}
    engine = _VisualAuditEngine(
        MultiScenario.from_dict(copy.deepcopy(scenario)),
        frame_interval_s=float(payload["frame_interval_s"]),
        width=int(payload["width"]), height=int(payload["height"]))
    engine.run({})
    recording = engine.recording()
    gif_path = output_dir / "gifs" / f"{network_id}.gif"
    recording_path = output_dir / "recordings" / f"{network_id}.json"
    audit_path = output_dir / "maps" / f"{network_id}.json"
    engine._audit_renderer.render_gif(
        recording, str(gif_path), show_labels=True)
    recording.save_json(str(recording_path))
    gif_info = _decode_gif(gif_path)
    audit = _audit_recording(
        recording, gif_info,
        [vehicle["vehicle_id"] for vehicle in scenario["vehicles"]])
    result = {
        "schema": VISUAL_AUDIT_SCHEMA,
        "network_id": network_id,
        "scenario_id": scenario["scenario_id"],
        "status": audit["status"],
        "gif": str(Path("gifs") / gif_path.name),
        "recording": str(Path("recordings") / recording_path.name),
        **audit,
    }
    _write_json(audit_path, result)
    return result


def _contact_sheets(results: List[dict], output_dir: Path,
                    maps_per_sheet: int = 4) -> List[str]:
    output = []
    font = ImageFont.load_default(size=12)
    thumb = 96
    label_width = 170
    row_height = thumb + 24
    for sheet_index, start in enumerate(
            range(0, len(results), maps_per_sheet), 1):
        group = results[start:start + maps_per_sheet]
        decoded = []
        max_frames = 0
        for result in group:
            frames = []
            with Image.open(output_dir / result["gif"]) as gif:
                for index in range(gif.n_frames):
                    gif.seek(index)
                    image = gif.convert("RGB")
                    image.thumbnail((thumb, thumb))
                    tile = Image.new("RGB", (thumb, thumb), (12, 20, 28))
                    tile.paste(
                        image, ((thumb - image.width) // 2,
                                (thumb - image.height) // 2))
                    frames.append(tile)
            max_frames = max(max_frames, len(frames))
            decoded.append((result, frames))
        sheet = Image.new(
            "RGB", (label_width + max_frames * thumb,
                    len(group) * row_height), (18, 28, 38))
        draw = ImageDraw.Draw(sheet)
        for row, (result, frames) in enumerate(decoded):
            y = row * row_height
            draw.text((8, y + 10), result["network_id"],
                      font=font, fill=(245, 249, 252))
            draw.text((8, y + 30), f"{len(frames)} frames / {result['status']}",
                      font=font, fill=(110, 220, 150))
            for index, frame in enumerate(frames):
                sheet.paste(frame, (label_width + index * thumb, y))
                draw.text(
                    (label_width + index * thumb + 3, y + thumb + 3),
                    f"f{index:02d}", font=font, fill=(185, 199, 208))
        relative = Path("contact_sheets") / f"sheet_{sheet_index:02d}.png"
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        output.append(str(relative))
    return output


def render_map_visual_audit(
    catalog_dir: Path, output_dir: Path, *, workers: int = 1,
    network_ids: Iterable[str] | None = None, duration_s: float = 1.0,
    frame_interval_s: float = 0.1, width: int = 640, height: int = 640,
) -> dict:
    """Render, decode and audit one intersection GIF for every selected map."""
    catalog_dir = Path(catalog_dir)
    output_dir = Path(output_dir)
    catalog = json.loads(
        (catalog_dir / "catalog.json").read_text(encoding="utf-8"))
    selected = set(network_ids or ())
    entries = [entry for entry in catalog["entries"]
               if entry["kind"] == "intersection_conflict"
               and (not selected or entry["network_id"] in selected)]
    found = {entry["network_id"] for entry in entries}
    if selected - found:
        raise ValueError(
            f"unknown or uncovered networks: {sorted(selected - found)}")
    payloads = [{
        "scenario_path": str(catalog_dir / entry["scenario"]),
        "output_dir": str(output_dir),
        "duration_s": duration_s,
        "frame_interval_s": frame_interval_s,
        "width": width,
        "height": height,
    } for entry in entries]
    results = []
    if workers <= 1:
        for payload in payloads:
            results.append(_render_one(payload))
    else:
        with ProcessPoolExecutor(
                max_workers=workers, max_tasks_per_child=1) as pool:
            future_payloads = {
                pool.submit(_render_one, payload): payload
                for payload in payloads}
            for future in as_completed(future_payloads):
                payload = future_payloads[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    scenario = json.loads(Path(
                        payload["scenario_path"]).read_text(encoding="utf-8"))
                    results.append({
                        "schema": VISUAL_AUDIT_SCHEMA,
                        "network_id": scenario["road_network_id"],
                        "scenario_id": scenario["scenario_id"],
                        "status": "failed",
                        "failures": [
                            f"render: {type(exc).__name__}: {exc}"],
                    })
    results.sort(key=lambda item: item["network_id"])
    passed_results = [item for item in results if item["status"] == "passed"]
    sheets = _contact_sheets(passed_results, output_dir)
    summary = {
        "schema": VISUAL_AUDIT_SCHEMA,
        "map_count": len(results),
        "passed_maps": len(passed_results),
        "failed_maps": len(results) - len(passed_results),
        "frame_count": sum(item.get("frame_count", 0) for item in results),
        "contact_sheets": sheets,
        "maps": [{key: item.get(key) for key in (
            "network_id", "status", "gif", "recording", "frame_count",
            "failures")} for item in results],
    }
    _write_json(output_dir / "summary.json", summary)
    lines = [
        "# VehicleArena 全地图车辆 GIF 索引", "",
        f"地图：{len(results)}；通过：{len(passed_results)}；"
        f"逐帧记录：{summary['frame_count']}。", "",
    ]
    for item in results:
        gif = item.get("gif", "")
        lines.append(
            f"- `{item['network_id']}` — {item['status']}"
            + (f" — [{gif}]({gif})" if gif else ""))
    _write_text(output_dir / "README.md", "\n".join(lines))
    return summary
