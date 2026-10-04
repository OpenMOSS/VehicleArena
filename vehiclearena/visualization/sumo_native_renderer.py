"""Reusable SUMO-GUI screenshot and interval-GIF renderer.

This renderer captures the same live SUMO instance that advances the traffic
world. It is intended for whole-map inspection and physics debugging; LLM
camera observations come from the synchronized Web3D cockpit, with a separate
local geometry view for vehicles equipped with LiDAR.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image

from visualization.lane_world_renderer import Viewport


@dataclass(frozen=True)
class SumoRenderResult:
    """Metadata returned by a native SUMO render operation."""

    output: str
    start_time_s: float
    end_time_s: float
    frame_interval_s: float
    frame_count: int
    width: int
    height: int


class SumoNativeRenderer:
    """Capture PNG/GIF output from a live GUI-enabled SUMO engine."""

    def __init__(
        self,
        traffic_manager: Any,
        viewport: Optional[Viewport] = None,
        *,
        width: int = 960,
        height: int = 960,
    ):
        if getattr(traffic_manager, "engine_name", "") != "sumo":
            raise ValueError("SumoNativeRenderer requires the SUMO engine")
        if not getattr(traffic_manager, "native_rendering_enabled", False):
            raise ValueError(
                "SumoNativeRenderer requires sumo_config.gui=true")
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("render width and height must be positive")
        self.manager = traffic_manager
        self.width = int(width)
        self.height = int(height)
        lane_map = traffic_manager._lane_geometry.data
        self.viewport = viewport or Viewport.full_map(lane_map)

    def set_viewport(self, viewport: Viewport) -> None:
        self.viewport = viewport

    @staticmethod
    def _require_physics_grid(value: float, name: str) -> None:
        if abs(round(float(value) / 0.1) * 0.1 - float(value)) > 1e-7:
            raise ValueError(f"{name} must align to the 0.1-second physics grid")

    def _ensure_bootstrapped(self) -> None:
        if not self.manager._initial_state_bootstrapped:
            self.manager.advance_world_to(self.manager._physics_time)

    def _advance_to(self, target_time_s: float) -> None:
        if target_time_s < self.manager._physics_time - 1e-7:
            raise ValueError(
                "Cannot render past state from a live SUMO episode")
        while self.manager._physics_time < target_time_s - 1e-7:
            now = self.manager._physics_time
            next_time = min(target_time_s, now + 0.1)
            self.manager.recalculate_speeds(now)
            self.manager.advance_world_to(next_time)

    def _advance_and_capture(
        self, target_time_s: float, path: str | os.PathLike[str],
    ) -> float:
        """Queue a frame, then advance exactly once into its timestamp."""
        self._ensure_bootstrapped()
        target = float(target_time_s)
        current = float(self.manager._physics_time)
        if target < current - 1e-7:
            raise ValueError("Cannot render past state from a live SUMO episode")
        if target <= current + 1e-7 and current <= 1e-7:
            target = self.manager._physics_time + 0.1
        elif target <= current + 1e-7:
            raise ValueError(
                "SUMO screenshots must be queued before their simulation step")
        while self.manager._physics_time < target - 1e-7:
            now = self.manager._physics_time
            next_time = min(target, now + 0.1)
            self.manager.recalculate_speeds(now)
            if next_time >= target - 1e-7:
                output = self.manager.request_native_screenshot(
                    path,
                    viewport=self.viewport,
                    width=self.width,
                    height=self.height,
                )
            self.manager.advance_world_to(next_time)
        self.manager.wait_for_native_screenshot(output)
        return float(self.manager._physics_time)

    def render_png_at(
        self, time_s: float, path: str | os.PathLike[str],
    ) -> SumoRenderResult:
        """Advance to ``time_s`` and capture one native SUMO PNG."""
        target = float(time_s)
        if target < 0:
            raise ValueError("render time must be non-negative")
        self._require_physics_grid(target, "render time")
        actual_time = self._advance_and_capture(target, path)
        output = str(Path(path).resolve())
        return SumoRenderResult(
            output=output,
            start_time_s=actual_time,
            end_time_s=actual_time,
            frame_interval_s=0.0,
            frame_count=1,
            width=self.width,
            height=self.height,
        )

    def render_gif(
        self,
        start_time_s: float,
        end_time_s: float,
        frame_interval_s: float,
        path: str | os.PathLike[str],
        *,
        loop: int = 0,
    ) -> SumoRenderResult:
        """Capture a native SUMO frame sequence and encode it as GIF."""
        start = float(start_time_s)
        end = float(end_time_s)
        interval = float(frame_interval_s)
        if start < 0 or end < start:
            raise ValueError("render interval must satisfy 0 <= start <= end")
        if interval < 0.1 - 1e-9:
            raise ValueError("frame_interval_s must be at least 0.1 seconds")
        self._require_physics_grid(start, "start_time_s")
        self._require_physics_grid(end, "end_time_s")
        self._require_physics_grid(interval, "frame_interval_s")
        self._ensure_bootstrapped()
        current = float(self.manager._physics_time)
        if start < current - 1e-7:
            raise ValueError("Cannot render past state from a live SUMO episode")
        if start <= current + 1e-7 and current <= 1e-7:
            # SUMO creates GUI-visible vehicles on its first step. The earliest
            # exact native frame is consequently public t=0.1 s.
            start = self.manager._physics_time + 0.1
        elif start <= current + 1e-7:
            raise ValueError(
                "SUMO screenshots must be queued before their simulation step")
        end = max(end, start)
        self._advance_to(start - 0.1)
        times: List[float] = [start]
        frame_time = start + interval
        while frame_time <= end + 1e-7:
            times.append(min(frame_time, end))
            frame_time += interval
        if times[-1] < end - 1e-7:
            times.append(end)

        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        indexed: List[Image.Image] = []
        shared: Optional[Image.Image] = None
        with tempfile.TemporaryDirectory(prefix="vehiclearena-sumo-frames-") as root:
            for index, time_s in enumerate(times):
                frame_path = Path(root) / f"frame-{index:06d}.png"
                self._advance_and_capture(time_s, frame_path)
                with Image.open(frame_path) as frame:
                    rgb = frame.convert("RGB")
                    if shared is None:
                        shared = rgb.convert(
                            "P", palette=Image.Palette.ADAPTIVE, colors=128)
                        indexed.append(shared)
                    else:
                        indexed.append(rgb.quantize(
                            palette=shared, dither=Image.Dither.NONE))
        indexed[0].save(
            output,
            save_all=True,
            append_images=indexed[1:],
            duration=max(20, round(interval * 1000)),
            loop=int(loop),
            optimize=False,
            disposal=1,
        )
        return SumoRenderResult(
            output=str(output),
            start_time_s=start,
            end_time_s=times[-1],
            frame_interval_s=interval,
            frame_count=len(times),
            width=self.width,
            height=self.height,
        )
