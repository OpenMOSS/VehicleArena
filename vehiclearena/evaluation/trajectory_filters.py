"""Small deterministic filters shared by trajectory evaluators."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


def windowed_longitudinal_acceleration(
    history: Sequence[Tuple[float, float]],
    time_s: float,
    speed_kmh: float,
    window_s: float,
) -> Optional[float]:
    """Estimate acceleration over a causal speed window.

    ``history`` contains prior ``(time_s, speed_kmh)`` observations. A value
    is returned only after a complete window is available. Using a one-second
    physical speed delta prevents 10 Hz car-following noise from becoming
    fictitious chassis jerk while preserving real acceleration transitions.
    """
    window = float(window_s)
    if window <= 0.0 or not history:
        return None
    cutoff = float(time_s) - window
    anchor = None
    for sample_time_s, sample_speed_kmh in reversed(history):
        if float(sample_time_s) <= cutoff + 1e-9:
            anchor = (float(sample_time_s), float(sample_speed_kmh))
            break
    if anchor is None:
        return None
    elapsed = float(time_s) - anchor[0]
    if elapsed + 1e-9 < window:
        return None
    return ((float(speed_kmh) - anchor[1]) / 3.6) / elapsed
