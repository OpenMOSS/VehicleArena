"""Small deterministic statistics helpers used by experiment aggregation."""

from __future__ import annotations

import math
from typing import Iterable


def percentile(values: Iterable[float], q: float):
    data = sorted(float(value) for value in values if value is not None)
    if not data:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    position = (len(data) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return data[lower]
    return data[lower] + (data[upper] - data[lower]) * (position - lower)


def gini(values: Iterable[float]):
    data = sorted(max(0.0, float(value)) for value in values
                  if value is not None)
    if not data:
        return None
    total = sum(data)
    if total <= 0.0:
        return 0.0
    weighted = sum((index + 1) * value
                   for index, value in enumerate(data))
    return (2.0 * weighted) / (len(data) * total) - (
        len(data) + 1.0) / len(data)
