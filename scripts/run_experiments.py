#!/usr/bin/env python3
"""Run the experiment CLI from a source checkout without PYTHONPATH setup."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "vehiclearena"))

from evaluation.experiments.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
