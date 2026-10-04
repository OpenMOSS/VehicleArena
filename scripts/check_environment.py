#!/usr/bin/env python3
"""Check whether a source checkout can run VehicleArena end to end."""

from __future__ import annotations

import importlib
import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "vehiclearena"))

from scripts.manage_map_bundle import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_MAP_DIR,
    map_status,
)


REFERENCE_SUMO_VERSION = "1.25.0"
PYTHON_MODULES = (
    "RestrictedPython",
    "PIL",
    "openai",
    "transformers",
    "yaml",
    "libsumo",
    "traci",
    "sumolib",
)


def _sumo_version(command: str) -> str:
    completed = subprocess.run(
        [command, "--version"], check=False, capture_output=True,
        text=True, timeout=20,
    )
    output = "\n".join((completed.stdout, completed.stderr)).strip()
    match = re.search(r"\bsumo\s+(\d+\.\d+\.\d+)\b", output, re.IGNORECASE)
    return match.group(1) if match else ""


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    modules = {}
    for name in PYTHON_MODULES:
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                module = importlib.import_module(name)
            modules[name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:
            modules[name] = None
            errors.append(f"missing Python module {name}: {exc}")

    commands = {name: shutil.which(name) for name in ("sumo", "netconvert")}
    for name, path in commands.items():
        if path is None:
            errors.append(f"missing executable: {name}")

    installed_sumo = _sumo_version(commands["sumo"]) if commands["sumo"] else ""
    if installed_sumo and installed_sumo != REFERENCE_SUMO_VERSION:
        warnings.append(
            f"SUMO {installed_sumo} can be used for development, but frozen "
            f"benchmark references use {REFERENCE_SUMO_VERSION}"
        )

    maps = map_status(DEFAULT_MAP_DIR, DEFAULT_MANIFEST)
    if not maps.get("complete"):
        errors.append(
            "road maps are incomplete; install vehiclearena-road-networks.tar.gz"
        )

    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required")

    report = {
        "ready": not errors,
        "python": sys.version.split()[0],
        "sumo": {
            "installed": installed_sumo or None,
            "benchmark_reference": REFERENCE_SUMO_VERSION,
            "commands": commands,
        },
        "maps": maps,
        "python_modules": modules,
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
