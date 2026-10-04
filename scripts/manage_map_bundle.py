#!/usr/bin/env python3
"""Package, install, and inspect VehicleArena's separately distributed maps."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_DIR = (
    REPOSITORY_ROOT / "vehiclearena" / "simulation" / "road_networks"
)
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / "vehiclearena" / "simulation" / "map_bundle_manifest.json"
)
EMBEDDED_MANIFEST_NAME = "map_bundle.json"
SCHEMA_VERSION = 1
CHUNK_SIZE = 1024 * 1024


class MapBundleError(RuntimeError):
    """Raised when a map bundle is missing, invalid, or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_files(map_dir: Path) -> List[Path]:
    files = sorted(
        path for path in map_dir.glob("*.json")
        if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise MapBundleError(f"No map JSON files found in {map_dir}")
    return files


def _file_records(files: Iterable[Path]) -> List[dict]:
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def _embedded_manifest(records: List[dict]) -> dict:
    base_networks = sorted(
        Path(record["name"]).stem
        for record in records
        if not record["name"].endswith("_lane_level.json")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "format": "vehiclearena-road-network-bundle",
        "map_file_count": len(records),
        "lane_level_count": sum(
            record["name"].endswith("_lane_level.json")
            for record in records
        ),
        "uncompressed_bytes": sum(record["bytes"] for record in records),
        "base_networks": base_networks,
        "files": records,
    }


def _tar_info(path: Path, arcname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    info.size = path.stat().st_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def package_maps(map_dir: Path, output: Path, manifest_path: Path) -> dict:
    """Create a deterministic gzip tar archive and its public manifest."""
    map_dir = map_dir.resolve()
    output = output.resolve()
    manifest_path = manifest_path.resolve()
    files = _json_files(map_dir)
    embedded = _embedded_manifest(_file_records(files))
    embedded_bytes = (
        json.dumps(embedded, ensure_ascii=False, indent=2, sort_keys=True)
        .encode("utf-8") + b"\n"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp",
        dir=str(output.parent), delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="", mode="wb", compresslevel=6,
                fileobj=raw_stream, mtime=0,
            ) as gzip_stream:
                with tarfile.open(fileobj=gzip_stream, mode="w") as archive:
                    manifest_info = tarfile.TarInfo(EMBEDDED_MANIFEST_NAME)
                    manifest_info.size = len(embedded_bytes)
                    manifest_info.mode = 0o644
                    manifest_info.mtime = 0
                    archive.addfile(manifest_info, io.BytesIO(embedded_bytes))
                    for path in files:
                        info = _tar_info(path, f"road_networks/{path.name}")
                        with path.open("rb") as source:
                            archive.addfile(info, source)
        os.replace(temporary, output)
        output.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()

    public_manifest = dict(embedded)
    public_manifest.pop("files")
    public_manifest.update({
        "archive_name": output.name,
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
    })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(public_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o644)
    return public_manifest


def _read_public_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MapBundleError(f"Cannot read bundle manifest {path}: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise MapBundleError(
            f"Unsupported manifest schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    return payload


def _obtain_archive(source: str, temporary_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https"}:
        destination = temporary_dir / "map_bundle.tar.gz"
        try:
            with urllib.request.urlopen(source) as response, destination.open("wb") as out:
                shutil.copyfileobj(response, out, length=CHUNK_SIZE)
        except OSError as exc:
            raise MapBundleError(f"Failed to download map bundle: {exc}") from exc
        return destination
    if parsed.scheme == "file":
        path = Path(urllib.request.url2pathname(parsed.path))
    else:
        path = Path(source).expanduser()
    if not path.is_file():
        raise MapBundleError(f"Map bundle not found: {path}")
    return path.resolve()


def _validate_archive(archive: tarfile.TarFile) -> Tuple[dict, Dict[str, tarfile.TarInfo]]:
    members = archive.getmembers()
    by_name = {member.name: member for member in members}
    if len(by_name) != len(members):
        raise MapBundleError("Map bundle contains duplicate archive paths")
    manifest_info = by_name.get(EMBEDDED_MANIFEST_NAME)
    if manifest_info is None or not manifest_info.isfile():
        raise MapBundleError("Map bundle has no embedded manifest")
    manifest_stream = archive.extractfile(manifest_info)
    if manifest_stream is None:
        raise MapBundleError("Cannot read embedded map manifest")
    try:
        embedded = json.loads(manifest_stream.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MapBundleError(f"Invalid embedded map manifest: {exc}") from exc
    if embedded.get("schema_version") != SCHEMA_VERSION:
        raise MapBundleError("Unsupported embedded map manifest schema")

    records = embedded.get("files")
    if not isinstance(records, list) or not records:
        raise MapBundleError("Embedded map manifest contains no files")
    expected_names = {f"road_networks/{record.get('name', '')}" for record in records}
    actual_names = {name for name in by_name if name != EMBEDDED_MANIFEST_NAME}
    if expected_names != actual_names:
        raise MapBundleError("Archive contents do not match the embedded manifest")
    for name in actual_names:
        member = by_name[name]
        path = Path(name)
        if (
            not member.isfile()
            or path.parent != Path("road_networks")
            or path.suffix != ".json"
            or path.name in {"", ".", ".."}
        ):
            raise MapBundleError(f"Unsafe map bundle member: {name!r}")
    return embedded, by_name


def install_maps(
    source: str,
    target: Path,
    manifest_path: Optional[Path],
    expected_sha256: Optional[str],
    replace: bool,
) -> dict:
    """Verify and install a local or remote bundle into the map data directory."""
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the final directory.  Map bundles can exceed small tmpfs
    # mounts, and a same-filesystem staging area keeps the final os.replace()
    # operations atomic.
    with tempfile.TemporaryDirectory(
        prefix=".vehiclearena-maps-", dir=str(target.parent)
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        archive_path = _obtain_archive(source, temporary_dir)
        if manifest_path is not None:
            public_manifest = _read_public_manifest(manifest_path)
            manifest_sha = public_manifest.get("archive_sha256")
            if expected_sha256 and manifest_sha != expected_sha256:
                raise MapBundleError(
                    "--sha256 disagrees with the selected public manifest"
                )
            expected_sha256 = manifest_sha
        actual_sha256 = sha256_file(archive_path)
        if not expected_sha256:
            raise MapBundleError(
                "A checksum is required; use --manifest or --sha256"
            )
        if actual_sha256.lower() != str(expected_sha256).lower():
            raise MapBundleError(
                f"Map bundle SHA-256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )

        staging = temporary_dir / "road_networks"
        staging.mkdir()
        with tarfile.open(archive_path, mode="r:gz") as archive:
            embedded, members = _validate_archive(archive)
            records = embedded["files"]
            for record in records:
                name = record["name"]
                source_stream = archive.extractfile(members[f"road_networks/{name}"])
                if source_stream is None:
                    raise MapBundleError(f"Cannot read archived map {name}")
                destination = staging / name
                digest = hashlib.sha256()
                with destination.open("wb") as output:
                    while True:
                        chunk = source_stream.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        digest.update(chunk)
                        output.write(chunk)
                if destination.stat().st_size != record["bytes"]:
                    raise MapBundleError(f"Size mismatch while extracting {name}")
                if digest.hexdigest() != record["sha256"]:
                    raise MapBundleError(f"SHA-256 mismatch while extracting {name}")

        target.mkdir(parents=True, exist_ok=True)
        existing = sorted(target.glob("*.json"))
        if existing and not replace:
            raise MapBundleError(
                f"{target} already contains {len(existing)} map files; "
                "use --replace to install a complete bundle"
            )
        if replace:
            for path in existing:
                if path.is_file() or path.is_symlink():
                    path.unlink()
        for path in sorted(staging.glob("*.json")):
            os.replace(path, target / path.name)
        return {
            "archive_sha256": actual_sha256,
            "installed_files": len(records),
            "lane_level_count": embedded["lane_level_count"],
            "target": str(target),
        }


def map_status(target: Path, manifest_path: Optional[Path]) -> dict:
    files = sorted(target.glob("*.json")) if target.is_dir() else []
    status = {
        "target": str(target.resolve()),
        "installed_files": len(files),
        "lane_level_count": sum(
            path.name.endswith("_lane_level.json") for path in files
        ),
        "installed_bytes": sum(path.stat().st_size for path in files),
    }
    if manifest_path is not None and manifest_path.is_file():
        expected = _read_public_manifest(manifest_path)
        status["expected_files"] = expected.get("map_file_count")
        status["expected_lane_level_count"] = expected.get("lane_level_count")
        status["complete"] = (
            status["installed_files"] == status["expected_files"]
            and status["lane_level_count"] == status["expected_lane_level_count"]
        )
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("pack", help="build a separately distributed map bundle")
    pack.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--manifest", type=Path, required=True)

    install = subparsers.add_parser("install", help="verify and install a map bundle")
    install.add_argument("--source", required=True, help="local path, file:// URL, or HTTPS URL")
    install.add_argument("--target", type=Path, default=DEFAULT_MAP_DIR)
    install.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    install.add_argument("--sha256")
    install.add_argument("--replace", action="store_true")

    status = subparsers.add_parser("status", help="show installed map data status")
    status.add_argument("--target", type=Path, default=DEFAULT_MAP_DIR)
    status.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "pack":
            result = package_maps(args.map_dir, args.output, args.manifest)
        elif args.command == "install":
            result = install_maps(
                args.source, args.target, args.manifest,
                args.sha256, args.replace,
            )
        else:
            result = map_status(args.target, args.manifest)
    except MapBundleError as exc:
        print(f"map bundle error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
