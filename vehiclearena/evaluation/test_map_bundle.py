"""Regression tests for separately distributed road-network data."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "manage_map_bundle.py"
SPEC = importlib.util.spec_from_file_location("manage_map_bundle", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
manage_map_bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manage_map_bundle)

VEHICLEARENA_ROOT = REPOSITORY_ROOT / "vehiclearena"
if str(VEHICLEARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(VEHICLEARENA_ROOT))
from simulation import road_networks  # noqa: E402


class MapBundleTest(unittest.TestCase):
    def test_catalog_is_available_without_installed_json(self) -> None:
        self.assertEqual(len(road_networks.list_available()), 116)
        self.assertFalse(any(
            name.endswith("_lane_level")
            for name in road_networks.list_available()
        ))
        with tempfile.TemporaryDirectory() as temporary_name:
            original_directory = road_networks._NETWORK_DIR
            road_networks._NETWORK_DIR = temporary_name
            try:
                with self.assertRaisesRegex(
                    ValueError, "distributed separately from Git"
                ):
                    road_networks.load_road_network("beijing_guomao")
            finally:
                road_networks._NETWORK_DIR = original_directory

    def test_pack_verify_install_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            source = root / "source"
            source.mkdir()
            payloads = {
                "test_city.json": {"nodes": {"n1": {"lat": 1.0}}},
                "test_city_lane_level.json": {"lanes": [{"id": "lane-1"}]},
            }
            for name, payload in payloads.items():
                (source / name).write_text(
                    json.dumps(payload), encoding="utf-8")

            archive = root / "maps.tar.gz"
            manifest = root / "maps.manifest.json"
            packaged = manage_map_bundle.package_maps(
                source, archive, manifest)

            self.assertEqual(packaged["map_file_count"], 2)
            self.assertEqual(packaged["lane_level_count"], 1)
            self.assertEqual(packaged["base_networks"], ["test_city"])
            self.assertEqual(
                packaged["archive_sha256"],
                manage_map_bundle.sha256_file(archive),
            )

            target = root / "installed"
            installed = manage_map_bundle.install_maps(
                str(archive), target, manifest, None, False)
            self.assertEqual(installed["installed_files"], 2)
            for name, payload in payloads.items():
                self.assertEqual(
                    json.loads((target / name).read_text(encoding="utf-8")),
                    payload,
                )
            status = manage_map_bundle.map_status(target, manifest)
            self.assertTrue(status["complete"])

    def test_checksum_mismatch_does_not_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            source = root / "source"
            source.mkdir()
            (source / "test.json").write_text("{}", encoding="utf-8")
            archive = root / "maps.tar.gz"
            manifest = root / "maps.manifest.json"
            manage_map_bundle.package_maps(source, archive, manifest)
            target = root / "installed"

            with self.assertRaises(manage_map_bundle.MapBundleError):
                manage_map_bundle.install_maps(
                    str(archive), target, None, "0" * 64, False)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
