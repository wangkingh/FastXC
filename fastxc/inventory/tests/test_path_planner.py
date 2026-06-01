import struct
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fastxc.inventory.planner import (
    build_path_plan,
    filter_group_by_path_plan,
    station_time_rows_from_group,
    write_path_plan,
    write_timestamp_manifests,
)


HEADER_BYTES = 632
NVHDR_OFFSET = 70 * 4 + 6 * 4
STLA_OFFSET = 31 * 4
STLO_OFFSET = 32 * 4
STEL_OFFSET = 33 * 4


class TestPathPlanner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.timestamp = datetime(2023, 1, 1)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_builds_four_digit_nsl_ids_and_eight_digit_path_ids(self):
        files_group = {
            ("STA01", self.timestamp): self._station_info("STA01", 0.0, 0.0),
            ("STA02", self.timestamp): self._station_info("STA02", 0.0, 0.1),
            ("STA03", self.timestamp): self._station_info("STA03", 10.0, 10.0),
        }

        plan = build_path_plan(
            files_group,
            distance_range="0/20",
            azimuth_range="-1/360",
            allow_autocorr=False,
        )

        self.assertEqual([node.id_text for node in plan.nodes], ["0001", "0002", "0003"])
        self.assertEqual([path.path_id_text for path in plan.paths], ["00010002"])
        self.assertEqual(plan.allowed_path_ids, {10002})

        filtered = filter_group_by_path_plan(files_group, "1", plan)
        self.assertEqual(set(key[0] for key in filtered), {"STA01", "STA02"})
        stations, times = station_time_rows_from_group(filtered)
        self.assertEqual(stations, ["STA01", "STA02"])
        self.assertEqual(times, [self.timestamp, self.timestamp])

    def test_default_plan_keeps_single_station_autocorr(self):
        files_group = {
            ("STA01", self.timestamp): self._station_info("STA01", 0.0, 0.0),
        }

        plan = build_path_plan(files_group)

        self.assertEqual(len(plan.nodes), 1)
        self.assertEqual([path.path_id_text for path in plan.paths], ["00010001"])
        self.assertEqual(plan.paths[0].distance_km, 0.0)
        self.assertEqual(plan.paths[0].azimuth_deg, 0.0)
        self.assertEqual(plan.paths[0].back_azimuth_deg, 0.0)

    def test_writes_path_plan_files(self):
        files_group = {
            ("STA01", self.timestamp): self._station_info("STA01", 0.0, 0.0),
            ("STA02", self.timestamp): self._station_info("STA02", 0.0, 0.1),
        }
        plan = build_path_plan(files_group, allow_autocorr=False)

        output_dir = self.root / "path_plan"
        write_path_plan(plan, output_dir)

        self.assertTrue((output_dir / "nsl_catalog.tsv").is_file())
        self.assertTrue((output_dir / "allowed_paths.tsv").is_file())
        path_lines = (output_dir / "allowed_paths.tsv").read_text().splitlines()
        self.assertIn("great_circle_deg", path_lines[0])
        self.assertIn("distance_km", path_lines[0])
        self.assertNotIn("\t\t", path_lines[1])
        self.assertEqual(
            (output_dir / "allowed_path_ids.txt").read_text().strip(),
            "00010002",
        )

    def test_external_geo_tsv_overrides_sac_header_for_filtering(self):
        files_group = {
            ("STA01", self.timestamp): self._station_info("STA01", 0.0, 0.0),
            ("STA02", self.timestamp): self._station_info("STA02", 30.0, 30.0),
        }
        external_geo = self.root / "external_geo.tsv"
        external_geo.write_text(
            "station\tlat\tlon\tele\n"
            "STA01\t0.0\t0.0\t1.0\n"
            "STA02\t0.0\t0.1\t2.0\n"
        )

        plan = build_path_plan(
            files_group,
            distance_range="0/20",
            azimuth_range="-1/360",
            allow_autocorr=False,
            external_geo_tsv_path=external_geo,
        )

        self.assertEqual(plan.external_geo_row_count, 2)
        self.assertEqual(plan.external_geo_node_fill_count, 1)
        self.assertEqual(plan.external_geo_conflict_node_count, 1)
        self.assertEqual([path.path_id_text for path in plan.paths], ["00010002"])
        self.assertAlmostEqual(plan.nodes[1].lat, 0.0)
        self.assertAlmostEqual(plan.nodes[1].lon, 0.1)

    def test_writes_timestamp_manifest_without_repeating_timestamp(self):
        files_group = {
            ("STA01", self.timestamp): self._station_info("STA01", 0.0, 0.0),
            ("STA02", self.timestamp): self._station_info("STA02", 0.0, 0.1),
        }
        plan = build_path_plan(files_group, allow_autocorr=False)

        manifest_root = self.root / "manifest" / "seisarray1"
        write_timestamp_manifests(files_group, "1", plan, manifest_root)

        index_text = (manifest_root / "timestamp_index.tsv").read_text()
        self.assertIn("2023.001.0000\tby_timestamp/2023.001.0000.tsv\t6", index_text)

        list_path = manifest_root / "timestamp_manifest_list.txt"
        list_lines = list_path.read_text().splitlines()
        self.assertEqual(len(list_lines), 1)
        self.assertEqual(
            Path(list_lines[0]),
            (manifest_root / "by_timestamp" / "2023.001.0000.tsv").resolve(),
        )

        shard = manifest_root / "by_timestamp" / "2023.001.0000.tsv"
        lines = shard.read_text().splitlines()
        self.assertEqual(
            lines[0],
            "nsl_id\tgroup\tnetwork\tstation\tlocation\tcomponent\tsac_path",
        )
        self.assertEqual(len(lines), 7)
        self.assertNotIn("timestamp", lines[0])

    def test_group_pair_mode_intra_inter_all(self):
        group1 = {
            ("STA01", self.timestamp, "XX", "00"): self._station_info("STA01", 0.0, 0.0),
            ("STA02", self.timestamp, "XX", "00"): self._station_info("STA02", 0.0, 0.1),
        }
        group2 = {
            ("STA03", self.timestamp, "YY", "00"): self._station_info("STA03", 0.0, 0.2),
        }

        intra = build_path_plan(
            files_groups={"1": group1, "2": group2},
            allow_autocorr=False,
            group_pair_mode="intra",
        )
        inter = build_path_plan(
            files_groups={"1": group1, "2": group2},
            allow_autocorr=False,
            group_pair_mode="inter",
        )
        all_pairs = build_path_plan(
            files_groups={"1": group1, "2": group2},
            allow_autocorr=False,
            group_pair_mode="all",
        )

        self.assertEqual(len(intra.paths), 1)
        self.assertEqual(len(inter.paths), 2)
        self.assertEqual(len(all_pairs.paths), 3)

    def _station_info(self, station: str, lat: float, lon: float):
        paths = []
        for component in ("E", "N", "Z"):
            path = self.root / f"{station}.{component}.sac"
            _write_sac_header(path, lat=lat, lon=lon, ele=1.0)
            paths.append(str(path))
        return {
            "network": ["XX", "XX", "XX"],
            "location": ["00", "00", "00"],
            "component": ["E", "N", "Z"],
            "path": paths,
        }


def _write_sac_header(path: Path, *, lat: float, lon: float, ele: float) -> None:
    raw = bytearray(HEADER_BYTES)
    struct.pack_into("<i", raw, NVHDR_OFFSET, 6)
    struct.pack_into("<f", raw, STLA_OFFSET, lat)
    struct.pack_into("<f", raw, STLO_OFFSET, lon)
    struct.pack_into("<f", raw, STEL_OFFSET, ele)
    path.write_bytes(raw)


if __name__ == "__main__":
    unittest.main()
