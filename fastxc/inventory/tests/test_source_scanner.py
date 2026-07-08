import tempfile
import unittest
from pathlib import Path

from fastxc.config_parser.schema import TimeFilter
from fastxc.inventory.source_scanner import _gen_seis_file_group


class TestSourceScanner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, relative_path: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test data")

    def _time_filter(self) -> TimeFilter:
        return TimeFilter(
            time_start="2023-01-01 00:00:00",
            time_end="2023-01-01 23:59:59",
        )

    def test_three_component_list_order_maps_to_enz(self):
        for component in ("BHZ", "BHN", "BHE"):
            self._write(f"2023/001/ABC.{component}.sac")

        grouped = _gen_seis_file_group(
            str(self.root),
            "{home}/{YYYY}/{JJJ}/{station}.{component}.sac",
            "NONE",
            ["BHZ", "BHN", "BHE"],
            self._time_filter(),
            1,
        )

        info = next(iter(grouped.values()))
        self.assertEqual(info["component"], ["E", "N", "Z"])
        self.assertEqual(info["raw_component"], ["BHZ", "BHN", "BHE"])
        self.assertEqual(info["channel"], ["BHZ", "BHN", "BHE"])

    def test_single_component_keeps_original_label(self):
        self._write("2023/001/ABC.BHZ.sac")

        grouped = _gen_seis_file_group(
            str(self.root),
            "{home}/{YYYY}/{JJJ}/{station}.{component}.sac",
            "NONE",
            ["BHZ"],
            self._time_filter(),
            1,
        )

        info = next(iter(grouped.values()))
        self.assertEqual(info["component"], ["BHZ"])
        self.assertNotIn("raw_component", info)


if __name__ == "__main__":
    unittest.main()
