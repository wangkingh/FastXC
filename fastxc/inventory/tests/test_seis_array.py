import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from fastxc.inventory.arrays import SeisArray


class TestSeisArray(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, relative_path: str) -> None:
        path = self.test_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test data")

    def test_match_filter_and_group(self):
        self._write("2023/ABC_BHZ.sac")
        self._write("2023/DEF_LHZ.sac")
        self._write("2023/XYZ_BHZ.sac")

        sa = SeisArray(
            array_dir=str(self.test_dir),
            pattern="{home}/{YYYY}/{station}_{component}.sac",
        )

        matched = sa.match(threads=1)
        self.assertEqual(len(matched), 3)
        self.assertEqual(sorted(item["station"] for item in matched), ["ABC", "DEF", "XYZ"])
        self.assertEqual(sorted(item["component"] for item in matched), ["BHZ", "BHZ", "LHZ"])

        criteria = {
            "station": {
                "type": "list",
                "data_type": "str",
                "value": ["ABC", "XYZ"],
            }
        }
        filtered = sa.filter(criteria=criteria, threads=1)
        self.assertEqual(len(filtered), 2)

        grouped = sa.group(labels=["station"], sort_labels=["station"], filtered=True)
        self.assertEqual(set(grouped), {"ABC", "XYZ"})
        self.assertEqual(grouped["ABC"]["component"], ["BHZ"])
        self.assertIn("path", grouped["ABC"])

    def test_time_filter(self):
        self._write("2023/01/01/ABC_BHZ.sac")
        self._write("2023/01/02/ABC_BHZ.sac")
        self._write("2023/01/01/XYZ_BHZ.sac")

        sa = SeisArray(
            array_dir=str(self.test_dir),
            pattern="{home}/{YYYY}/{MM}/{DD}/{station}_{component}.sac",
        )
        sa.match()

        times = sorted(item["time"] for item in sa.files)
        self.assertTrue(all(isinstance(item, datetime) for item in times))

        filtered = sa.filter(
            criteria={
                "time": {
                    "type": "range",
                    "data_type": "datetime",
                    "value": [
                        datetime(2023, 1, 2),
                        datetime(2023, 1, 2, 23, 59, 59),
                    ],
                }
            }
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["station"], "ABC")

    def test_custom_fields(self):
        self._write("2023/ABC_123_BHZ.sac")
        self._write("2023/ABC_456_BHZ.sac")

        sa = SeisArray(
            array_dir=str(self.test_dir),
            pattern="{home}/{YYYY}/{station}_{shot}_{component}.sac",
            custom_fields={"shot": r"\d+"},
        )
        matched = sa.match()

        self.assertEqual(len(matched), 2)
        self.assertEqual(sorted(item["shot"] for item in matched), ["123", "456"])

    def test_empty_directory(self):
        sa = SeisArray(
            array_dir=str(self.test_dir),
            pattern="{home}/{YYYY}/{station}_{component}.sac",
        )
        self.assertEqual(sa.match(), [])

    def test_invalid_pattern(self):
        with self.assertRaises(ValueError):
            SeisArray(
                array_dir=str(self.test_dir),
                pattern="{home}/{YYYY}/some_{component}.sac",
            )

    def test_filter_without_match(self):
        sa = SeisArray(
            array_dir=str(self.test_dir),
            pattern="{home}/{YYYY}/{station}_{component}.sac",
        )

        with patch("logging.Logger.warning") as mock_warn:
            result = sa.filter(criteria={"station": {"type": "list", "value": ["ABC"]}})
            mock_warn.assert_called_once()
            self.assertIsNone(result)

    def test_multi_thread_match_and_filter(self):
        for index in range(10):
            self._write(f"2023/STA{index:02d}_BHZ.sac")

        sa = SeisArray(
            array_dir=str(self.test_dir),
            pattern="{home}/{YYYY}/{station}_{component}.sac",
        )
        self.assertEqual(len(sa.match(threads=4)), 10)

        filtered = sa.filter(
            criteria={
                "station": {
                    "type": "list",
                    "data_type": "str",
                    "value": ["STA00", "STA01", "STA02"],
                }
            },
            threads=4,
        )
        self.assertEqual(len(filtered), 3)

    def test_group_uses_tuple_keys_for_multiple_labels(self):
        self._write("2023/123/ABC_BHZ.sac")
        self._write("2023/123/ABC_BHN.sac")

        sa = SeisArray(
            array_dir=str(self.test_dir),
            pattern="{home}/{YYYY}/{JJJ}/{station}_{component}.sac",
        )
        sa.match()
        grouped = sa.group(labels=["station", "time"], sort_labels=["component"], filtered=False)

        key = ("ABC", datetime(2023, 5, 3))
        self.assertEqual(grouped[key]["component"], ["BHN", "BHZ"])


if __name__ == "__main__":
    unittest.main()
