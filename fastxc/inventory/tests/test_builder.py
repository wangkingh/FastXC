import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastxc.config_parser.schema import ArrayInfo, TimeFilter
from fastxc.inventory.builder import build_inventory, require_inventory


class _Config(SimpleNamespace):
    @property
    def is_double_array(self):
        return False


class TestInventoryBuilder(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_sac_placeholder(self, relative_path: str) -> None:
        path = self.root / "data" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a complete SAC header")

    def _config(self, *, external_geo_tsv: Path) -> _Config:
        ini_path = self.root / "config.ini"
        ini_path.write_text("[test]\n", encoding="utf-8")
        return _Config(
            ini_path=ini_path,
            seisarrays=[
                ArrayInfo(
                    sac_dir=str(self.root / "data"),
                    pattern="{home}/{YYYY}/{JJJ}/{station}.{component}.sac",
                    sta_list="NONE",
                    component_list=["BHZ"],
                )
            ],
            time_filter=TimeFilter(
                time_start="2023-01-01 00:00:00",
                time_end="2023-01-01 23:59:59",
            ),
            device=SimpleNamespace(cpu_workers=1),
            xcorr=SimpleNamespace(
                distance_range="0/100",
                azimuth_range="-1/360",
                autocorr_mode="off",
                group_pair_mode="all",
            ),
            geometry=SimpleNamespace(external_geo_tsv=str(external_geo_tsv)),
            storage=SimpleNamespace(output_dir=self.root / "workspace"),
        )

    def test_build_inventory_fails_when_external_geo_matches_no_nodes(self):
        self._write_sac_placeholder("2023/001/AAA.BHZ.sac")
        self._write_sac_placeholder("2023/001/BBB.BHZ.sac")
        external_geo = self.root / "external_geo.tsv"
        external_geo.write_text(
            "station\tlat\tlon\n"
            "CCC\t1.0\t2.0\n",
            encoding="utf-8",
        )

        cfg = self._config(external_geo_tsv=external_geo)
        with self.assertRaisesRegex(ValueError, "0 allowed paths") as caught:
            build_inventory(cfg)

        message = str(caught.exception)
        self.assertIn("External geo rows: 1", message)
        self.assertIn("External geo nodes updated: 0", message)
        self.assertIn("did not update any station", message)
        self.assertTrue((cfg.storage.output_dir / "path_plan" / "allowed_paths.tsv").is_file())
        self.assertTrue((cfg.storage.output_dir / "path_plan" / "nsl_catalog.tsv").is_file())

    def test_require_inventory_rejects_existing_empty_tables(self):
        cfg = _Config(
            ini_path=self.root / "config.ini",
            storage=SimpleNamespace(output_dir=self.root / "workspace"),
        )
        cfg.ini_path.write_text("[test]\n", encoding="utf-8")
        path_plan = cfg.storage.output_dir / "path_plan"
        manifest = cfg.storage.output_dir / "manifest"
        path_plan.mkdir(parents=True)
        manifest.mkdir(parents=True)
        (path_plan / "allowed_paths.tsv").write_text("path_id\tsrc_nsl\trec_nsl\n", encoding="utf-8")
        (manifest / "sac_index.tsv").write_text("timestamp\tnsl_id\tsac_path\n", encoding="utf-8")
        (manifest / "timestamp_index.tsv").write_text("timestamp\tsac_row_count\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "no runnable work"):
            require_inventory(cfg)


if __name__ == "__main__":
    unittest.main()
