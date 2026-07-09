import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastxc.io import SacHeader, read_sac, write_sac
from fastxc.tools.extract_ncf import extract_ncf, parse_component_pair, resolve_record_path


class ExtractNcfTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parse_component_pair_accepts_dash_or_slash(self):
        self.assertEqual(parse_component_pair("BHE-BHZ"), ("BHE", "BHZ"))
        self.assertEqual(parse_component_pair("R/Z"), ("R", "Z"))

    def test_extracts_record_from_sourcepack_with_migrated_xcpack_path(self):
        workspace = self.root / "workspace"
        xcpack_dir = workspace / "ncf" / "xcpack"
        sourcepack_dir = workspace / "sourcepack" / "20111222T00_00"
        xcpack_dir.mkdir(parents=True)
        sourcepack_dir.mkdir(parents=True)

        pack_path = xcpack_dir / "20111222T00_00.i000000.j000000.w000.p000.xcpack"
        record = self._sac_record([0.0, 1.0, -1.0])
        pack_path.write_bytes(b"padding" + record)
        offset = len(b"padding")

        index_path = sourcepack_dir / "sourcepack_index.tsv"
        index_path.write_text(
            "timestamp\tpath_id\tcomponent_slot\tsource_key\treceiver_key\tsrc_id\trec_id\t"
            "src_network\tsrc_station\tsrc_location\tsrc_component\t"
            "rec_network\trec_station\trec_location\trec_component\tnpts\tdt\tdist\taz\tbaz\t"
            "record_path\trecord_offset\tbytes\tstorage_kind\tfinal_pair_path\n"
            "20111222T00:00\t00010002\t2\tVV.45002.01\tVV.45009.01\t1\t2\t"
            "VV\t45002\t01\tBHE\tVV\t45009\t01\tBHZ\t3\t0.2\t1\t2\t3\t"
            f"/pie/workspace/ncf/xcpack/{pack_path.name.replace('_', ':')}\t{offset}\t{len(record)}\txcpack\tignored\n",
            encoding="utf-8",
        )

        output = self.root / "one.SAC"
        result = extract_ncf(
            workspace=workspace,
            timestamp="20111222T00_00",
            source="45002",
            receiver="45009",
            component_pair="BHE-BHZ",
            output=output,
        )

        self.assertEqual(result.record_path, pack_path.resolve())
        header, data = read_sac(output)
        self.assertEqual(header.npts, 3)
        np.testing.assert_allclose(data, np.array([0.0, 1.0, -1.0], dtype=np.float32))

    def test_resolve_record_path_uses_workspace_fallback(self):
        workspace = self.root / "workspace"
        xcpack_dir = workspace / "ncf" / "xcpack"
        index_dir = workspace / "sourcepack" / "DAY"
        xcpack_dir.mkdir(parents=True)
        index_dir.mkdir(parents=True)
        pack_path = xcpack_dir / "DAY.xcpack"
        pack_path.write_bytes(b"x")

        resolved = resolve_record_path(
            {"record_path": "/container/workspace/ncf/xcpack/DAY.xcpack"},
            index_path=index_dir / "sourcepack_index.tsv",
            workspace=workspace,
        )

        self.assertEqual(resolved, pack_path.resolve())

    def _sac_record(self, values):
        header = SacHeader.empty()
        header.set_int("npts", len(values))
        header.set_float("delta", 0.2)
        data = np.asarray(values, dtype=np.float32)
        tmp = self.root / "tmp.SAC"
        write_sac(tmp, header, data)
        return tmp.read_bytes()


if __name__ == "__main__":
    unittest.main()