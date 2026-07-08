import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastxc.io import SacHeader, write_sac
from fastxc.tools.plot_rtz_grid import (
    _infer_component_layout,
    _load_grid_traces,
    _read_receiver_summary,
)


class PlotRtzGridLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_single_component_layout_accepts_original_component_name(self):
        receiver_dir = self.root / "AA.SRC" / "BB.REC"
        self._write_sac(receiver_dir / "AA-BB.SRC-REC.BHZ-BHZ.ncf.SAC", dist=12.5)

        summary = _read_receiver_summary(receiver_dir)
        source_components, receiver_components = _infer_component_layout([summary])
        traces = _load_grid_traces(
            [summary],
            source_components=source_components,
            receiver_components=receiver_components,
            lag_window=0.0,
        )

        self.assertEqual(source_components, ("BHZ",))
        self.assertEqual(receiver_components, ("BHZ",))
        self.assertEqual(set(traces), {("BHZ", "BHZ")})
        self.assertEqual(len(traces[("BHZ", "BHZ")]), 1)

    def test_three_component_layout_uses_rtz_order(self):
        receiver_dir = self.root / "AA.SRC" / "BB.REC"
        for src in ("Z", "R", "T"):
            for rec in ("T", "Z", "R"):
                self._write_sac(receiver_dir / f"AA-BB.SRC-REC.{src}-{rec}.ncf.SAC", dist=15.0)

        summary = _read_receiver_summary(receiver_dir)
        source_components, receiver_components = _infer_component_layout([summary])

        self.assertEqual(source_components, ("R", "T", "Z"))
        self.assertEqual(receiver_components, ("R", "T", "Z"))

    def _write_sac(self, path: Path, *, dist: float) -> None:
        header = SacHeader.empty()
        header.set_int("npts", 4)
        header.set_float("delta", 0.5)
        header.set_float("b", -1.0)
        header.set_float("dist", dist)
        write_sac(path, header, np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
