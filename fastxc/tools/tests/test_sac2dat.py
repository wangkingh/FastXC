from __future__ import annotations

import numpy as np

from fastxc.io import SacHeader, write_sac
from fastxc.tools.sac2dat import convert_sac_dir


def test_convert_sac_dir_accepts_case_insensitive_sac_suffix(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    lower = input_dir / "lower.sac"
    upper = input_dir / "AA.BB" / "CC.DD.ncf.SAC"
    mixed = input_dir / "nested" / "mixed.SaC"

    for path in (lower, upper, mixed):
        _write_test_sac(path)

    results = convert_sac_dir(input_dir, output_dir)

    assert len(results) == 3
    assert (output_dir / "lower.dat").is_file()
    assert (output_dir / "AA.BB" / "CC.DD.ncf.dat").is_file()
    assert (output_dir / "nested" / "mixed.dat").is_file()


def _write_test_sac(path):
    header = SacHeader.empty()
    header.set_int("npts", 4)
    header.set_float("delta", 0.5)
    write_sac(path, header, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
