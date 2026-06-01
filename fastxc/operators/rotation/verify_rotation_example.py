from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastxc.operators.rotation.python_rotate import build_rotate_matrix, rotate_enz_to_rtz


def explicit_rotate(enz: np.ndarray, azimuth_deg: float, back_azimuth_deg: float) -> np.ndarray:
    az = np.float32(np.deg2rad(azimuth_deg))
    baz = np.float32(np.deg2rad(back_azimuth_deg))
    sin_a = np.sin(az, dtype=np.float32)
    cos_a = np.cos(az, dtype=np.float32)
    sin_b = np.sin(baz, dtype=np.float32)
    cos_b = np.cos(baz, dtype=np.float32)

    ee, en, ez, ne, nn, nz, ze, zn, zz = enz
    out = np.empty_like(enz, dtype=np.float32)
    out[0] = ee * (-sin_a * sin_b) + en * (-sin_a * cos_b) + ne * (-cos_a * sin_b) + nn * (-cos_a * cos_b)
    out[1] = ee * (-sin_a * cos_b) + en * (sin_a * sin_b) + ne * (-cos_a * cos_b) + nn * (cos_a * sin_b)
    out[2] = ez * sin_a + nz * cos_a
    out[3] = ee * (-cos_a * sin_b) + en * (-cos_a * cos_b) + ne * (sin_a * sin_b) + nn * (sin_a * cos_b)
    out[4] = ee * (-cos_a * cos_b) + en * (cos_a * sin_b) + ne * (sin_a * cos_b) + nn * (-sin_a * sin_b)
    out[5] = ez * cos_a + nz * (-sin_a)
    out[6] = ze * (-sin_b) + zn * (-cos_b)
    out[7] = ze * (-cos_b) + zn * sin_b
    out[8] = zz
    return out


def main() -> None:
    rng = np.random.default_rng(20260520)
    enz = rng.normal(size=(9, 128)).astype(np.float32)
    azimuth = 37.5
    back_azimuth = 221.25

    via_matrix = rotate_enz_to_rtz(enz, azimuth, back_azimuth)
    via_formula = explicit_rotate(enz, azimuth, back_azimuth)
    np.testing.assert_allclose(via_matrix, via_formula, rtol=2.0e-6, atol=2.0e-6)

    matrix = build_rotate_matrix(azimuth, back_azimuth)
    print("rotation verification passed")
    print(f"matrix shape: {matrix.shape}, max abs diff: {np.max(np.abs(via_matrix - via_formula)):.3e}")


if __name__ == "__main__":
    main()
