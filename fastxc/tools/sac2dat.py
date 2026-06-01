from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastxc.io import SacHeader, read_sac

_SAC_NULL = -12345.0


@dataclass(frozen=True)
class DatConversionResult:
    sac_path: Path
    dat_path: Path


def _header_float(header: SacHeader, name: str, default: float = 0.0) -> float:
    value = header.get_float(name)
    if not np.isfinite(value) or abs(value - _SAC_NULL) < 1.0e-3:
        return default
    return float(value)


def _split_ccf_trace(data: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = data.size // 2
    neg = data[:half][::-1]
    pos = data[half:]
    if neg.size < pos.size:
        neg = np.pad(neg, (0, pos.size - neg.size))
    t = np.arange(pos.size, dtype=np.float64) * float(dt)
    return t, neg.astype(np.float64, copy=False), pos.astype(np.float64, copy=False)


def sac_to_dat(sac_path: str | Path, dat_path: str | Path) -> DatConversionResult:
    sac_path = Path(sac_path).expanduser().resolve()
    dat_path = Path(dat_path).expanduser().resolve()

    header, data = read_sac(sac_path)
    dt = _header_float(header, "delta", 0.0)
    if dt <= 0.0:
        raise ValueError(f"{sac_path}: invalid SAC delta: {dt}")

    evlo = _header_float(header, "evlo", 0.0)
    evla = _header_float(header, "evla", 0.0)
    stlo = _header_float(header, "stlo", 0.0)
    stla = _header_float(header, "stla", 0.0)
    evel = _header_float(header, "evel", 0.0)
    stel = _header_float(header, "stel", 0.0)

    t, neg, pos = _split_ccf_trace(data.astype(np.float32, copy=False), dt)
    dat_path.parent.mkdir(parents=True, exist_ok=True)
    with dat_path.open("w", encoding="ascii") as handle:
        handle.write(f"{evlo:.7e} {evla:.7e} {evel:.7e}\n")
        handle.write(f"{stlo:.7e} {stla:.7e} {stel:.7e}\n")
        for tt, nn, pp in zip(t, neg, pos):
            handle.write(f"{tt:.7e} {nn:.7e} {pp:.7e}\n")

    return DatConversionResult(sac_path=sac_path, dat_path=dat_path)


def convert_sac_dir(input_dir: str | Path, output_dir: str | Path) -> list[DatConversionResult]:
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    results: list[DatConversionResult] = []
    for sac_path in sorted(input_dir.rglob("*.sac")):
        rel_path = sac_path.relative_to(input_dir).with_suffix(".dat")
        results.append(sac_to_dat(sac_path, output_dir / rel_path))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a directory of SAC files to DAT files.")
    parser.add_argument("-I", "--input", required=True, help="input directory containing .sac files")
    parser.add_argument("-O", "--output", required=True, help="output directory for .dat files")
    args = parser.parse_args(argv)

    results = convert_sac_dir(args.input, args.output)
    print(f"Converted {len(results)} SAC file(s) to DAT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
