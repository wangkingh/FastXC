from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from fastxc.io import SAC_HEADER_BYTES, SacHeader


def _timestamp_text(header: SacHeader) -> str:
    year = header.get_int("nzyear")
    jday = header.get_int("nzjday")
    hour = header.get_int("nzhour")
    minute = header.get_int("nzmin")
    second = max(0, header.get_int("nzsec"))
    offset = max(0.0, header.get_float("user4"))

    if year <= 0 or jday <= 0 or hour < 0 or minute < 0:
        return "unknown"

    start = (
        datetime(year, 1, 1)
        + timedelta(days=jday - 1, hours=hour, minutes=minute, seconds=second)
        + timedelta(seconds=offset)
    )
    start_jday = (start - datetime(start.year, 1, 1)).days + 1
    return f"{start.year}.{start_jday:03d}.{start.hour:02d}{start.minute:02d}"


def _segment_name(bigsac_path: Path, header: SacHeader, index: int) -> str:
    parts = bigsac_path.name.split(".")
    if len(parts) >= 4 and parts[-1].lower() == "bigsac":
        network_pair, station_pair, component_pair = parts[0], parts[1], parts[2]
        return f"{network_pair}.{station_pair}.{_timestamp_text(header)}.{index:04d}.{component_pair}.sac"
    return f"{bigsac_path.stem}.{_timestamp_text(header)}.{index:04d}.sac"


def extract_bigsac(bigsac_path: str | Path, output_dir: str | Path) -> int:
    bigsac_path = Path(bigsac_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with bigsac_path.open("rb") as handle:
        while True:
            raw_header = handle.read(SAC_HEADER_BYTES)
            if not raw_header:
                break
            if len(raw_header) != SAC_HEADER_BYTES:
                raise ValueError(f"{bigsac_path}: truncated SAC header at segment {count}")

            header = SacHeader.from_bytes(raw_header)
            data_nbytes = header.data_nbytes()
            data = handle.read(data_nbytes)
            if len(data) != data_nbytes:
                raise ValueError(f"{bigsac_path}: truncated SAC data at segment {count}")

            out_path = output_dir / _segment_name(bigsac_path, header, count)
            with out_path.open("wb") as out:
                out.write(raw_header)
                out.write(data)
            count += 1

    return count


def extract_bigsac_dir(input_dir: str | Path, output_dir: str | Path) -> int:
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    total = 0
    for bigsac_path in sorted(input_dir.rglob("*.bigsac")):
        rel_parent = bigsac_path.parent.relative_to(input_dir)
        total += extract_bigsac(bigsac_path, output_dir / rel_parent)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a directory of BigSAC files to SAC files.")
    parser.add_argument("-I", "--input", required=True, help="input directory containing .bigsac files")
    parser.add_argument("-O", "--output", required=True, help="output directory for extracted .sac files")
    args = parser.parse_args(argv)

    count = extract_bigsac_dir(args.input, args.output)
    print(f"Extracted {count} SAC segment(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
