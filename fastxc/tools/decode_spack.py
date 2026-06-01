from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fastxc.io.spack import (
    SegspecSource,
    read_spack_sources,
    read_spack_timestamp_sources,
)


@dataclass(frozen=True)
class SpackDecodeResult:
    output_dir: Path
    record_count: int
    bytes_written: int


def decode_spack(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    timestamp: str | None = None,
    nsl_id: int | None = None,
    network: str | None = None,
    station: str | None = None,
    location: str | None = None,
    component: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> SpackDecodeResult:
    """Decode raw SEGSPEC blocks from SAC2SPEC spack output.

    The decoded files are byte-for-byte SEGSPEC records copied from the spack
    block payloads. This is intended for debugging and compatibility exports,
    not for the normal FastXC production path.
    """

    out_root = Path(output_dir).expanduser().resolve()
    grouped = _load_spack_groups(Path(input_path).expanduser(), timestamp=timestamp)

    record_count = 0
    bytes_written = 0
    for source_timestamp, sources in sorted(grouped.items()):
        if timestamp is not None and source_timestamp != timestamp:
            continue
        for source in sources:
            if not _matches(
                source,
                nsl_id=nsl_id,
                network=network,
                station=station,
                location=location,
                component=component,
            ):
                continue

            record_count += 1
            bytes_written += int(source.block_bytes)
            if not dry_run:
                output_path = _output_path(out_root, source_timestamp, source)
                _copy_source_record(source, output_path, overwrite=overwrite)

            if limit is not None and limit > 0 and record_count >= limit:
                return SpackDecodeResult(out_root, record_count, bytes_written)

    return SpackDecodeResult(out_root, record_count, bytes_written)


def _load_spack_groups(input_path: Path, *, timestamp: str | None) -> dict[str, list[SegspecSource]]:
    path = input_path.resolve()

    if path.is_file() and path.suffix.lower() == ".tsv":
        path = path.parent

    if not path.is_dir():
        raise FileNotFoundError(f"Spack input is not a directory or TSV sidecar: {input_path}")

    grouped = read_spack_sources(path)
    if grouped is not None:
        return grouped

    source_timestamp = timestamp or _timestamp_from_success(path) or path.name
    return {
        source_timestamp: read_spack_timestamp_sources(
            path,
            timestamp=timestamp,
        )
    }


def _timestamp_from_success(timestamp_dir: Path) -> str | None:
    success = timestamp_dir / "_SUCCESS"
    if not success.is_file():
        return None
    for raw in success.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("timestamp="):
            return line.split("=", 1)[1].strip()
    return None


def _matches(
    source: SegspecSource,
    *,
    nsl_id: int | None,
    network: str | None,
    station: str | None,
    location: str | None,
    component: str | None,
) -> bool:
    return all(
        [
            nsl_id is None or source.nsl_id == nsl_id,
            network is None or source.network == network,
            station is None or source.station == station,
            location is None or source.location == location,
            component is None or source.component == component,
        ]
    )


def _output_path(output_root: Path, timestamp: str, source: SegspecSource) -> Path:
    name = (
        f"{source.nsl_id:04d}."
        f"{source.network}.{source.station}.{source.location}.{source.component}.SEGSPEC"
    )
    return output_root / _safe_path_part(timestamp) / name


def _safe_path_part(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._:-" else "_" for ch in text)


def _copy_source_record(source: SegspecSource, output_path: Path, *, overwrite: bool) -> None:
    if not source.pack_path:
        raise ValueError(f"Source has no spack pack path: {source.segspec_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Decoded SEGSPEC already exists, pass --force to overwrite: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    remaining = int(source.block_bytes)
    with Path(source.pack_path).open("rb") as src, output_path.open("wb") as dst:
        src.seek(int(source.pack_offset))
        while remaining:
            chunk = src.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"Short spack read: {source.pack_path}")
            dst.write(chunk)
            remaining -= len(chunk)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode SAC2SPEC spack records to SEGSPEC files.")
    parser.add_argument("-I", "--input", required=True, help="workspace root, spack root, timestamp dir, or TSV")
    parser.add_argument("-O", "--output", required=True, help="output directory for decoded SEGSPEC files")
    parser.add_argument("--timestamp", help="only decode one timestamp")
    parser.add_argument("--nsl-id", type=int, help="only decode one nsl_id")
    parser.add_argument("--network", help="only decode one network")
    parser.add_argument("--station", help="only decode one station")
    parser.add_argument("--location", help="only decode one location")
    parser.add_argument("--component", help="only decode one component")
    parser.add_argument("--limit", type=_positive_int, help="maximum number of records to decode")
    parser.add_argument("--dry-run", action="store_true", help="count matching records without writing files")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite existing decoded files")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = decode_spack(
        args.input,
        args.output,
        timestamp=args.timestamp,
        nsl_id=args.nsl_id,
        network=args.network,
        station=args.station,
        location=args.location,
        component=args.component,
        limit=args.limit,
        dry_run=args.dry_run,
        overwrite=args.force,
    )
    verb = "Would decode" if args.dry_run else "Decoded"
    print(f"{verb} {result.record_count} SEGSPEC record(s), {result.bytes_written} byte(s).")
    print(f"Output: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
