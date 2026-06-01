from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Iterable

from fastxc.io.xcspec import XCSPEC_MAGIC, XCSPEC_SOURCE_ENTRY_SIZE
from fastxc.operators.xcache.writer import INDEX_NAME


@dataclass(frozen=True)
class XCacheSourceEntry:
    file_index: int
    nsl_id: int
    stla: float
    stlo: float
    network: str
    station: str
    location: str
    component: str


@dataclass(frozen=True)
class XCacheInspection:
    path: str
    size: int
    magic: str
    version: int
    endian_tag: int
    header_size: int
    source_entry_size: int
    source_table_offset: int
    payload_offset: int
    layout: int
    dtype: int
    string_encoding: int
    timestamp: str
    file_count: int
    source_count: int
    nstep: int
    nspec: int
    nfft: int
    dt: float
    df: float
    step_bytes: int
    payload_bytes: int
    source_table_bytes: int
    manifest_hash_u64: int
    source_table_sha256: str
    payload_sha256: str | None
    sources: list[XCacheSourceEntry]


def inspect_xcache(
    input_path: str | Path,
    *,
    source_limit: int = 10,
    hash_payload: bool = False,
) -> list[XCacheInspection]:
    return [
        inspect_xcspec(path, source_limit=source_limit, hash_payload=hash_payload)
        for path in _discover_xcspec_paths(Path(input_path).expanduser())
    ]


def inspect_xcspec(
    path: str | Path,
    *,
    source_limit: int = 10,
    hash_payload: bool = False,
) -> XCacheInspection:
    xcspec = Path(path).expanduser().resolve()
    stat = xcspec.stat()
    with xcspec.open("rb") as handle:
        header = handle.read(256)
        fields = _parse_header(header, xcspec)
        source_table_bytes = int(fields["source_table_bytes"])
        source_table_offset = int(fields["source_table_offset"])
        source_entry_size = int(fields["source_entry_size"])
        file_count = int(fields["file_count"])

        handle.seek(source_table_offset)
        source_table = handle.read(source_table_bytes)
        if len(source_table) != source_table_bytes:
            raise ValueError(f"XCache source table is truncated: {xcspec}")

        sources = _parse_sources(source_table, source_entry_size, min(source_limit, file_count))
        payload_sha256 = None
        if hash_payload:
            payload_sha256 = _hash_file_region(
                handle,
                int(fields["payload_offset"]),
                int(fields["payload_bytes"]),
            )

    return XCacheInspection(
        path=xcspec.as_posix(),
        size=stat.st_size,
        source_table_sha256=hashlib.sha256(source_table).hexdigest(),
        payload_sha256=payload_sha256,
        sources=sources,
        **fields,
    )


def _discover_xcspec_paths(input_path: Path) -> list[Path]:
    path = input_path.resolve()
    if path.is_file() and path.suffix.lower() == ".xcspec":
        return [path]
    if path.is_file():
        return _paths_from_index(path)
    if not path.is_dir():
        raise FileNotFoundError(f"XCache input does not exist: {input_path}")

    index = path / INDEX_NAME
    if index.is_file():
        return _paths_from_index(index)

    paths = sorted(path.glob("*.xcspec"))
    if not paths:
        raise ValueError(f"No .xcspec files found: {path}")
    return paths


def _paths_from_index(index: Path) -> list[Path]:
    paths: list[Path] = []
    with index.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "xcspec_path" not in (reader.fieldnames or []):
            raise ValueError(f"XCache index has no xcspec_path column: {index}")
        for row in reader:
            item = Path(row["xcspec_path"]).expanduser()
            paths.append((item if item.is_absolute() else index.parent / item).resolve())
    if not paths:
        raise ValueError(f"XCache index is empty: {index}")
    return paths


def _parse_header(header: bytes, path: Path) -> dict[str, object]:
    if len(header) < 256:
        raise ValueError(f"XCache header is truncated: {path}")
    magic = struct.unpack_from("<8s", header, 0)[0]
    if magic != XCSPEC_MAGIC:
        raise ValueError(f"Unexpected XCache magic in {path}: {magic!r}")
    return {
        "magic": magic.decode("ascii", errors="replace"),
        "version": struct.unpack_from("<I", header, 8)[0],
        "endian_tag": struct.unpack_from("<I", header, 12)[0],
        "header_size": struct.unpack_from("<I", header, 16)[0],
        "source_entry_size": struct.unpack_from("<I", header, 20)[0],
        "source_table_offset": struct.unpack_from("<Q", header, 24)[0],
        "file_count": struct.unpack_from("<I", header, 32)[0],
        "source_count": struct.unpack_from("<I", header, 36)[0],
        "payload_offset": struct.unpack_from("<Q", header, 40)[0],
        "layout": struct.unpack_from("<I", header, 48)[0],
        "dtype": struct.unpack_from("<I", header, 52)[0],
        "string_encoding": struct.unpack_from("<I", header, 56)[0],
        "timestamp": _decode_fixed(header[64:128]),
        "nstep": struct.unpack_from("<I", header, 128)[0],
        "nspec": struct.unpack_from("<I", header, 132)[0],
        "nfft": struct.unpack_from("<I", header, 136)[0],
        "dt": struct.unpack_from("<f", header, 144)[0],
        "df": struct.unpack_from("<f", header, 148)[0],
        "step_bytes": struct.unpack_from("<Q", header, 152)[0],
        "payload_bytes": struct.unpack_from("<Q", header, 160)[0],
        "manifest_hash_u64": struct.unpack_from("<Q", header, 168)[0],
        "source_table_bytes": struct.unpack_from("<Q", header, 176)[0],
    }


def _parse_sources(source_table: bytes, entry_size: int, count: int) -> list[XCacheSourceEntry]:
    if entry_size < XCSPEC_SOURCE_ENTRY_SIZE:
        raise ValueError(f"XCache source_entry_size is too small: {entry_size}")
    sources: list[XCacheSourceEntry] = []
    for idx in range(count):
        offset = idx * entry_size
        entry = source_table[offset : offset + entry_size]
        if len(entry) != entry_size:
            raise ValueError("XCache source entry is truncated")
        sources.append(
            XCacheSourceEntry(
                file_index=struct.unpack_from("<I", entry, 0)[0],
                nsl_id=struct.unpack_from("<I", entry, 4)[0],
                stla=struct.unpack_from("<f", entry, 8)[0],
                stlo=struct.unpack_from("<f", entry, 12)[0],
                network=_decode_fixed(entry[16:32]),
                station=_decode_fixed(entry[32:64]),
                location=_decode_fixed(entry[64:80]),
                component=_decode_fixed(entry[80:96]),
            )
        )
    return sources


def _decode_fixed(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace")


def _hash_file_region(handle, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    handle.seek(offset)
    remaining = size
    while remaining:
        chunk = handle.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise ValueError("XCache payload is truncated")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def format_inspections(inspections: list[XCacheInspection]) -> str:
    chunks: list[str] = []
    for item in inspections:
        chunks.append(_format_one(item))
    return "\n\n".join(chunks)


def _format_one(item: XCacheInspection) -> str:
    lines = [
        f"XCache: {item.path}",
        f"  timestamp: {item.timestamp}",
        f"  file_count/source_count: {item.file_count}/{item.source_count}",
        f"  nstep/nspec/nfft: {item.nstep}/{item.nspec}/{item.nfft}",
        f"  dt/df: {item.dt:.9g}/{item.df:.9g}",
        f"  source_table_offset/payload_offset: {item.source_table_offset}/{item.payload_offset}",
        f"  step_bytes/payload_bytes: {item.step_bytes}/{item.payload_bytes}",
        f"  manifest_hash_u64: {item.manifest_hash_u64}",
        f"  source_table_sha256: {item.source_table_sha256}",
    ]
    if item.payload_sha256 is not None:
        lines.append(f"  payload_sha256: {item.payload_sha256}")
    if item.sources:
        lines.append(f"  sources(first {len(item.sources)}):")
        for source in item.sources:
            lines.append(
                "    "
                f"{source.file_index:5d} "
                f"nsl={source.nsl_id} "
                f"{source.network}.{source.station}.{source.location}.{source.component} "
                f"stla={source.stla:.6g} stlo={source.stlo:.6g}"
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect FastXC xcache binary shards.")
    parser.add_argument("-I", "--input", required=True, help=".xcspec file, xcache dir, or xcspec_index.tsv")
    parser.add_argument("--sources", type=int, default=10, help="number of SourceEntry rows to print per shard")
    parser.add_argument("--hash-payload", action="store_true", help="compute SHA256 for the payload bytes")
    parser.add_argument("--json", action="store_true", help="write machine-readable JSON")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    inspections = inspect_xcache(
        args.input,
        source_limit=max(args.sources, 0),
        hash_payload=args.hash_payload,
    )
    if args.json:
        print(json.dumps([asdict(item) for item in inspections], indent=2, ensure_ascii=False))
    else:
        print(format_inspections(inspections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
