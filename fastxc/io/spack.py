from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from .xcspec import COMPLEX64_BYTES, SEGSPEC_HEADER_STRUCT


@dataclass(frozen=True)
class SegspecSource:
    file_index: int
    nsl_id: int
    network: str
    station: str
    location: str
    component: str
    stla: float
    stlo: float
    nstep: int
    nspec: int
    dt: float
    df: float
    segspec_path: str
    size: int
    mtime_ns: int
    pack_path: str = ""
    pack_offset: int = 0
    block_bytes: int = 0


def read_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"XCache list not found: {path}")

    paths: list[Path] = []
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        item = Path(line).expanduser()
        paths.append((item if item.is_absolute() else path.parent / item).resolve())

    if not paths:
        raise ValueError(f"XCache list is empty: {path}")
    return paths


def read_sources(speclist: Path, timestamp: str) -> list[SegspecSource]:
    rows = [_read_source(path, timestamp) for path in read_list(speclist)]
    return finalize_sources(rows, timestamp, speclist.as_posix())


def read_spack_sources(root: Path) -> dict[str, list[SegspecSource]] | None:
    if (root / "_SUCCESS").is_file() and any(child.is_dir() for child in root.iterdir()):
        return _read_spack_sources_from_root(root, recursive=True)

    partition_root = root / "spack_by_timestamp"
    if (partition_root / "_SUCCESS").is_file():
        return _read_spack_sources_from_root(partition_root, recursive=True)

    spack_root = root / "spack"
    if not (spack_root / "_SUCCESS").is_file():
        return None

    return _read_spack_sources_from_root(spack_root, recursive=False)


def read_spack_timestamp_sources(timestamp_dir: Path, timestamp: str | None = None) -> list[SegspecSource]:
    """Read one completed ``spack_by_timestamp/<timestamp>`` directory."""

    timestamp_dir = Path(timestamp_dir).expanduser().resolve()
    tsvs = sorted(timestamp_dir.glob("*.tsv"))
    if not tsvs:
        raise ValueError(f"SAC2SPEC timestamp spack has no TSV sidecars: {timestamp_dir}")

    grouped: dict[str, list[SegspecSource]] = {}
    for tsv in tsvs:
        with tsv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                source = _read_spack_row(row, tsv)
                grouped.setdefault(row["timestamp"], []).append(source)

    if not grouped:
        raise ValueError(f"SAC2SPEC timestamp spack sidecars are empty: {timestamp_dir}")
    if timestamp is not None:
        if timestamp not in grouped:
            raise ValueError(f"Timestamp {timestamp} not found in {timestamp_dir}")
        if len(grouped) != 1:
            raise ValueError(f"Timestamp directory contains extra timestamps: {timestamp_dir}")
        return finalize_sources(grouped[timestamp], timestamp, timestamp_dir.as_posix())
    if len(grouped) != 1:
        raise ValueError(f"Timestamp directory contains multiple timestamps: {timestamp_dir}")
    source_timestamp, rows = next(iter(grouped.items()))
    return finalize_sources(rows, source_timestamp, timestamp_dir.as_posix())


def _read_spack_sources_from_root(spack_root: Path, *, recursive: bool) -> dict[str, list[SegspecSource]]:
    tsvs = sorted(spack_root.glob("*/*.tsv") if recursive else spack_root.glob("*.tsv"))
    if not tsvs:
        raise ValueError(f"SAC2SPEC spack _SUCCESS exists but no TSV sidecars found: {spack_root}")

    grouped: dict[str, list[SegspecSource]] = {}
    for tsv in tsvs:
        with tsv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                source = _read_spack_row(row, tsv)
                grouped.setdefault(row["timestamp"], []).append(source)

    if not grouped:
        raise ValueError(f"SAC2SPEC spack sidecars are empty: {spack_root}")
    return {
        timestamp: finalize_sources(rows, timestamp, spack_root.as_posix())
        for timestamp, rows in grouped.items()
    }


def finalize_sources(rows: list[SegspecSource], timestamp: str, origin: str) -> list[SegspecSource]:
    _validate_shape(rows, timestamp)
    rows.sort(key=_source_sort_key)
    out: list[SegspecSource] = []
    seen: set[tuple[int, str]] = set()
    for file_index, row in enumerate(rows):
        key = (row.nsl_id, row.component)
        if key in seen:
            raise ValueError(f"Duplicate nsl_id/component in {origin}: {key}")
        seen.add(key)
        out.append(replace(row, file_index=file_index))
    return out


def _read_source(path: Path, timestamp: str) -> SegspecSource:
    if not path.is_file():
        raise FileNotFoundError(f"Listed SEGSPEC is missing for {timestamp}: {path}")

    fields = path.name.split(".")
    if len(fields) != 6 or fields[-1].lower() != "segspec":
        raise ValueError(f"Unexpected SEGSPEC file name: {path.name}")
    try:
        nsl_id = int(fields[0])
    except ValueError as exc:
        raise ValueError(f"Invalid nsl_id in SEGSPEC name: {path.name}") from exc
    network, station, location, component = fields[1:5]

    with path.open("rb") as handle:
        header_bytes = handle.read(SEGSPEC_HEADER_STRUCT.size)
    if len(header_bytes) != SEGSPEC_HEADER_STRUCT.size:
        raise ValueError(f"SEGSPEC header is truncated: {path}")

    stla, stlo, nstep, nspec, df, dt, header_nsl_id = SEGSPEC_HEADER_STRUCT.unpack(header_bytes)
    if header_nsl_id != nsl_id:
        raise ValueError(f"SEGSPEC nsl_id mismatch in {path}: name={nsl_id} header={header_nsl_id}")
    if nstep <= 0 or nspec <= 1 or dt <= 0.0 or df <= 0.0:
        raise ValueError(f"Invalid SEGSPEC header: {path}")

    stat = path.stat()
    expected_size = SEGSPEC_HEADER_STRUCT.size + nstep * nspec * COMPLEX64_BYTES
    if stat.st_size != expected_size:
        raise ValueError(f"SEGSPEC size mismatch in {path}: expected={expected_size} got={stat.st_size}")

    return SegspecSource(
        file_index=-1,
        nsl_id=nsl_id,
        network=network,
        station=station,
        location=location,
        component=component,
        stla=stla,
        stlo=stlo,
        nstep=nstep,
        nspec=nspec,
        dt=dt,
        df=df,
        segspec_path=path.resolve().as_posix(),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        block_bytes=stat.st_size,
    )


def _read_spack_row(row: dict[str, str], tsv: Path) -> SegspecSource:
    try:
        timestamp = row["timestamp"]
        pack_path = _resolve_row_path(row["pack_path"], tsv)
        offset = int(row["offset"])
        block_bytes = int(row["bytes"])
        nsl_id = int(row["nsl_id"])
        network = row["network"]
        station = row["station"]
        location = row["location"]
        component = row["component"]
    except KeyError as exc:
        raise ValueError(f"Missing spack TSV field {exc.args[0]!r}: {tsv}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid numeric spack TSV field in {tsv}: {row}") from exc

    if offset < 0 or block_bytes <= SEGSPEC_HEADER_STRUCT.size:
        raise ValueError(f"Invalid spack offset/bytes in {tsv}: {row}")
    stat = pack_path.stat()
    if offset + block_bytes > stat.st_size:
        raise ValueError(f"Spack record exceeds pack size: {pack_path} offset={offset} bytes={block_bytes}")

    with pack_path.open("rb") as handle:
        handle.seek(offset)
        header_bytes = handle.read(SEGSPEC_HEADER_STRUCT.size)
    if len(header_bytes) != SEGSPEC_HEADER_STRUCT.size:
        raise ValueError(f"Spack SEGSPEC header is truncated: {pack_path} offset={offset}")

    stla, stlo, nstep, nspec, df, dt, header_nsl_id = SEGSPEC_HEADER_STRUCT.unpack(header_bytes)
    if header_nsl_id != nsl_id:
        raise ValueError(f"Spack nsl_id mismatch in {tsv}: tsv={nsl_id} header={header_nsl_id}")
    if nstep <= 0 or nspec <= 1 or dt <= 0.0 or df <= 0.0:
        raise ValueError(f"Invalid spack SEGSPEC header: {pack_path} offset={offset}")

    expected_size = SEGSPEC_HEADER_STRUCT.size + nstep * nspec * COMPLEX64_BYTES
    if block_bytes != expected_size:
        raise ValueError(f"Spack record size mismatch in {tsv}: expected={expected_size} got={block_bytes}")

    if row.get("nstep") and int(row["nstep"]) != nstep:
        raise ValueError(f"Spack nstep mismatch in {tsv}: tsv={row['nstep']} header={nstep}")
    if row.get("nspec") and int(row["nspec"]) != nspec:
        raise ValueError(f"Spack nspec mismatch in {tsv}: tsv={row['nspec']} header={nspec}")

    original_spec_path = row.get("original_spec_path") or f"{pack_path.as_posix()}:{offset}"

    return SegspecSource(
        file_index=-1,
        nsl_id=nsl_id,
        network=network,
        station=station,
        location=location,
        component=component,
        stla=stla,
        stlo=stlo,
        nstep=nstep,
        nspec=nspec,
        dt=dt,
        df=df,
        segspec_path=original_spec_path,
        size=block_bytes,
        mtime_ns=stat.st_mtime_ns,
        pack_path=pack_path.as_posix(),
        pack_offset=offset,
        block_bytes=block_bytes,
    )


def _resolve_row_path(text: str, tsv: Path) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = tsv.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Spack pack file is missing for {tsv}: {path}")
    return path.resolve()


def _validate_shape(rows: list[SegspecSource], timestamp: str) -> None:
    if not rows:
        raise ValueError(f"No SEGSPEC sources for timestamp {timestamp}")

    first = rows[0]
    for row in rows[1:]:
        if (
            row.nstep != first.nstep
            or row.nspec != first.nspec
            or abs(row.dt - first.dt) > 1e-6
            or abs(row.df - first.df) > 1e-9
        ):
            raise ValueError(f"SEGSPEC shape mismatch for timestamp {timestamp}: {row.segspec_path}")


def _source_sort_key(row: SegspecSource) -> tuple[int, int, str, str, str, str, str]:
    return (
        row.nsl_id,
        _component_rank(row.component),
        row.network,
        row.station,
        row.location,
        row.component,
        row.segspec_path,
    )


def _component_rank(component: str) -> int:
    tail = component.upper()[-1:] if component else ""
    return {"E": 0, "1": 0, "N": 1, "2": 1, "Z": 2, "3": 2}.get(tail, 100)
