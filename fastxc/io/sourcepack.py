from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


XC_PACK_REQUIRED_FIELDS = {
    "timestamp",
    "pack_path",
    "offset",
    "bytes",
    "src_id",
    "rec_id",
    "src_network",
    "src_station",
    "src_location",
    "src_component",
    "rec_network",
    "rec_station",
    "rec_location",
    "rec_component",
    "npts",
    "dt",
    "dist",
    "az",
    "baz",
}

SOURCEPACK_INDEX_HEADER = [
    "timestamp",
    "path_id",
    "component_slot",
    "source_key",
    "receiver_key",
    "src_id",
    "rec_id",
    "src_network",
    "src_station",
    "src_location",
    "src_component",
    "rec_network",
    "rec_station",
    "rec_location",
    "rec_component",
    "npts",
    "dt",
    "dist",
    "az",
    "baz",
    "record_path",
    "record_offset",
    "bytes",
    "storage_kind",
    "final_pair_path",
]


@dataclass(frozen=True)
class XcPackRecord:
    sequence: int
    timestamp: str
    pack_path: Path
    offset: int
    bytes: int
    row: dict[str, str]

    @property
    def source_key(self) -> str:
        return station_key(self.row, "src")

    @property
    def receiver_key(self) -> str:
        return station_key(self.row, "rec")


def discover_sourcepack_indexes(sourcepack_input: str | Path) -> list[Path]:
    path = Path(sourcepack_input).expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"SourcePack input not found: {path}")

    direct = path / "sourcepack_index.tsv"
    if direct.is_file():
        return [direct]

    indexes = sorted(p for p in path.glob("*/sourcepack_index.tsv") if p.is_file())
    if not indexes:
        raise FileNotFoundError(f"No sourcepack_index.tsv found under {path}")
    return indexes


def read_sourcepack_input_list(list_path: str | Path) -> list[Path]:
    path = Path(list_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SourcePack input list not found: {path}")
    indexes: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        raw = raw_line.strip()
        if not raw or raw.startswith("#"):
            continue
        index_path = Path(raw).expanduser()
        if not index_path.is_absolute():
            index_path = path.parent / index_path
        if not index_path.is_file():
            raise FileNotFoundError(f"Listed SourcePack index not found: {index_path}")
        indexes.append(index_path.resolve())
    return indexes


def discover_workspace_sourcepack_inputs(workspace: str | Path) -> list[Path]:
    root = Path(workspace).expanduser().resolve()
    input_list = root / "sourcepack_inputs.txt"
    if input_list.is_file():
        return read_sourcepack_input_list(input_list)
    return discover_sourcepack_indexes(root / "sourcepack")


def read_xcpack_tsv(tsv_path: Path, first_sequence: int) -> list[XcPackRecord]:
    records: list[XcPackRecord] = []
    path_cache: dict[str, Path] = {}
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return records
        missing = XC_PACK_REQUIRED_FIELDS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"XC pack TSV missing fields {sorted(missing)}: {tsv_path}")

        for row in reader:
            try:
                pack_path = _resolve_row_path(row["pack_path"], tsv_path, path_cache)
                offset = int(row["offset"])
                size = int(row["bytes"])
            except ValueError as exc:
                raise ValueError(f"Invalid XC pack numeric field in {tsv_path}: {row}") from exc

            if offset < 0 or size <= 0:
                raise ValueError(f"Invalid XC pack offset/bytes in {tsv_path}: {row}")
            records.append(
                XcPackRecord(
                    sequence=first_sequence + len(records),
                    timestamp=row["timestamp"],
                    pack_path=pack_path,
                    offset=offset,
                    bytes=size,
                    row=row,
                )
            )
    return records


def station_key(row: dict[str, str], prefix: str) -> str:
    return ".".join(
        [
            row[f"{prefix}_network"],
            row[f"{prefix}_station"],
            row[f"{prefix}_location"],
        ]
    )


def _resolve_row_path(text: str, tsv_path: Path, cache: dict[str, Path]) -> Path:
    cached = cache.get(text)
    if cached is not None:
        return cached

    path = Path(text).expanduser()
    if not path.is_absolute():
        path = tsv_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"XC pack file is missing for {tsv_path}: {path}")
    cache[text] = path
    return path
