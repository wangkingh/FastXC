from __future__ import annotations

import csv
import heapq
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Sequence, TextIO

import numpy as np
from tqdm import tqdm

from fastxc.io import (
    ITIME,
    SacHeader,
    encode_sac_record,
    read_sac_record,
    read_sac_record_from,
)
from fastxc.io.sourcepack import SOURCEPACK_INDEX_HEADER

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StackResult:
    input_path: Path
    output_path: Path
    segment_count: int
    sample_count: int


@dataclass(frozen=True)
class SourceIndexRecord:
    row: dict[str, str]
    record_path: Path
    record_offset: int
    record_bytes: int

    @property
    def source_key(self) -> str:
        return self.row["source_key"]

    @property
    def receiver_key(self) -> str:
        return self.row["receiver_key"]

    @property
    def component_pair(self) -> str:
        return f"{self.row['src_component']}-{self.row['rec_component']}"


@dataclass(frozen=True)
class _SourceIndexStream:
    handle: TextIO
    records: Iterator[SourceIndexRecord]


def _parse_final_pair_name(path: Path) -> tuple[str, str, str]:
    parts = path.name.split(".")
    if parts and parts[-1].lower() == "sac":
        parts = parts[:-1]
    if len(parts) < 3:
        raise ValueError(f"Unexpected final pair path: {path.name}")
    return parts[0], parts[1], parts[2]


def linear_output_path(final_pair_path: str | Path, stack_dir: str | Path) -> Path:
    final_pair_path = Path(final_pair_path)
    stack_dir = Path(stack_dir).expanduser().resolve()
    net_pair, sta_pair, cmp_pair = _parse_final_pair_name(final_pair_path)
    pair_name = f"{net_pair}.{sta_pair}"
    return stack_dir / "linearstack" / pair_name / f"{pair_name}.{cmp_pair}.linearstack.sac"


def linear_sourcepack_dir(stack_dir: str | Path) -> Path:
    return Path(stack_dir).expanduser().resolve() / "linearstack_sourcepack" / "STACK"


def _stacked_header(first_header: SacHeader, data_count: int, divisor: int) -> SacHeader:
    out_header = first_header.copy()
    out_header.set_int("npts", int(data_count))
    out_header.set_int("iftype", ITIME)
    out_header.set_float("user0", float(divisor))
    out_header.set_int("nzyear", 2010)
    out_header.set_int("nzjday", 214)
    out_header.set_int("nzhour", 0)
    out_header.set_int("nzmin", 0)
    out_header.set_int("nzsec", 0)
    out_header.set_int("nzmsec", 0)
    return out_header


def _fallback_final_pair_path(row: dict[str, str]) -> Path:
    net_pair = f"{row['src_network']}-{row['rec_network']}"
    sta_pair = f"{row['src_station']}-{row['rec_station']}"
    cmp_pair = f"{row['src_component']}-{row['rec_component']}"
    return Path(f"{net_pair}.{sta_pair}.{cmp_pair}.sac")


def _read_source_index(path: str | Path) -> list[SourceIndexRecord]:
    with _open_source_index_stream(path) as stream:
        return list(stream.records)


class _SourceIndexStreamContext:
    def __init__(self, path: str | Path):
        self.index_path = Path(path).expanduser().resolve()
        self.handle: TextIO | None = None

    def __enter__(self) -> _SourceIndexStream:
        self.handle = self.index_path.open("r", encoding="utf-8-sig", newline="")
        try:
            reader = csv.DictReader(self.handle, delimiter="\t")
            if reader.fieldnames is None:
                return _SourceIndexStream(self.handle, iter(()))

            required = {
                "source_key",
                "receiver_key",
                "record_path",
                "record_offset",
                "bytes",
                "src_component",
                "rec_component",
            }
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"Source index missing fields {sorted(missing)}: {self.index_path}")
            return _SourceIndexStream(self.handle, _iter_source_index_rows(reader, self.index_path))
        except Exception:
            self.handle.close()
            self.handle = None
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.close()


def _iter_source_index_rows(reader: csv.DictReader, index_path: Path) -> Iterator[SourceIndexRecord]:
    for row in reader:
        record_path = Path(row["record_path"]).expanduser()
        if not record_path.is_absolute():
            record_path = index_path.parent / record_path
        yield SourceIndexRecord(
            row=row,
            record_path=record_path,
            record_offset=int(row["record_offset"]),
            record_bytes=int(row["bytes"]),
        )


def _open_source_index_stream(path: str | Path) -> _SourceIndexStreamContext:
    return _SourceIndexStreamContext(path)


def _stream_sourcepack_groups(indexes: Iterable[str | Path]) -> Iterator[tuple[tuple[str, int], list[SourceIndexRecord]]]:
    contexts: list[_SourceIndexStreamContext] = []
    heap: list[tuple[tuple[str, int], int, SourceIndexRecord, Iterator[SourceIndexRecord]]] = []
    counter = 0

    try:
        for index in indexes:
            context = _open_source_index_stream(index)
            stream = context.__enter__()
            contexts.append(context)
            record = _next_record(stream.records)
            if record is None:
                continue
            heapq.heappush(heap, (_record_group_key(record), counter, record, stream.records))
            counter += 1

        while heap:
            key = heap[0][0]
            records: list[SourceIndexRecord] = []
            while heap and heap[0][0] == key:
                _, _, record, iterator = heapq.heappop(heap)
                records.append(record)
                next_record = _next_record(iterator)
                if next_record is not None:
                    heapq.heappush(heap, (_record_group_key(next_record), counter, next_record, iterator))
                    counter += 1
            records.sort(key=lambda item: (item.row.get("timestamp", ""), item.record_path.as_posix(), item.record_offset))
            yield key, records
    finally:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


def _next_record(iterator: Iterator[SourceIndexRecord]) -> SourceIndexRecord | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _record_group_key(record: SourceIndexRecord) -> tuple[str, int]:
    return (_row_path_id(record.row), _row_component_slot(record.row))


def _record_final_pair(record: SourceIndexRecord) -> str:
    return record.row.get("final_pair_path") or str(_fallback_final_pair_path(record.row))


def _row_path_id(row: dict[str, str]) -> str:
    path_id = row.get("path_id", "").strip()
    if path_id:
        return path_id.zfill(8) if path_id.isdigit() else path_id
    try:
        return f"{int(row['src_id']) * 10000 + int(row['rec_id']):08d}"
    except (KeyError, ValueError):
        return ""


def _row_component_slot(row: dict[str, str]) -> int:
    text = row.get("component_slot", "").strip()
    if text.isdigit():
        return int(text)
    return _component_rank(row.get("src_component", "")) * 3 + _component_rank(row.get("rec_component", ""))


def _component_rank(component: str) -> int:
    tail = component.upper()[-1:] if component else ""
    return {"E": 0, "1": 0, "R": 0, "N": 1, "2": 1, "T": 1, "Z": 2, "3": 2}.get(tail, 100)


def _linear_stack_group(
    records: Sequence[SourceIndexRecord],
    handles: dict[Path, BinaryIO] | None = None,
) -> tuple[SacHeader, np.ndarray]:
    first_header, first_data = _read_source_record(records[0], handles)
    data_count = first_header.data_count
    delta = first_header.get_float("delta")
    accum = first_data.astype(np.float64, copy=False)

    for record in records[1:]:
        header, data = _read_source_record(record, handles)
        if header.data_count != data_count:
            raise ValueError(f"{record.record_path}: data_count mismatch ({header.data_count} vs {data_count})")
        if not np.isclose(header.get_float("delta"), delta, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"{record.record_path}: delta mismatch ({header.get_float('delta')} vs {delta})")
        accum += data.astype(np.float64, copy=False)

    divisor = len(records)
    stacked = (accum / float(divisor)).astype(np.float32, copy=False)
    return _stacked_header(first_header, data_count, divisor), stacked


def _read_source_record(
    record: SourceIndexRecord,
    handles: dict[Path, BinaryIO] | None,
) -> tuple[SacHeader, np.ndarray]:
    if handles is None:
        return read_sac_record(record.record_path, record.record_offset, record.record_bytes)

    handle = handles.get(record.record_path)
    if handle is None:
        handle = record.record_path.open("rb")
        handles[record.record_path] = handle
    return read_sac_record_from(handle, record.record_path, record.record_offset, record.record_bytes)


def _close_binary_handles(handles: dict[Path, BinaryIO]) -> None:
    for handle in handles.values():
        handle.close()


def _linear_index_row(
    template: dict[str, str],
    *,
    pack_path: Path,
    record_offset: int,
    record_bytes: int,
    final_pair_path: Path,
    header: SacHeader,
) -> dict[str, str]:
    row = {field: "" for field in SOURCEPACK_INDEX_HEADER}
    for key in (
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
        "dist",
        "az",
        "baz",
    ):
        row[key] = template.get(key, "")
    row.update(
        {
            "timestamp": "STACK",
            "npts": str(header.npts),
            "dt": f"{header.get_float('delta'):.9g}",
            "record_path": pack_path.as_posix(),
            "record_offset": str(record_offset),
            "bytes": str(record_bytes),
            "storage_kind": "linearstack_pack",
            "final_pair_path": final_pair_path.as_posix(),
        }
    )
    return row


def linear_stack_sourcepack_indexes(
    indexes: Iterable[str | Path],
    stack_dir: str | Path,
    *,
    dry_run: bool = False,
) -> list[StackResult]:
    indexes = list(indexes)

    out_dir = linear_sourcepack_dir(stack_dir)
    pack_path = out_dir / "linearstack.pack"
    index_path = out_dir / "sourcepack_index.tsv"
    if dry_run:
        results: list[StackResult] = []
        for _, records in _stream_sourcepack_groups(indexes):
            final_pair = _record_final_pair(records[0])
            results.append(StackResult(Path(final_pair), pack_path, len(records), 0))
        return results

    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_pack = tmp_dir / pack_path.name
    tmp_index = tmp_dir / index_path.name

    results: list[StackResult] = []
    handles: dict[Path, BinaryIO] = {}
    try:
        with tmp_pack.open("wb") as pack_handle, tmp_index.open("w", encoding="utf-8", newline="") as index_handle:
            writer = csv.DictWriter(index_handle, fieldnames=SOURCEPACK_INDEX_HEADER, delimiter="\t")
            writer.writeheader()
            for _, records in tqdm(
                _stream_sourcepack_groups(indexes),
                desc="Linear stack",
                unit="pair",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            ):
                final_pair = _record_final_pair(records[0])
                header, stacked = _linear_stack_group(records, handles)
                record_offset = pack_handle.tell()
                data = encode_sac_record(header, stacked)
                pack_handle.write(data)
                linear_path = linear_output_path(final_pair, stack_dir)
                writer.writerow(
                    _linear_index_row(
                        records[0].row,
                        pack_path=pack_path,
                        record_offset=record_offset,
                        record_bytes=len(data),
                        final_pair_path=linear_path,
                        header=header,
                    )
                )
                results.append(StackResult(Path(final_pair), linear_path, len(records), header.data_count))
    finally:
        _close_binary_handles(handles)

    if not results:
        if tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir)
        return []

    (tmp_dir / "_SUCCESS").write_text(
        f"records\t{len(results)}\npack\t{pack_path}\nindex\t{index_path}\n",
        encoding="utf-8",
    )
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    tmp_dir.replace(out_dir)
    return results
