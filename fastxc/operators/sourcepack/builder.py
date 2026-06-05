from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil
import threading

from fastxc.io.sourcepack import SOURCEPACK_INDEX_HEADER, XcPackRecord, read_xcpack_tsv
from fastxc.runtime import write_progress_file

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourcePackBuildResult:
    output_dir: Path
    index_paths: list[Path]
    record_count: int
    source_count: int
    bytes_indexed: int


class AsyncSourcePackMaterializer:
    """Watch native XC timestamp markers and materialize sourcepack indexes."""

    def __init__(
        self,
        xc_output_root: str | Path,
        sourcepack_dir: str | Path,
        *,
        sort_within_source: bool = True,
        progress_file: str | Path | None = None,
        poll_interval_sec: float = 5.0,
    ):
        self.xcpack_dir = _candidate_xcpack_dir(Path(xc_output_root).expanduser().resolve())
        self.output_dir = Path(sourcepack_dir).expanduser().resolve()
        self.sort_within_source = sort_within_source
        self.progress_file = progress_file
        self.poll_interval_sec = max(0.5, float(poll_interval_sec))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._done: set[str] = set()
        self._errors: list[BaseException] = []
        self._record_count = 0
        self._source_count = 0
        self._bytes_indexed = 0
        self._lock = threading.Lock()
        self._total_expected = _expected_timestamp_count(self.xcpack_dir)

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_progress_file(
            self.progress_file,
            "RUNNING",
            0,
            self._total_expected,
            "timestamps",
            "async source index build",
        )
        self._thread = threading.Thread(target=self._run, name="sourcepack-materializer", daemon=True)
        self._thread.start()

    def finish(
        self,
        *,
        mark_success: bool = True,
        status_message: str | None = None,
    ) -> SourcePackBuildResult:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._scan_once()
        if self._errors:
            write_progress_file(
                self.progress_file,
                "FAILED",
                len(self._done),
                self._total_expected,
                "timestamps",
                str(self._errors[0]),
            )
            raise RuntimeError("async SourcePack materialization failed") from self._errors[0]

        index_paths = sorted(self.output_dir.glob("*/sourcepack_index.tsv"))
        success_path = self.output_dir / "_SUCCESS"
        if mark_success:
            success_path.write_text(
                f"timestamps\t{len(index_paths)}\nrecords\t{self._record_count}\n"
                f"sources\t{self._source_count}\nbytes_indexed\t{self._bytes_indexed}\n",
                encoding="utf-8",
            )
            write_progress_file(
                self.progress_file,
                "DONE",
                len(index_paths),
                self._total_expected or len(index_paths),
                "timestamps",
                "source index ready",
            )
        else:
            success_path.unlink(missing_ok=True)
            write_progress_file(
                self.progress_file,
                "FAILED",
                len(index_paths),
                self._total_expected or len(index_paths),
                "timestamps",
                status_message or "upstream XC did not complete; global success marker withheld",
            )
        return SourcePackBuildResult(
            output_dir=self.output_dir,
            index_paths=index_paths,
            record_count=self._record_count,
            source_count=self._source_count,
            bytes_indexed=self._bytes_indexed,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan_once()
            except BaseException as exc:  # store and let the main thread fail cleanly
                self._errors.append(exc)
                self._stop.set()
                return
            self._stop.wait(self.poll_interval_sec)

    def _scan_once(self) -> None:
        if not self.xcpack_dir.is_dir():
            return
        for done_path in sorted(self.xcpack_dir.glob("*.done")):
            if done_path.name.startswith("_"):
                continue
            timestamp = done_path.name[: -len(".done")]
            with self._lock:
                if timestamp in self._done:
                    continue
            result = build_sourcepack_timestamp(
                self.xcpack_dir,
                self.output_dir,
                timestamp,
                sort_within_source=self.sort_within_source,
            )
            with self._lock:
                self._done.add(timestamp)
                self._record_count += result.record_count
                self._source_count += result.source_count
                self._bytes_indexed += result.bytes_indexed
                completed = len(self._done)
            write_progress_file(
                self.progress_file,
                "RUNNING",
                completed,
                self._total_expected,
                "timestamps",
                f"{timestamp}: {result.record_count} records",
            )


def build_sourcepack(
    xc_output_root: str | Path,
    sourcepack_dir: str | Path | None = None,
    *,
    sort_within_source: bool = True,
    progress_file: str | Path | None = None,
) -> SourcePackBuildResult:
    """Build source-grouped indexes over native XC pack output.

    This stage does not copy NCF bytes. The native ``xcpack`` files remain the
    binary storage, and the generated index provides source/receiver/component
    lookup with byte offsets into those packs.
    """

    xcpack_dir = _find_xcpack_dir(Path(xc_output_root).expanduser().resolve())
    output_dir = (
        Path(sourcepack_dir).expanduser().resolve()
        if sourcepack_dir is not None
        else xcpack_dir.parent / "sourcepack"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_timestamp: dict[str, list[XcPackRecord]] = {}
    sequence = 0
    tsv_paths = sorted(xcpack_dir.glob("*.tsv"))
    for tsv_path in tsv_paths:
        for record in read_xcpack_tsv(tsv_path, sequence):
            records_by_timestamp.setdefault(record.timestamp, []).append(record)
            sequence += 1

    expected_timestamps = _discover_expected_timestamps(xcpack_dir, records_by_timestamp)
    if not expected_timestamps:
        raise FileNotFoundError(f"No XC pack timestamps found in {xcpack_dir}")

    total_timestamps = len(expected_timestamps)
    write_progress_file(progress_file, "RUNNING", 0, total_timestamps, "timestamps", "source index build")

    index_paths: list[Path] = []
    total_records = 0
    total_sources = 0
    total_bytes = 0
    for done, timestamp in enumerate(expected_timestamps, start=1):
        records = records_by_timestamp.get(timestamp, [])
        result = _build_one_timestamp(
            timestamp,
            records,
            output_dir,
            sort_within_source=sort_within_source,
        )
        index_paths.extend(result.index_paths)
        total_records += result.record_count
        total_sources += result.source_count
        total_bytes += result.bytes_indexed
        write_progress_file(
            progress_file,
            "RUNNING",
            done,
            total_timestamps,
            "timestamps",
            f"{timestamp}: {result.record_count} records",
        )

    write_progress_file(progress_file, "DONE", total_timestamps, total_timestamps, "timestamps", "source index ready")
    log.info(
        "Source index ready: %s (%d timestamp(s), %d source(s), %d record(s)).",
        output_dir,
        len(index_paths),
        total_sources,
        total_records,
    )
    return SourcePackBuildResult(
        output_dir=output_dir,
        index_paths=index_paths,
        record_count=total_records,
        source_count=total_sources,
        bytes_indexed=total_bytes,
    )


def build_sourcepack_timestamp(
    xc_output_root: str | Path,
    sourcepack_dir: str | Path,
    timestamp: str,
    *,
    sort_within_source: bool = True,
) -> SourcePackBuildResult:
    """Build one ``sourcepack/<timestamp>`` index from completed XC pack shards."""

    xcpack_dir = _find_xcpack_dir(Path(xc_output_root).expanduser().resolve())
    output_dir = Path(sourcepack_dir).expanduser().resolve()
    sequence = 0
    records: list[XcPackRecord] = []
    for tsv_path in _timestamp_tsv_paths(xcpack_dir, timestamp):
        for record in read_xcpack_tsv(tsv_path, sequence):
            sequence += 1
            if record.timestamp == timestamp:
                records.append(record)
    return _build_one_timestamp(
        timestamp,
        records,
        output_dir,
        sort_within_source=sort_within_source,
    )


def _build_one_timestamp(
    timestamp: str,
    records: list[XcPackRecord],
    output_dir: Path,
    *,
    sort_within_source: bool,
) -> SourcePackBuildResult:
    target_dir = output_dir / _safe_name(timestamp)
    tmp_dir = output_dir / f".{_safe_name(timestamp)}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    records = sorted(records, key=_sort_key if sort_within_source else lambda record: record.sequence)
    index_path = tmp_dir / "sourcepack_index.tsv"
    source_keys: set[str] = set()
    bytes_indexed = 0
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCEPACK_INDEX_HEADER, delimiter="\t")
        writer.writeheader()
        for record in records:
            source_keys.add(record.source_key)
            bytes_indexed += record.bytes
            writer.writerow(_index_row(record))

    (tmp_dir / "_SUCCESS").write_text(
        f"timestamp\t{timestamp}\nrecords\t{len(records)}\nsources\t{len(source_keys)}\nbytes_indexed\t{bytes_indexed}\n",
        encoding="utf-8",
    )
    if target_dir.exists():
        shutil.rmtree(target_dir)
    tmp_dir.replace(target_dir)

    return SourcePackBuildResult(
        output_dir=target_dir,
        index_paths=[target_dir / "sourcepack_index.tsv"],
        record_count=len(records),
        source_count=len(source_keys),
        bytes_indexed=bytes_indexed,
    )


def _index_row(record: XcPackRecord) -> dict[str, str]:
    row = record.row
    return {
        "timestamp": record.timestamp,
        "path_id": _path_id_text(row),
        "component_slot": str(_component_slot(row["src_component"], row["rec_component"])),
        "source_key": record.source_key,
        "receiver_key": record.receiver_key,
        "src_id": row["src_id"],
        "rec_id": row["rec_id"],
        "src_network": row["src_network"],
        "src_station": row["src_station"],
        "src_location": row["src_location"],
        "src_component": row["src_component"],
        "rec_network": row["rec_network"],
        "rec_station": row["rec_station"],
        "rec_location": row["rec_location"],
        "rec_component": row["rec_component"],
        "npts": row["npts"],
        "dt": row["dt"],
        "dist": row["dist"],
        "az": row["az"],
        "baz": row["baz"],
        "record_path": record.pack_path.as_posix(),
        "record_offset": str(record.offset),
        "bytes": str(record.bytes),
        "storage_kind": "xcpack",
        "final_pair_path": row.get("final_pair_path", ""),
    }


def _discover_expected_timestamps(
    xcpack_dir: Path,
    records_by_timestamp: dict[str, list[XcPackRecord]],
) -> list[str]:
    timestamps = set(records_by_timestamp)
    timestamps.update(_read_inventory_timestamps(xcpack_dir))

    for done_path in xcpack_dir.glob("*.done"):
        if done_path.name.startswith("_"):
            continue
        timestamps.add(done_path.name[: -len(".done")])

    return sorted(timestamps)


def _expected_timestamp_count(xcpack_dir: Path) -> int:
    timestamps = _read_inventory_timestamps(xcpack_dir)
    if timestamps:
        return len(timestamps)
    return len([path for path in xcpack_dir.glob("*.done") if not path.name.startswith("_")])


def _read_inventory_timestamps(xcpack_dir: Path) -> list[str]:
    index_path = _workspace_root_from_xcpack(xcpack_dir) / "manifest" / "timestamp_index.tsv"
    if not index_path.is_file():
        return []

    timestamps: list[str] = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "timestamp" not in reader.fieldnames:
            return []
        for row in reader:
            timestamp = row.get("timestamp", "").strip()
            if timestamp:
                timestamps.append(timestamp)
    return sorted(set(timestamps))


def _workspace_root_from_xcpack(xcpack_dir: Path) -> Path:
    if xcpack_dir.parent.name == "ncf":
        return xcpack_dir.parent.parent
    return xcpack_dir.parent


def _timestamp_tsv_paths(xcpack_dir: Path, timestamp: str) -> list[Path]:
    paths = [
        path
        for path in xcpack_dir.glob("*.tsv")
        if path.name.startswith(timestamp)
    ]
    if paths:
        return sorted(paths)
    return sorted(xcpack_dir.glob("*.tsv"))


def _find_xcpack_dir(root: Path) -> Path:
    candidate = _candidate_xcpack_dir(root)
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"XC pack directory not found: {root} or {candidate}")


def _candidate_xcpack_dir(root: Path) -> Path:
    return root if root.name == "xcpack" else root / "xcpack"


def _sort_key(record: XcPackRecord) -> tuple[str, int, str, str, int]:
    return (
        _path_id_text(record.row),
        _component_slot(record.row["src_component"], record.row["rec_component"]),
        record.source_key,
        record.receiver_key,
        record.sequence,
    )


def _component_rank(component: str) -> int:
    tail = component.upper()[-1:] if component else ""
    return {"E": 0, "1": 0, "N": 1, "2": 1, "Z": 2, "3": 2, "R": 3, "T": 4}.get(tail, 100)


def _component_slot(src_component: str, rec_component: str) -> int:
    return _component_rank(src_component) * 3 + _component_rank(rec_component)


def _path_id_text(row: dict[str, str]) -> str:
    path_id = row.get("path_id", "").strip()
    if path_id:
        return path_id.zfill(8) if path_id.isdigit() else path_id
    return f"{int(row['src_id']) * 10000 + int(row['rec_id']):08d}"


def _safe_name(text: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z._:-]+", "-", text).strip("-")
    return clean or "item"
