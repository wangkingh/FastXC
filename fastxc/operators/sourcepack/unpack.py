from __future__ import annotations

import concurrent.futures
import csv
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from fastxc.io.sourcepack import discover_sourcepack_indexes


@dataclass(frozen=True)
class UnpackRecord:
    output_path: Path
    record_path: Path
    record_offset: int
    record_bytes: int


@dataclass(frozen=True)
class UnpackResult:
    output_dir: Path
    file_count: int
    record_count: int
    bytes_written: int


def unpack_sourcepack(
    sourcepack_input: str | Path,
    output_dir: str | Path,
    *,
    max_workers: int = 1,
    dry_run: bool = False,
) -> UnpackResult:
    indexes = discover_sourcepack_indexes(sourcepack_input)
    output_root = Path(output_dir).expanduser().resolve()
    groups = _group_unpack_records(indexes, output_root)
    if dry_run:
        return UnpackResult(output_root, len(groups), sum(len(v) for v in groups.values()), 0)

    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [(path, records) for path, records in sorted(groups.items())]
    max_workers = max(1, int(max_workers))

    if max_workers == 1 or len(tasks) <= 1:
        results = [
            _write_group(task)
            for task in tqdm(
                tasks,
                desc="Unpack sourcepack",
                unit="file",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            )
        ]
    else:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_write_group, task) for task in tasks]
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Unpack sourcepack",
                unit="file",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            ):
                results.append(fut.result())

    return UnpackResult(
        output_root,
        file_count=len(results),
        record_count=sum(item[0] for item in results),
        bytes_written=sum(item[1] for item in results),
    )


def _group_unpack_records(indexes: list[Path], output_root: Path) -> dict[Path, list[UnpackRecord]]:
    groups: dict[Path, list[UnpackRecord]] = {}
    for index in indexes:
        with index.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                continue
            required = {"record_path", "record_offset", "bytes"}
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"SourcePack index missing fields {sorted(missing)}: {index}")

            for row in reader:
                output_path = _unpack_output_path(row, output_root)
                record_path = Path(row["record_path"]).expanduser()
                if not record_path.is_absolute():
                    record_path = index.parent / record_path
                groups.setdefault(output_path, []).append(
                    UnpackRecord(
                        output_path=output_path,
                        record_path=record_path,
                        record_offset=int(row["record_offset"]),
                        record_bytes=int(row["bytes"]),
                    )
                )
    return groups


def _unpack_output_path(row: dict[str, str], output_root: Path) -> Path:
    src_dir = f"{row['src_network']}.{row['src_station']}"
    rec_dir = f"{row['rec_network']}.{row['rec_station']}"
    name = (
        f"{row['src_network']}-{row['rec_network']}."
        f"{row['src_station']}-{row['rec_station']}."
        f"{row['src_component']}-{row['rec_component']}.ncf.SAC"
    )
    return output_root / src_dir / rec_dir / name


def _write_group(task: tuple[Path, list[UnpackRecord]]) -> tuple[int, int]:
    output_path, records = task
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0

    try:
        with tmp_path.open("wb") as out:
            for record in records:
                with record.record_path.open("rb") as src:
                    src.seek(record.record_offset)
                    data = src.read(record.record_bytes)
                if len(data) != record.record_bytes:
                    raise ValueError(
                        f"Truncated sourcepack record: {record.record_path} "
                        f"offset={record.record_offset} bytes={record.record_bytes}"
                    )
                out.write(data)
                bytes_written += len(data)
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return len(records), bytes_written
