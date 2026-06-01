from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import BinaryIO

from fastxc.io.spack import SegspecSource
from fastxc.io.xcspec import (
    COMPLEX64_BYTES,
    SEGSPEC_HEADER_STRUCT,
    pack_header,
    pack_source_entry,
)
from .manifest import write_text_atomic

INDEX_NAME = "xcspec_index.tsv"


@dataclass(frozen=True)
class XCacheShard:
    timestamp: str
    xcspec_path: Path
    manifest_path: Path
    file_count: int
    source_nstep: int
    window_start: int
    window_count: int
    nstep: int
    nspec: int
    nfft: int
    dt: float
    df: float
    payload_offset: int
    step_bytes: int
    manifest_hash: str


def write_xcspec(
    path: Path,
    *,
    timestamp: str,
    sources: list[SegspecSource],
    nstep: int,
    nspec: int,
    dt: float,
    df: float,
    source_table_offset: int,
    payload_offset: int,
    step_bytes: int,
    payload_bytes: int,
    manifest_hash_u64: int,
    source_step_start: int = 0,
) -> None:
    source_step_bytes = nspec * COMPLEX64_BYTES
    tmp = path.with_suffix(path.suffix + ".tmp")
    handles: dict[Path, BinaryIO] = {}

    try:
        with tmp.open("wb") as out:
            out.write(
                pack_header(
                    timestamp=timestamp,
                    file_count=len(sources),
                    nstep=nstep,
                    nspec=nspec,
                    dt=dt,
                    df=df,
                    source_table_offset=source_table_offset,
                    payload_offset=payload_offset,
                    step_bytes=step_bytes,
                    payload_bytes=payload_bytes,
                    manifest_hash_u64=manifest_hash_u64,
                )
            )
            for source in sources:
                out.write(
                    pack_source_entry(
                        file_index=source.file_index,
                        nsl_id=source.nsl_id,
                        stla=source.stla,
                        stlo=source.stlo,
                        network=source.network,
                        station=source.station,
                        location=source.location,
                        component=source.component,
                    )
                )

            pad = payload_offset - out.tell()
            if pad < 0:
                raise ValueError("XCache payload offset is before source table end")
            out.write(b"\0" * pad)

            for step in range(nstep):
                for source in sources:
                    out.write(
                        _read_source_step(
                            source,
                            source_step_start + step,
                            source_step_bytes,
                            handles,
                        )
                    )
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles.values():
            handle.close()


def write_xcspec_index(cache_dir: Path, shards: list[XCacheShard]) -> Path:
    path = cache_dir / INDEX_NAME
    lines = [
        "timestamp\txcspec_path\tmanifest_path\tfile_count\tnstep\tnspec\t"
        "nfft\tdt\tdf\tpayload_offset\tstep_bytes\tmanifest_hash\t"
        "window_start\twindow_count\tsource_nstep\n"
    ]
    for shard in shards:
        lines.append(
            f"{shard.timestamp}\t{shard.xcspec_path.resolve().as_posix()}\t"
            f"{shard.manifest_path.resolve().as_posix()}\t{shard.file_count}\t"
            f"{shard.nstep}\t{shard.nspec}\t{shard.nfft}\t"
            f"{shard.dt:.9g}\t{shard.df:.9g}\t{shard.payload_offset}\t"
            f"{shard.step_bytes}\t{shard.manifest_hash}\t"
            f"{shard.window_start}\t{shard.window_count}\t{shard.source_nstep}\n"
        )
    write_text_atomic(path, "".join(lines))
    return path


def _read_source_step(
    source: SegspecSource,
    step: int,
    step_bytes: int,
    handles: dict[Path, BinaryIO],
) -> bytes:
    path = Path(source.pack_path or source.segspec_path)
    base_offset = int(source.pack_offset) if source.pack_path else 0
    offset = base_offset + SEGSPEC_HEADER_STRUCT.size + step * step_bytes
    handle = handles.get(path)
    if handle is None:
        handle = path.open("rb")
        handles[path] = handle
    handle.seek(offset)
    data = handle.read(step_bytes)
    if len(data) != step_bytes:
        raise ValueError(f"Short SEGSPEC step read: {path} step={step}")
    return data
