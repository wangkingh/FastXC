from __future__ import annotations

import logging
from pathlib import Path
import re

from fastxc.io.spack import SegspecSource, read_list, read_sources, read_spack_sources
from fastxc.io.xcspec import (
    COMPLEX64_BYTES,
    XCSPEC_HEADER_SIZE,
    XCSPEC_SOURCE_ENTRY_SIZE,
    align_offset,
)
from .manifest import can_reuse, manifest_hash, write_manifest
from .timestamp import normalize_timestamp
from .writer import XCacheShard, write_xcspec, write_xcspec_index

log = logging.getLogger(__name__)


def build_xcache(
    sac2spec_root: str | Path,
    xcache_dir: str | Path | None = None,
    *,
    windows_per_xcache: int | None = None,
) -> Path:
    """Build one step-major .xcspec shard for each SAC2SPEC timestamp."""

    root = Path(sac2spec_root).expanduser().resolve()
    cache_dir = Path(xcache_dir).expanduser().resolve() if xcache_dir else root / "xcache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    shards: list[XCacheShard] = []
    spack_sources = read_spack_sources(root)
    if spack_sources is not None:
        source_index_path = root / "spack_by_timestamp"
        if not (source_index_path / "_SUCCESS").is_file():
            source_index_path = root / "spack"
        for source_timestamp, sources in sorted(spack_sources.items()):
            shards.extend(
                build_one_xcache_from_sources(
                    source_timestamp,
                    sources,
                    source_index_path,
                    cache_dir,
                    windows_per_xcache=windows_per_xcache,
                )
            )
        index_path = write_xcspec_index(cache_dir, shards)
        log.info("XCache ready: %s (%d shard(s)).", index_path, len(shards))
        return index_path

    speclists = _load_or_create_speclists(root)
    for speclist in speclists:
        shards.extend(
            build_one_xcache(
                speclist,
                cache_dir,
                windows_per_xcache=windows_per_xcache,
            )
        )

    index_path = write_xcspec_index(cache_dir, shards)
    log.info("XCache ready: %s (%d shard(s)).", index_path, len(shards))
    return index_path


def build_one_xcache(
    speclist_path: str | Path,
    xcache_dir: str | Path,
    *,
    windows_per_xcache: int | None = None,
) -> list[XCacheShard]:
    """Build one or more .xcspec shards from one timestamp speclist."""

    speclist = Path(speclist_path).expanduser().resolve()
    source_timestamp = speclist.stem
    timestamp = normalize_timestamp(source_timestamp)
    cache_dir = Path(xcache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    sources = read_sources(speclist, source_timestamp)
    return build_one_xcache_from_sources(
        source_timestamp,
        sources,
        speclist,
        cache_dir,
        windows_per_xcache=windows_per_xcache,
    )


def build_one_xcache_from_sources(
    source_timestamp: str,
    sources: list[SegspecSource],
    source_index_path: Path,
    xcache_dir: Path,
    *,
    windows_per_xcache: int | None = None,
) -> list[XCacheShard]:
    timestamp = normalize_timestamp(source_timestamp)
    cache_dir = Path(xcache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_nstep = sources[0].nstep
    nspec = sources[0].nspec
    nfft = 2 * (nspec - 1)
    dt = sources[0].dt
    df = sources[0].df

    source_step_bytes = nspec * COMPLEX64_BYTES
    step_bytes = len(sources) * source_step_bytes
    source_table_offset = XCSPEC_HEADER_SIZE
    payload_offset = align_offset(source_table_offset + len(sources) * XCSPEC_SOURCE_ENTRY_SIZE)

    stem = _sanitize_timestamp(timestamp)
    shards: list[XCacheShard] = []
    for window_start, window_count in _window_ranges(source_nstep, windows_per_xcache):
        nstep = window_count
        payload_bytes = nstep * step_bytes
        chunk_stem = _window_stem(stem, window_start, window_count, source_nstep)
        xcspec_path = cache_dir / f"{chunk_stem}.xcspec"
        manifest_path = cache_dir / f"{chunk_stem}.xcspec.json"
        hash_value = manifest_hash(
            timestamp=timestamp,
            speclist_path=source_index_path,
            sources=sources,
            payload_offset=payload_offset,
            step_bytes=step_bytes,
            payload_bytes=payload_bytes,
            source_nstep=source_nstep,
            window_start=window_start,
            window_count=window_count,
        )

        shard = XCacheShard(
            timestamp=timestamp,
            xcspec_path=xcspec_path,
            manifest_path=manifest_path,
            file_count=len(sources),
            source_nstep=source_nstep,
            window_start=window_start,
            window_count=window_count,
            nstep=nstep,
            nspec=nspec,
            nfft=nfft,
            dt=dt,
            df=df,
            payload_offset=payload_offset,
            step_bytes=step_bytes,
            manifest_hash=hash_value,
        )
        shards.append(shard)

        if can_reuse(manifest_path, xcspec_path, hash_value, payload_offset + payload_bytes):
            continue

        write_xcspec(
            xcspec_path,
            timestamp=timestamp,
            sources=sources,
            nstep=nstep,
            nspec=nspec,
            dt=dt,
            df=df,
            source_table_offset=source_table_offset,
            payload_offset=payload_offset,
            step_bytes=step_bytes,
            payload_bytes=payload_bytes,
            manifest_hash_u64=int.from_bytes(bytes.fromhex(hash_value[:16]), "little"),
            source_step_start=window_start,
        )
        write_manifest(
            manifest_path,
            timestamp=timestamp,
            xcspec_path=xcspec_path,
            speclist_path=source_index_path,
            sources=sources,
            nstep=nstep,
            nspec=nspec,
            dt=dt,
            df=df,
            source_table_offset=source_table_offset,
            payload_offset=payload_offset,
            step_bytes=step_bytes,
            payload_bytes=payload_bytes,
            manifest_hash_value=hash_value,
            source_nstep=source_nstep,
            window_start=window_start,
            window_count=window_count,
        )
    return shards


def _window_ranges(source_nstep: int, windows_per_xcache: int | None) -> list[tuple[int, int]]:
    if windows_per_xcache is None or windows_per_xcache >= source_nstep:
        return [(0, source_nstep)]
    if windows_per_xcache < 1:
        raise ValueError("windows_per_xcache must be >= 1 or AUTO")
    return [
        (start, min(windows_per_xcache, source_nstep - start))
        for start in range(0, source_nstep, windows_per_xcache)
    ]


def _window_stem(stem: str, window_start: int, window_count: int, source_nstep: int) -> str:
    if window_start == 0 and window_count == source_nstep:
        return stem
    window_end = window_start + window_count - 1
    return f"{stem}.w{window_start:06d}-{window_end:06d}"


def _sanitize_timestamp(timestamp: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._:-]+", "-", timestamp).strip("-")
    return text or "timestamp"


def _load_or_create_speclists(root: Path) -> list[Path]:
    index_path = root / "timestamp_speclist_index.txt"
    if index_path.is_file():
        return read_list(index_path)

    segspec_root = root / "segspec"
    if not segspec_root.is_dir():
        raise FileNotFoundError(f"XCache SEGSPEC directory not found: {segspec_root}")

    speclist_dir = root / "speclists"
    speclist_dir.mkdir(parents=True, exist_ok=True)
    speclists: list[Path] = []

    for timestamp_dir in sorted(path for path in segspec_root.iterdir() if path.is_dir()):
        sources = sorted(timestamp_dir.glob("*.SEGSPEC"))
        if not sources:
            continue
        speclist = speclist_dir / f"{timestamp_dir.name}.txt"
        speclist.write_text(
            "".join(f"{source.resolve().as_posix()}\n" for source in sources),
            encoding="utf-8",
        )
        speclists.append(speclist.resolve())

    if not speclists:
        raise ValueError(f"No SEGSPEC files found under {segspec_root}")

    index_path.write_text(
        "".join(f"{speclist.as_posix()}\n" for speclist in speclists),
        encoding="utf-8",
    )
    return speclists
