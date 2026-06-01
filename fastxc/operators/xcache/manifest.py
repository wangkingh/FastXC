from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from fastxc.io.spack import SegspecSource


def manifest_hash(
    *,
    timestamp: str,
    speclist_path: Path,
    sources: list[SegspecSource],
    payload_offset: int,
    step_bytes: int,
    payload_bytes: int,
    source_nstep: int,
    window_start: int,
    window_count: int,
) -> str:
    payload = {
        "format_version": 1,
        "timestamp": timestamp,
        "speclist_path": speclist_path.resolve().as_posix(),
        "payload_offset": payload_offset,
        "step_bytes": step_bytes,
        "payload_bytes": payload_bytes,
        "source_nstep": source_nstep,
        "window_start": window_start,
        "window_count": window_count,
        "sources": [asdict(source) for source in sources],
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def can_reuse(
    manifest_path: Path,
    xcspec_path: Path,
    expected_hash: str,
    expected_size: int,
) -> bool:
    if not manifest_path.is_file() or not xcspec_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("manifest_hash") == expected_hash and xcspec_path.stat().st_size == expected_size


def write_manifest(
    path: Path,
    *,
    timestamp: str,
    xcspec_path: Path,
    speclist_path: Path,
    sources: list[SegspecSource],
    nstep: int,
    nspec: int,
    dt: float,
    df: float,
    source_table_offset: int,
    payload_offset: int,
    step_bytes: int,
    payload_bytes: int,
    manifest_hash_value: str,
    source_nstep: int,
    window_start: int,
    window_count: int,
) -> None:
    manifest = {
        "format_version": 1,
        "timestamp": timestamp,
        "xcspec_path": xcspec_path.resolve().as_posix(),
        "speclist_path": speclist_path.resolve().as_posix(),
        "layout": "step_file_freq",
        "dtype": "complex64_interleaved",
        "file_count": len(sources),
        "nstep": nstep,
        "nspec": nspec,
        "nfft": 2 * (nspec - 1),
        "dt": dt,
        "df": df,
        "source_table_offset": source_table_offset,
        "payload_offset": payload_offset,
        "step_bytes": step_bytes,
        "payload_bytes": payload_bytes,
        "manifest_hash": manifest_hash_value,
        "source_nstep": source_nstep,
        "window_start": window_start,
        "window_count": window_count,
        "sources": [asdict(source) for source in sources],
    }
    write_text_atomic(path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
