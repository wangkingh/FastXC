from __future__ import annotations

import logging
import os
from pathlib import Path
import shlex
import subprocess
from typing import Mapping, Sequence

from fastxc.inventory import ensure_sac_index
from fastxc.runtime import run_command, write_command_review

log = logging.getLogger(__name__)


def _normalize_skip(skip: str) -> str:
    parts = [part for part in skip.strip().replace("/", ",").split(",") if part]
    if not parts:
        return "-1"
    if parts[-1] != "-1":
        parts.append("-1")
    return ",".join(parts)


def _quote(value: str | Path) -> str:
    text = str(value)
    if os.name == "nt":
        return subprocess.list2cmdline([text])
    return shlex.quote(text)


def _as_posix_path(path: Path) -> str:
    return path.expanduser().resolve().as_posix()


def _format_gpu_list(gpu_ids: Sequence[int]) -> str:
    cleaned: list[int] = []
    for gpu_id in gpu_ids:
        gpu_id = int(gpu_id)
        if gpu_id < 0:
            raise ValueError(f"GPU id must be non-negative: {gpu_id}")
        cleaned.append(gpu_id)
    if not cleaned:
        cleaned = [0]
    return ",".join(str(gpu_id) for gpu_id in cleaned)


def _format_gpu_memory_mib(limits: Sequence[float] | None) -> str | None:
    if not limits:
        return None
    cleaned: list[str] = []
    for limit in limits:
        limit = float(limit)
        if limit < 0:
            raise ValueError(f"GPU memory limit must be non-negative: {limit}")
        cleaned.append(f"{limit:g}")
    return ",".join(cleaned)


def gen_sac2spec_cmd(
    component_num: int,
    sac2spec_exe: str | Path,
    output_dir: str | Path,
    sac_len: int,
    win_len: int,
    shift_len: int,
    normalize: str,
    cpu_workers: int,
    whiten: str,
    skip_step: str,
    *,
    xcorr_lag_sec: float | None = None,
    gpu_ids: Sequence[int] | None = None,
    gpu_memory_mib: Sequence[float] | None = None,
    output_phase_only: bool = False,
    debug_mode: bool = False,
) -> list[str]:
    """Build the SAC2SPEC command."""
    out_dir = Path(output_dir).expanduser().resolve()

    sac_index = ensure_sac_index(out_dir)
    progress_dir = out_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / "sac2spec_progress.tsv"
    if progress_file.exists():
        progress_file.unlink()

    whiten_code = {"OFF": 0, "BEFORE": 1, "AFTER": 2, "BOTH": 3}[whiten]
    normalize_code = {"OFF": 0, "RUN-ABS-MF": 1, "ONE-BIT": 2, "RUN-ABS": 3}[normalize]
    skip_fixed = _normalize_skip(skip_step)
    lag_sec = float(win_len if xcorr_lag_sec is None else xcorr_lag_sec)
    selected_gpu_ids = list(gpu_ids or [0])
    total_io_threads = max(1, int(cpu_workers))
    gpu_list = _format_gpu_list(selected_gpu_ids)
    gpu_memory_list = _format_gpu_memory_mib(gpu_memory_mib)
    window_spec = f"{float(sac_len):g}/{float(win_len):g}/{float(shift_len):g}/{lag_sec:g}"
    whiten_spec = f"{whiten_code}/{'1' if output_phase_only else '0'}"

    cmd_parts = [
        str(sac2spec_exe),
        "-I",
        _as_posix_path(sac_index),
        "-O",
        _as_posix_path(out_dir),
        "-C",
        str(component_num),
        "-G",
        gpu_list,
        "-T",
        str(total_io_threads),
        "-L",
        window_spec,
        "-W",
        whiten_spec,
        "-N",
        str(normalize_code),
        "-Q",
        skip_fixed,
        "-B",
        _as_posix_path(out_dir / "filter.txt"),
    ]
    if gpu_memory_list is not None:
        cmd_parts.extend(["-M", gpu_memory_list])
    if debug_mode:
        cmd_parts.append("--debug")

    command = " ".join(_quote(part) for part in cmd_parts)
    review_path = write_command_review(out_dir, "sac2spec", [command])
    log.info("Built SAC2SPEC command.")
    log.info("SAC2SPEC command review file: %s", review_path)
    return [command]


def sac2spec_deployer(
    commands: Sequence[str],
    output_dir: str | Path,
    log_file_path: str | Path,
    dry_run: bool,
    side_progress_files: Mapping[str, str | Path] | None = None,
) -> None:
    """Run SAC2SPEC command(s)."""

    output_dir = Path(output_dir).expanduser().resolve()
    progress_file = output_dir / "progress" / "sac2spec_progress.tsv"
    for index, command in enumerate(commands, start=1):
        run_command(
            command,
            log_file_path=log_file_path,
            label="SAC2SPEC" if len(commands) == 1 else f"SAC2SPEC#{index}",
            dry_run=dry_run,
            progress_file=progress_file if index == 1 else None,
            side_progress_files=side_progress_files if index == 1 else None,
        )
    log.info("Done SAC2SPEC (%d command(s)).\n", len(commands))
