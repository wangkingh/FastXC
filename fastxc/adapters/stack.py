from __future__ import annotations

import logging
import os
from pathlib import Path
import shlex
import subprocess
from typing import Sequence

from fastxc.io.sourcepack import discover_workspace_sourcepack_inputs
from fastxc.runtime import run_command, write_command_review

log = logging.getLogger(__name__)


def _quote(value: str | Path) -> str:
    text = str(value)
    if os.name == "nt":
        return subprocess.list2cmdline([text])
    return shlex.quote(text)


def _format_gpu_list(gpu_ids: Sequence[int] | None) -> str:
    if not gpu_ids:
        return "0"
    return ",".join(str(int(gpu_id)) for gpu_id in gpu_ids)


def _format_gpu_memory_mib(limits: Sequence[float] | None) -> str | None:
    if not limits:
        return None
    return ",".join(f"{float(limit):g}" for limit in limits)


def _write_list(path: Path, rows: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row.as_posix()}\n" for row in rows), encoding="utf-8")


def gen_pws_sourcepack_cmd(
    stack_exe: str | Path,
    output_dir: str | Path,
    pre_stack_size: int = 10,
    *,
    gpu_ids: Sequence[int] | None = None,
    gpu_memory_mib: Sequence[float] | None = None,
    cpu_workers: int | None = None,
) -> list[str]:
    """Generate one sourcepack-mode PWS command for native ncf_pws."""

    del cpu_workers
    stack_exe = str(stack_exe)
    out_root = Path(output_dir).expanduser().resolve()
    indexes = discover_workspace_sourcepack_inputs(out_root)
    stack_dir = out_root / "stack"
    manifest_dir = stack_dir / "pws_sourcepack" / "manifests"
    output_sourcepack = stack_dir / "pws_sourcepack" / "STACK"
    progress_dir = out_root / "progress"

    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_sourcepack.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    input_list = manifest_dir / "pws_sourcepack_inputs.txt"
    progress_file = progress_dir / "pws_progress.tsv"
    if progress_file.exists():
        progress_file.unlink()
    _write_list(input_list, indexes)

    cmd_parts: list[str | Path] = [
        stack_exe,
        "--sourcepack-list",
        input_list,
        "--output-sourcepack",
        output_sourcepack,
        "--progress",
        progress_file,
        "-G",
        _format_gpu_list(gpu_ids),
    ]

    gpu_memory_list = _format_gpu_memory_mib(gpu_memory_mib)
    if gpu_memory_list is not None:
        cmd_parts.extend(["-M", gpu_memory_list])
    cmd_parts.extend(["-B", str(max(1, int(pre_stack_size)))])

    command = " ".join(_quote(part) for part in cmd_parts)
    review_path = write_command_review(out_root, "pws_sourcepack", [command])
    log.info("PWS sourcepack command built (%d input index files).", len(indexes))
    log.info("PWS sourcepack command review file: %s", review_path)
    return [command]


def gen_tfpws_sourcepack_cmd(
    stack_exe: str | Path,
    output_dir: str | Path,
    pre_stack_size: int = 10,
    *,
    gpu_ids: Sequence[int] | None = None,
    gpu_memory_mib: Sequence[float] | None = None,
    tfpws_band: str | None = None,
    tfpws_taper_hz: str | float | None = None,
) -> list[str]:
    """Generate one sourcepack-mode TFPWS command for native ncf_tfpws."""

    stack_exe = str(stack_exe)
    out_root = Path(output_dir).expanduser().resolve()
    indexes = discover_workspace_sourcepack_inputs(out_root)
    stack_dir = out_root / "stack"
    manifest_dir = stack_dir / "tfpws_sourcepack" / "manifests"
    output_sourcepack = stack_dir / "tfpws_sourcepack" / "STACK"
    progress_dir = out_root / "progress"

    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_sourcepack.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    input_list = manifest_dir / "tfpws_sourcepack_inputs.txt"
    progress_file = progress_dir / "tfpws_progress.tsv"
    if progress_file.exists():
        progress_file.unlink()
    _write_list(input_list, indexes)

    cmd_parts: list[str | Path] = [
        stack_exe,
        "--sourcepack-list",
        input_list,
        "--output-sourcepack",
        output_sourcepack,
        "--progress",
        progress_file,
        "-G",
        _format_gpu_list(gpu_ids),
        "-S",
        "001",
    ]

    gpu_memory_list = _format_gpu_memory_mib(gpu_memory_mib)
    if gpu_memory_list is not None:
        cmd_parts.extend(["-M", gpu_memory_list])
    band = (tfpws_band or "FULL").strip()
    if band.upper() not in {"", "FULL", "ALL", "NONE", "OFF"}:
        cmd_parts.extend(["-F", band])
        taper = "AUTO" if tfpws_taper_hz is None else str(tfpws_taper_hz).strip()
        if taper.upper() not in {"", "AUTO", "NONE", "OFF"}:
            cmd_parts.extend(["-T", taper])
    cmd_parts.extend(["-B", str(max(1, int(pre_stack_size)))])

    command = " ".join(_quote(part) for part in cmd_parts)
    review_path = write_command_review(out_root, "tfpws_sourcepack", [command])
    log.info("TFPWS sourcepack command built (%d input index files).", len(indexes))
    log.info("TFPWS sourcepack command review file: %s", review_path)
    return [command]


def weighted_stack_deployer(
    commands: Sequence[str],
    output_dir: str | Path,
    method: str,
    log_file_path: str | Path,
    dry_run: bool,
) -> None:
    """Run native weighted-stack command(s)."""

    progress_file = Path(output_dir).expanduser().resolve() / "progress" / f"{method}_progress.tsv"
    label = method.upper()
    for index, command in enumerate(commands, start=1):
        run_command(
            command,
            log_file_path=log_file_path,
            label=label if len(commands) == 1 else f"{label}#{index}",
            dry_run=dry_run,
            progress_file=progress_file if index == 1 else None,
        )
    log.info("Done %s stack (%d command(s)).\n", label, len(commands))
