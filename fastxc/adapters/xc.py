from __future__ import annotations

import logging
from pathlib import Path
import shlex
from typing import Mapping, Sequence

from fastxc.runtime import run_command, write_command_review

logger = logging.getLogger(__name__)


def _as_posix_path(path: str | Path) -> str:
    return Path(path).expanduser().resolve().as_posix()


def _format_gpu_list(gpu_ids: Sequence[int]) -> str:
    if not gpu_ids:
        return "0"
    return ",".join(str(int(gpu_id)) for gpu_id in gpu_ids)


def _format_gpu_memory_mib(limits: Sequence[float] | None) -> str | None:
    if not limits:
        return None
    return ",".join(f"{float(limit):g}" for limit in limits)


def _build_xc_args(
    *,
    xc_input_path: str | Path,
    allowed_paths_file: str | Path,
    ncf_dir: str | Path,
    cclength: float | int,
    gpu_ids: Sequence[int],
    gpu_memory_mib: Sequence[float] | None = None,
    progress_file: str | Path | None = None,
    debug_mode: bool = False,
) -> list[str]:
    xc_input = Path(xc_input_path).expanduser().resolve()
    allowed_paths = Path(allowed_paths_file).expanduser().resolve()
    if not xc_input.exists():
        raise FileNotFoundError(f"XC input not found: {xc_input}")
    if not allowed_paths.is_file():
        raise FileNotFoundError(f"XC allowed path table not found: {allowed_paths}")

    args = [
        "-I",
        xc_input.as_posix(),
        "-P",
        allowed_paths.as_posix(),
        "-O",
        _as_posix_path(ncf_dir),
        "-C",
        str(cclength),
        "-G",
        _format_gpu_list(gpu_ids),
    ]
    gpu_memory_list = _format_gpu_memory_mib(gpu_memory_mib)
    if gpu_memory_list is not None:
        args.extend(["-M", gpu_memory_list])
    if progress_file is not None:
        args.extend(["--progress", _as_posix_path(progress_file)])
    if debug_mode:
        args.append("--debug")
    return args


def gen_xc_cmd(
    *,
    xc_input_path: str | Path,
    allowed_paths_file: str | Path,
    output_dir: str | Path,
    xc_exe: str | Path,
    ncf_dir: str | Path,
    cclength: float | int,
    gpu_ids: Sequence[int],
    debug_mode: bool = False,
    gpu_memory_mib: Sequence[float] | None = None,
) -> list[str]:
    """Generate one XC command from the SAC2SPEC stepack workspace."""
    progress_dir = Path(output_dir).expanduser().resolve() / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / "xc_progress.tsv"
    if progress_file.exists():
        progress_file.unlink()

    args = _build_xc_args(
        xc_input_path=xc_input_path,
        allowed_paths_file=allowed_paths_file,
        ncf_dir=ncf_dir,
        cclength=cclength,
        gpu_ids=gpu_ids,
        gpu_memory_mib=gpu_memory_mib,
        progress_file=progress_file,
        debug_mode=debug_mode,
    )
    cmd = shlex.join([str(xc_exe), *args])
    review_path = write_command_review(output_dir, "xc", [cmd])
    logger.info("Built one XC command.")
    logger.info("XC command review file: %s", review_path)
    return [cmd]


def xc_deployer(
    commands: Sequence[str],
    output_dir: str | Path,
    log_file_path: str | Path,
    dry_run: bool,
    side_progress_files: Mapping[str, str | Path] | None = None,
) -> None:
    """Run generated XC command(s)."""

    progress_file = Path(output_dir).expanduser().resolve() / "progress" / "xc_progress.tsv"
    for index, command in enumerate(commands, start=1):
        run_command(
            command,
            log_file_path=log_file_path,
            label="XC" if len(commands) == 1 else f"XC#{index}",
            dry_run=dry_run,
            progress_file=progress_file if index == 1 else None,
            side_progress_files=side_progress_files if index == 1 else None,
        )
    logger.info("Done Cross Correlation (%d command(s)).\n", len(commands))
