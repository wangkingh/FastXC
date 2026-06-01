from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping, Sequence

from .progress import log_progress_bundle_snapshot, mark_progress_file

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    elapsed_sec: float


class _ForwardedSignal(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"received signal {signum}")


def _native_command_env() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("FASTXC_LOG_LEVEL"):
        return env
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        env["FASTXC_LOG_LEVEL"] = "DEBUG"
    return env


def write_command_review(
    output_dir: str | Path,
    name: str,
    commands: Sequence[str],
) -> Path:
    """Write a readable copy of commands under workspace/commands."""

    path = Path(output_dir).expanduser().resolve() / "commands" / f"{name}.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(commands)
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _terminate_process_tree(
    process: subprocess.Popen,
    *,
    label: str,
    log_handle,
    reason: str,
    grace_sec: float = 8.0,
) -> None:
    if process.poll() is not None:
        return

    log_handle.write(f"[{label}] {reason}; sending SIGTERM to child process group\n")
    log_handle.flush()

    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        log_handle.write(f"[{label}] child still running after {grace_sec:.1f}s; sending SIGKILL\n")
        log_handle.flush()

    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def run_command(
    command: str,
    *,
    log_file_path: str | Path,
    label: str,
    dry_run: bool = False,
    progress_file: str | Path | None = None,
    side_progress_files: Mapping[str, str | Path] | None = None,
    progress_interval_sec: float | None = None,
) -> CommandResult:
    """Run one shell command and append stdout/stderr to the FastXC log."""

    if dry_run:
        log.info("[DryRun] %s: %s", label, command)
        print(f"[DryRun] Would execute: {command}")
        return CommandResult(command, 0, 0.0)

    log_file = Path(log_file_path).expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    progress_path = Path(progress_file).expanduser().resolve() if progress_file is not None else None
    side_progress_paths = {
        name: Path(path).expanduser().resolve()
        for name, path in (side_progress_files or {}).items()
        if path is not None
    }
    poll_interval = 10.0 if progress_interval_sec is None else max(0.2, float(progress_interval_sec))

    with log_file.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n[{label}] START {command}\n")
        if progress_file is not None:
            handle.write(f"[{label}] PROGRESS {Path(progress_file).expanduser().resolve()}\n")
        for side_label, side_path in side_progress_paths.items():
            handle.write(f"[{label}] SIDE_PROGRESS {side_label} {side_path}\n")
        handle.flush()

        old_handlers: dict[int, signal.Handlers] = {}

        def _signal_handler(signum: int, _frame) -> None:
            raise _ForwardedSignal(signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, _signal_handler)
            except ValueError:
                pass

        process: subprocess.Popen | None = None
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=_native_command_env(),
                start_new_session=(os.name != "nt"),
            )
            last_progress_snapshot: str | None = None
            waiting_logged = False
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                if progress_path is not None or side_progress_paths:
                    last_progress_snapshot, waiting_logged = log_progress_bundle_snapshot(
                        label=label,
                        progress_path=progress_path,
                        side_progress_paths=side_progress_paths,
                        last_snapshot=last_progress_snapshot,
                        waiting_logged=waiting_logged,
                        elapsed=time.monotonic() - start,
                    )
                time.sleep(poll_interval)

            if progress_path is not None or side_progress_paths:
                log_progress_bundle_snapshot(
                    label=label,
                    progress_path=progress_path,
                    side_progress_paths=side_progress_paths,
                    last_snapshot=last_progress_snapshot,
                    waiting_logged=waiting_logged,
                    elapsed=time.monotonic() - start,
                )
        except _ForwardedSignal as exc:
            if process is not None:
                _terminate_process_tree(
                    process,
                    label=label,
                    log_handle=handle,
                    reason=f"FastXC received signal {exc.signum}",
                )
            elapsed = time.monotonic() - start
            returncode = process.returncode if process is not None else None
            handle.write(f"[{label}] END signal={exc.signum} returncode={returncode} elapsed_sec={elapsed:.3f}\n")
            mark_progress_file(progress_file, "INTERRUPTED", f"FastXC received signal {exc.signum}")
            if exc.signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + exc.signum)
        finally:
            for signum, old_handler in old_handlers.items():
                signal.signal(signum, old_handler)

        elapsed = time.monotonic() - start
        handle.write(f"[{label}] END returncode={returncode} elapsed_sec={elapsed:.3f}\n")

    if returncode != 0:
        mark_progress_file(progress_file, "FAILED", f"{label} exited with code {returncode}")
        log.error("%s command failed with code %d: %s", label, returncode, command)
        raise subprocess.CalledProcessError(returncode, command)

    log.info("%s command finished in %.2f s.", label, elapsed)
    return CommandResult(command, returncode, elapsed)
