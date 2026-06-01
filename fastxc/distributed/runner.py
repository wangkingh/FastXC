from __future__ import annotations

import os
import subprocess
import sys
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path

from .planner import RUN_PLAN_HEADER, collect_plan_sourcepacks, load_distributed_plan
from .resources import LOCAL_HOSTS


def run_distributed_plan(
    run_plan: str | Path,
    *,
    repo_dir: str | Path | None = None,
    python_exe: str | None = None,
    collect: bool = True,
    main_workspace: str | Path | None = None,
    jobs: int = 1,
) -> Path | None:
    plan_path = Path(run_plan).expanduser().resolve()
    rows = load_distributed_plan(plan_path)
    if not rows:
        raise ValueError(f"Run plan has no tasks: {plan_path}")

    repo = Path(repo_dir).expanduser().resolve() if repo_dir else Path.cwd().resolve()
    py = python_exe or sys.executable
    jobs = max(1, int(jobs))
    lock = Lock()

    def run_one(row: dict[str, str]) -> None:
        command = _task_command(row, repo=repo, python_exe=py)
        with lock:
            row["status"] = "RUNNING"
            _write_status(plan_path, rows)
        try:
            subprocess.run(command, shell=True, check=True)
        except BaseException:
            with lock:
                row["status"] = "FAILED"
                _write_status(plan_path, rows)
            raise
        with lock:
            row["status"] = "DONE"
            _write_status(plan_path, rows)

    if jobs == 1:
        for row in rows:
            run_one(row)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_one, row) for row in rows]
            for future in as_completed(futures):
                future.result()

    if not collect:
        return None
    return collect_plan_sourcepacks(plan_path, main_workspace=main_workspace)


def _task_command(row: dict[str, str], *, repo: Path, python_exe: str) -> str:
    config = Path(row["config"]).expanduser().resolve()
    host = row.get("host", "localhost").strip().lower()
    if host in LOCAL_HOSTS:
        return _shell_join(
            [
                f"cd {_quote(repo.as_posix())}",
                f"{_quote(python_exe)} -m fastxc.cli run {_quote(config.as_posix())}",
            ]
        )

    remote = _shell_join(
        [
            f"cd {_quote(repo.as_posix())}",
            f"{_quote(python_exe)} -m fastxc.cli run {_quote(config.as_posix())}",
        ]
    )
    return f"ssh {_quote(row['host'])} {_quote(remote)}"


def _shell_join(parts: list[str]) -> str:
    return " && ".join(parts)


def _quote(value: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _write_status(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_PLAN_HEADER, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
