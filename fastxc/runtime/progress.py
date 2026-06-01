from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

PROGRESS_HEADER = ["task", "status", "completed", "total", "unit", "detail"]


def clean_progress_field(value: object) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def write_progress_file(
    progress_file: str | Path | None,
    status: str,
    completed: int,
    total: int,
    unit: str,
    detail: str,
    *,
    task: str = "overall",
) -> None:
    if progress_file is None:
        return

    path = Path(progress_file).expanduser().resolve()
    tmp_path = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = [
        task,
        status,
        str(completed),
        str(total),
        unit,
        detail,
    ]
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(PROGRESS_HEADER) + "\n")
        handle.write("\t".join(clean_progress_field(field) for field in row) + "\n")
    tmp_path.replace(path)


def mark_progress_file(progress_file: str | Path | None, status: str, detail: str) -> None:
    if progress_file is None:
        return

    path = Path(progress_file).expanduser().resolve()
    rows: list[list[str]] = []
    if path.is_file():
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        if lines and lines[0].split("\t")[:6] == PROGRESS_HEADER:
            for line in lines[1:]:
                fields = (line.split("\t", 5) + [""] * 6)[:6]
                fields[1] = status
                fields[5] = clean_progress_field(
                    f"{fields[5]}; {detail}" if fields[5] else detail
                )
                rows.append(fields)

    if not rows:
        rows = [["overall", status, "0", "0", "commands", detail]]

    tmp_path = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(PROGRESS_HEADER) + "\n")
        for row in rows:
            handle.write("\t".join(clean_progress_field(field) for field in row) + "\n")
    tmp_path.replace(path)


def read_progress_file(progress_file: Path) -> list[dict[str, str]] | None:
    try:
        lines = progress_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    if not lines:
        return None
    headers = lines[0].split("\t")
    if headers[:6] != PROGRESS_HEADER:
        return None

    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        fields = (line.split("\t", 5) + [""] * 6)[:6]
        rows.append(dict(zip(headers[:6], fields)))
    return rows


def format_progress_rows(rows: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for row in rows:
        task = row.get("task", "overall")
        status = row.get("status", "")
        completed = row.get("completed", "0")
        total = row.get("total", "0")
        unit = row.get("unit", "")
        detail = row.get("detail", "")
        if total and total != "0":
            text = f"{task} {status} {completed}/{total} {unit}".rstrip()
        else:
            text = f"{task} {status} {completed} {unit}".rstrip()
        if detail:
            text += f" ({detail})"
        parts.append(text)
    return "; ".join(parts)


def format_progress_group(name: str, rows: list[dict[str, str]]) -> str:
    snapshot = format_progress_rows(rows)
    if not name:
        return snapshot
    return f"{name}: {snapshot}"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def estimate_progress_eta(rows: list[dict[str, str]], elapsed: float) -> str | None:
    completed = 0
    total = 0
    for row in rows:
        try:
            row_completed = int(row.get("completed", "0"))
            row_total = int(row.get("total", "0"))
        except ValueError:
            continue
        if row_total > 0:
            completed += max(0, row_completed)
            total += row_total

    if completed <= 0 or total <= 0 or completed >= total:
        return None

    eta = elapsed * (total - completed) / completed
    return format_duration(eta)


def log_progress_snapshot(
    *,
    label: str,
    progress_path: Path,
    last_snapshot: str | None,
    waiting_logged: bool,
    elapsed: float,
) -> tuple[str | None, bool]:
    rows = read_progress_file(progress_path)
    if rows is None:
        if not waiting_logged:
            log.info(
                "%s progress: waiting for %s (elapsed=%s)",
                label,
                progress_path,
                format_duration(elapsed),
            )
            return last_snapshot, True
        return last_snapshot, waiting_logged

    snapshot = format_progress_rows(rows)
    if snapshot and snapshot != last_snapshot:
        eta = estimate_progress_eta(rows, elapsed)
        suffix = f"elapsed={format_duration(elapsed)}"
        if eta is not None:
            suffix += f" eta={eta}"
        log.info("%s progress: %s (%s)", label, snapshot, suffix)
        return snapshot, waiting_logged
    return last_snapshot, waiting_logged


def log_progress_bundle_snapshot(
    *,
    label: str,
    progress_path: Path | None,
    side_progress_paths: dict[str, Path] | None,
    last_snapshot: str | None,
    waiting_logged: bool,
    elapsed: float,
) -> tuple[str | None, bool]:
    parts: list[str] = []
    eta_rows: list[dict[str, str]] | None = None

    if progress_path is not None:
        rows = read_progress_file(progress_path)
        if rows is not None:
            parts.append(format_progress_group("native", rows))
            eta_rows = rows

    for name, side_path in (side_progress_paths or {}).items():
        rows = read_progress_file(side_path)
        if rows is not None:
            parts.append(format_progress_group(name, rows))
            if eta_rows is None:
                eta_rows = rows

    if not parts:
        if not waiting_logged:
            wait_target = progress_path or next(iter((side_progress_paths or {}).values()), None)
            log.info(
                "%s progress: waiting for %s (elapsed=%s)",
                label,
                wait_target or "sidecar progress",
                format_duration(elapsed),
            )
            return last_snapshot, True
        return last_snapshot, waiting_logged

    snapshot = " | ".join(parts)
    if snapshot and snapshot != last_snapshot:
        eta = estimate_progress_eta(eta_rows, elapsed) if eta_rows is not None else None
        suffix = f"elapsed={format_duration(elapsed)}"
        if eta is not None:
            suffix += f" eta={eta}"
        log.info("%s progress: %s (%s)", label, snapshot, suffix)
        return snapshot, waiting_logged
    return last_snapshot, waiting_logged
