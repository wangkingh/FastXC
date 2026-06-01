from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import threading

from fastxc.runtime import write_progress_file


@dataclass(frozen=True)
class SpackSweepResult:
    marker_dir: Path
    deleted_count: int
    bytes_deleted: int


class AsyncSpackSweeper:
    """Delete timestamp-local SAC2SPEC spack after xcache has consumed it."""

    def __init__(
        self,
        workspace: str | Path,
        marker_dir: str | Path,
        *,
        progress_file: str | Path | None = None,
        poll_interval_sec: float = 5.0,
        dry_run: bool = False,
    ):
        self.workspace = Path(workspace).expanduser().resolve()
        self.spack_root = self.workspace / "spack_by_timestamp"
        self.marker_dir = Path(marker_dir).expanduser().resolve()
        self.progress_file = Path(progress_file).expanduser().resolve() if progress_file else None
        self.poll_interval_sec = max(0.5, float(poll_interval_sec))
        self.dry_run = dry_run
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._done: set[Path] = set()
        self._errors: list[BaseException] = []
        self._deleted_count = 0
        self._bytes_deleted = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        write_progress_file(self.progress_file, "RUNNING", 0, 0, "timestamps", "spack sweeper")
        self._thread = threading.Thread(target=self._run, name="spack-sweeper", daemon=True)
        self._thread.start()

    def finish(self) -> SpackSweepResult:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._scan_once()
        if self._errors:
            write_progress_file(
                self.progress_file,
                "FAILED",
                self._deleted_count,
                0,
                "timestamps",
                str(self._errors[0]),
            )
            raise RuntimeError("async spack cleanup failed") from self._errors[0]

        write_progress_file(
            self.progress_file,
            "DONE",
            self._deleted_count,
            self._deleted_count,
            "timestamps",
            f"deleted {self._bytes_deleted} bytes",
        )
        return SpackSweepResult(
            marker_dir=self.marker_dir,
            deleted_count=self._deleted_count,
            bytes_deleted=self._bytes_deleted,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan_once()
            except BaseException as exc:
                self._errors.append(exc)
                self._stop.set()
                return
            self._stop.wait(self.poll_interval_sec)

    def _scan_once(self) -> None:
        if not self.marker_dir.is_dir():
            return
        for marker in sorted(self.marker_dir.glob("*.ready")):
            marker = marker.resolve()
            with self._lock:
                if marker in self._done:
                    continue
            timestamp_dir = _read_marker_timestamp_dir(marker)
            bytes_deleted = self._delete_timestamp_dir(timestamp_dir)
            with self._lock:
                self._done.add(marker)
                self._deleted_count += 1
                self._bytes_deleted += bytes_deleted
                done = self._deleted_count
                total_bytes = self._bytes_deleted
            write_progress_file(
                self.progress_file,
                "RUNNING",
                done,
                0,
                "timestamps",
                f"{timestamp_dir.name}: deleted {total_bytes} bytes total",
            )

    def _delete_timestamp_dir(self, timestamp_dir: Path) -> int:
        timestamp_dir = timestamp_dir.expanduser().resolve()
        _ensure_safe_timestamp_dir(self.spack_root, timestamp_dir)
        if not timestamp_dir.exists():
            return 0
        if not (timestamp_dir / "_SUCCESS").is_file():
            raise FileNotFoundError(f"Refusing to delete unfinished spack timestamp: {timestamp_dir}")

        byte_count = _directory_size(timestamp_dir)
        if not self.dry_run:
            shutil.rmtree(timestamp_dir)
        return byte_count


def write_spack_cleanup_marker(
    marker_dir: str | Path,
    *,
    timestamp: str,
    timestamp_dir: str | Path,
    shard_count: int,
) -> Path:
    marker_dir = Path(marker_dir).expanduser().resolve()
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{_safe_leaf(timestamp)}.ready"
    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text(
        f"timestamp\t{timestamp}\n"
        f"timestamp_dir\t{Path(timestamp_dir).expanduser().resolve().as_posix()}\n"
        f"shards\t{shard_count}\n",
        encoding="utf-8",
    )
    tmp.replace(marker)
    return marker


def _read_marker_timestamp_dir(marker: Path) -> Path:
    values: dict[str, str] = {}
    text = marker.read_text(encoding="utf-8")
    for raw in text.splitlines():
        key, sep, value = raw.partition("\t")
        if sep:
            values[key.strip()] = value.strip()
    if not values.get("timestamp_dir"):
        raise ValueError(f"Spack cleanup marker missing timestamp_dir: {marker}")
    return Path(values["timestamp_dir"])


def _ensure_safe_timestamp_dir(spack_root: Path, timestamp_dir: Path) -> None:
    spack_root = spack_root.expanduser().resolve()
    timestamp_dir = timestamp_dir.expanduser().resolve()
    if timestamp_dir == spack_root or spack_root not in timestamp_dir.parents:
        raise ValueError(f"Refusing to delete path outside spack root: {timestamp_dir}")


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _safe_leaf(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._:-" else "-" for ch in text).strip("-")
    return out or "timestamp"
