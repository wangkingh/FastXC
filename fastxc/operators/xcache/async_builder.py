from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading

from ..cleanup.sweeper import write_spack_cleanup_marker
from fastxc.io.spack import read_spack_timestamp_sources
from fastxc.runtime import write_progress_file
from .builder import build_one_xcache_from_sources
from .writer import XCacheShard, write_xcspec_index

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class XCacheBuildResult:
    output_dir: Path
    index_path: Path
    timestamp_count: int
    shard_count: int


class AsyncXCacheMaterializer:
    """Watch SAC2SPEC timestamp markers and build xcache shards as they finish."""

    def __init__(
        self,
        sac2spec_root: str | Path,
        xcache_dir: str | Path | None = None,
        *,
        windows_per_xcache: int | None = None,
        progress_file: str | Path | None = None,
        poll_interval_sec: float = 5.0,
        cleanup_marker_dir: str | Path | None = None,
    ):
        self.root = Path(sac2spec_root).expanduser().resolve()
        self.spack_root = self.root / "spack_by_timestamp"
        self.output_dir = Path(xcache_dir).expanduser().resolve() if xcache_dir else self.root / "xcache"
        self.windows_per_xcache = windows_per_xcache
        self.progress_file = Path(progress_file).expanduser().resolve() if progress_file else None
        self.poll_interval_sec = max(0.5, float(poll_interval_sec))
        self.cleanup_marker_dir = (
            Path(cleanup_marker_dir).expanduser().resolve()
            if cleanup_marker_dir is not None
            else None
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._done: set[str] = set()
        self._shards: list[XCacheShard] = []
        self._errors: list[BaseException] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_progress_file(self.progress_file, "RUNNING", 0, 0, "timestamps", "async xcache build")
        self._thread = threading.Thread(target=self._run, name="xcache-materializer", daemon=True)
        self._thread.start()

    def finish(self) -> XCacheBuildResult:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._scan_once()
        if self._errors:
            write_progress_file(
                self.progress_file,
                "FAILED",
                len(self._done),
                0,
                "timestamps",
                str(self._errors[0]),
            )
            raise RuntimeError("async XCache materialization failed") from self._errors[0]
        if not self._done:
            write_progress_file(self.progress_file, "DONE", 0, 0, "timestamps", "no timestamp spack observed")
            return XCacheBuildResult(
                output_dir=self.output_dir,
                index_path=self.output_dir / "xcspec_index.tsv",
                timestamp_count=0,
                shard_count=0,
            )

        shards = sorted(self._shards, key=lambda shard: (shard.timestamp, shard.window_start, str(shard.xcspec_path)))
        index_path = write_xcspec_index(self.output_dir, shards)
        (self.output_dir / "_SUCCESS").write_text(
            f"timestamps\t{len(self._done)}\nshards\t{len(shards)}\n",
            encoding="utf-8",
        )
        write_progress_file(
            self.progress_file,
            "DONE",
            len(self._done),
            len(self._done),
            "timestamps",
            "xcache ready",
        )
        return XCacheBuildResult(
            output_dir=self.output_dir,
            index_path=index_path,
            timestamp_count=len(self._done),
            shard_count=len(shards),
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan_once()
            except BaseException as exc:  # keep the worker quiet and fail on finish()
                self._errors.append(exc)
                self._stop.set()
                return
            self._stop.wait(self.poll_interval_sec)

    def _scan_once(self) -> None:
        if not self.spack_root.is_dir():
            return
        for timestamp_dir in sorted(path for path in self.spack_root.iterdir() if path.is_dir()):
            timestamp = _read_timestamp_success(timestamp_dir / "_SUCCESS")
            if timestamp is None:
                continue
            with self._lock:
                if timestamp in self._done:
                    continue
            shards = self._build_timestamp(timestamp_dir, timestamp)
            with self._lock:
                self._done.add(timestamp)
                self._shards.extend(shards)
                completed = len(self._done)
                shard_count = len(self._shards)
            write_progress_file(
                self.progress_file,
                "RUNNING",
                completed,
                0,
                "timestamps",
                f"{timestamp}: {len(shards)} shard(s), total={shard_count}",
            )

    def _build_timestamp(self, timestamp_dir: Path, timestamp: str) -> list[XCacheShard]:
        sources = read_spack_timestamp_sources(timestamp_dir, timestamp)
        shards = build_one_xcache_from_sources(
            timestamp,
            sources,
            timestamp_dir,
            self.output_dir,
            windows_per_xcache=self.windows_per_xcache,
        )
        if self.cleanup_marker_dir is not None:
            write_spack_cleanup_marker(
                self.cleanup_marker_dir,
                timestamp=timestamp,
                timestamp_dir=timestamp_dir,
                shard_count=len(shards),
            )
        return shards


def _read_timestamp_success(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    values: dict[str, str] = {}
    for raw in text.splitlines():
        key, sep, value = raw.partition("\t")
        if sep:
            values[key.strip()] = value.strip()
    return values.get("timestamp") or None
