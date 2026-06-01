from __future__ import annotations

import logging
from pathlib import Path
import sys


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_FASTXC_FILE_HANDLER = "_fastxc_file_handler"


def _coerce_level(level: int | str | None) -> int | None:
    if level is None:
        return None
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        value = getattr(logging, level.upper(), None)
        if isinstance(value, int):
            return value
    raise ValueError(f"Invalid logging level: {level!r}")


def configure_logging(
    *,
    level: int | str | None = None,
    log_file_path: str | Path | None = None,
) -> None:
    """Configure root logging consistently for FastXC scripts and pipelines."""

    root = logging.getLogger()
    resolved_level = _coerce_level(level)
    if resolved_level is not None:
        root.setLevel(resolved_level)
    elif not root.handlers or root.level in (logging.NOTSET, logging.WARNING):
        root.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    for handler in root.handlers:
        handler.setFormatter(formatter)

    stream_handlers = [
        handler for handler in root.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]
    if not stream_handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if log_file_path and str(log_file_path).upper() != "NONE":
        log_path = Path(log_file_path).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fastxc_file_handlers = [
            handler for handler in root.handlers
            if isinstance(handler, logging.FileHandler)
            and getattr(handler, _FASTXC_FILE_HANDLER, False)
        ]
        matching = [
            handler for handler in fastxc_file_handlers
            if Path(handler.baseFilename).resolve() == log_path
        ]
        if not matching:
            for handler in fastxc_file_handlers:
                root.removeHandler(handler)
                handler.close()
            handler = logging.FileHandler(log_path, encoding="utf-8")
            setattr(handler, _FASTXC_FILE_HANDLER, True)
            handler.setFormatter(formatter)
            root.addHandler(handler)
