from __future__ import annotations

from datetime import datetime


TIMESTAMP_FORMAT = "%Y%m%dT%H:%M"


def normalize_timestamp(text: str) -> str:
    raw = str(text).strip()
    for fmt in (
        "%Y.%j.%H%M",
        "%Y.%j.%H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%dT%H:%M",
        "%Y%m%dT%H:%M:%S",
        "%Y%m%dT%H%M",
        "%Y%m%d%H%M",
    ):
        try:
            return datetime.strptime(raw, fmt).strftime(TIMESTAMP_FORMAT)
        except ValueError:
            pass

    raise ValueError(
        "Unsupported timestamp format for xcache: "
        f"{text!r}; expected YYYY.JJJ.HHMM or YYYYMMDDTHH:MM"
    )
