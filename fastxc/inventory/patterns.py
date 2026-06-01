from __future__ import annotations

import os
import re
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

DEFAULT_NETWORK = "VV"
DEFAULT_LOCATION = "00"

TOKEN_PATTERN = re.compile(r"(\{\*\}|\{\?\}|\{\w+\})")
PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)}")
GROUP_NAMES = {
    "YYYY": "year",
    "YY": "year",
    "MM": "month",
    "DD": "day",
    "JJJ": "jday",
    "HH": "hour",
    "MI": "minute",
}


class CompiledPattern(str):
    """A root-relative path pattern compiled into a regex."""

    def __new__(
        cls,
        display_regex: str,
        *,
        regex: re.Pattern[str],
        root_dir: str,
        declared_fields: frozenset[str],
    ):
        obj = str.__new__(cls, display_regex)
        obj.regex = regex
        obj.root_dir = root_dir
        obj.declared_fields = declared_fields
        return obj


class FieldRegistry:
    def __init__(self, base_fields: Dict[str, str] | None = None):
        self._fields = OrderedDict(base_fields or DEFAULT_BASE_FIELDS)
        self._group_names = {
            field_name: GROUP_NAMES.get(field_name, field_name)
            for field_name in self._fields
        }

    def add_field(self, field_name: str, regex_str: str, overwrite: bool = False):
        if field_name in self._fields and not overwrite:
            return

        try:
            re.compile(f"(?P<{field_name}>{regex_str})")
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc

        self._fields[field_name] = regex_str
        self._group_names[field_name] = field_name

    def validate_pattern_fields(self, pattern: str):
        pattern_fields = set(PLACEHOLDER_PATTERN.findall(pattern))
        valid_fields = set(self._fields) | {"home"}
        invalid_fields = pattern_fields - valid_fields
        if invalid_fields:
            raise ValueError(f"pattern contains invalid fields: {invalid_fields}")

    def build_regex_pattern(self, pattern: str) -> str:
        """Build a regex for a pattern relative to the source root."""
        normalized = _strip_home_prefix(pattern)
        normalized = normalized.replace("\\", "/")

        parts: list[str] = ["^"]
        cursor = 0
        seen_fields: set[str] = set()

        for match in TOKEN_PATTERN.finditer(normalized):
            literal = normalized[cursor : match.start()]
            if literal:
                parts.append(re.escape(literal))

            token = match.group(0)
            if token == "{*}":
                parts.append(r".*")
            elif token == "{?}":
                parts.append(r"[^. _/]*")
            else:
                name = token[1:-1]
                if name == "home":
                    raise ValueError("{home} is only supported as the leading root token")
                if name in seen_fields:
                    parts.append(f"(?P={self._group_names.get(name, name)})")
                else:
                    group_name = self._group_names.get(name, name)
                    parts.append(f"(?P<{group_name}>{self._fields[name]})")
                    seen_fields.add(name)
            cursor = match.end()

        tail = normalized[cursor:]
        if tail:
            parts.append(re.escape(tail))
        parts.append("$")

        return "".join(parts)


DEFAULT_BASE_FIELDS = OrderedDict(
    {
        "YYYY": r"\d{4}",
        "YY": r"\d{2}",
        "MM": r"\d{2}",
        "DD": r"\d{2}",
        "JJJ": r"\d{3}",
        "HH": r"\d{2}",
        "MI": r"\d{2}",
        "network": r"[A-Za-z0-9_-]+",
        "event": r"[A-Za-z0-9_-]+",
        "station": r"[A-Za-z0-9_-]+",
        "location": r"[A-Za-z0-9_-]+",
        "component": r"[A-Za-z0-9_-]+",
        "channel": r"[A-Za-z0-9_-]+",
        "sampleF": r"[A-Za-z0-9_-]+",
        "quality": r"[A-Za-z0-9_-]+",
        "locid": r"[A-Za-z0-9_-]+",
        "suffix": r"[A-Za-z0-9_-]+",
        "arrayID": r"[A-Za-z0-9_-]+",
        "label0": r"[A-Za-z0-9_-]+",
        "label1": r"[A-Za-z0-9_-]+",
        "label2": r"[A-Za-z0-9_-]+",
        "label3": r"[A-Za-z0-9_-]+",
        "label4": r"[A-Za-z0-9_-]+",
        "label5": r"[A-Za-z0-9_-]+",
        "label6": r"[A-Za-z0-9_-]+",
        "label7": r"[A-Za-z0-9_-]+",
        "label8": r"[A-Za-z0-9_-]+",
        "label9": r"[A-Za-z0-9_-]+",
    }
)


def check_pattern(array_dir: str, pattern: str, registry: FieldRegistry) -> CompiledPattern:
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a string")

    registry.validate_pattern_fields(pattern)
    placeholder_list = PLACEHOLDER_PATTERN.findall(pattern)
    placeholders = set(placeholder_list)

    if "station" not in placeholders:
        raise ValueError("pattern must contain {station}")
    if "home" not in placeholders:
        raise ValueError("pattern must contain {home}")
    if "component" not in placeholders and "channel" not in placeholders:
        raise ValueError("pattern must contain {component} or {channel}")

    if not any(field in placeholders for field in ("YYYY", "YY", "MM", "DD", "JJJ")):
        raise ValueError("pattern must contain one set of date fields")

    if not os.path.isdir(array_dir):
        # Keep old behavior: validation logs elsewhere, matching simply yields no rows.
        pass

    root_dir = _normalize_path(array_dir)
    relative_regex = registry.build_regex_pattern(pattern)
    display_root = str(Path(array_dir).expanduser().resolve()).replace(".", r"\.")
    display_regex = display_root + r"\/" + relative_regex.lstrip("^")
    return CompiledPattern(
        display_regex,
        regex=re.compile(relative_regex),
        root_dir=root_dir,
        declared_fields=frozenset(placeholders),
    )


def match_path(compiled: CompiledPattern, file_path: str | Path) -> dict[str, str] | None:
    relative_path = _relative_to_root(_normalize_path(file_path), compiled.root_dir)
    if relative_path is None:
        return None

    match = compiled.regex.fullmatch(relative_path)
    if match is None:
        return None

    return _canonicalize_fields(match.groupdict(), compiled.declared_fields)


def parse_timestamp(fields: dict[str, str]) -> datetime | None:
    year_text = fields.get("YYYY") or fields.get("YY")
    if not year_text:
        return None

    year = int(year_text)
    if len(year_text) == 2:
        year += 2000

    hour = int(fields.get("HH") or 0)
    minute = int(fields.get("MI") or 0)
    month_text = fields.get("MM")
    day_text = fields.get("DD")
    jday_text = fields.get("JJJ")

    try:
        if month_text and day_text:
            return datetime(year, int(month_text), int(day_text), hour, minute)
        if jday_text:
            return datetime(year, 1, 1) + timedelta(
                days=int(jday_text) - 1,
                hours=hour,
                minutes=minute,
            )
    except ValueError:
        return None

    return None


def _strip_home_prefix(pattern: str) -> str:
    normalized = pattern.strip().replace("\\", "/")
    if normalized == "{home}":
        return "."
    if normalized.startswith("{home}/"):
        return normalized[len("{home}/") :]
    return normalized


def _canonicalize_fields(
    fields: dict[str, str], declared_fields: set[str] | frozenset[str]
) -> dict[str, str]:
    cleaned = {key: value for key, value in fields.items() if value not in (None, "")}

    aliases = {
        "year": "YYYY",
        "month": "MM",
        "day": "DD",
        "jday": "JJJ",
        "hour": "HH",
        "minute": "MI",
    }
    for old_name, new_name in aliases.items():
        if old_name in cleaned and new_name not in cleaned:
            cleaned[new_name] = cleaned[old_name]

    if "component" not in cleaned and "channel" in cleaned:
        cleaned["component"] = cleaned["channel"]
    if "channel" not in cleaned and "component" in cleaned:
        cleaned["channel"] = cleaned["component"]

    if "network" not in declared_fields and "network" not in cleaned:
        cleaned["network"] = DEFAULT_NETWORK
    if "location" not in declared_fields and "location" not in cleaned:
        cleaned["location"] = DEFAULT_LOCATION

    return cleaned


def _normalize_path(path: str | Path) -> str:
    return Path(path).expanduser().resolve().as_posix()


def _relative_to_root(file_path: str, root_dir: str) -> str | None:
    file_norm = file_path.replace("\\", "/")
    root_norm = root_dir.replace("\\", "/").rstrip("/")

    try:
        file_path_obj = Path(file_norm)
        root_path_obj = Path(root_norm)
        return file_path_obj.relative_to(root_path_obj).as_posix()
    except ValueError:
        pass

    if os.name == "nt":
        file_cmp = file_norm.lower()
        root_cmp = root_norm.lower()
    else:
        file_cmp = file_norm
        root_cmp = root_norm

    prefix = root_cmp + "/"
    if file_cmp == root_cmp:
        return "."
    if not file_cmp.startswith(prefix):
        return None
    return file_norm[len(root_norm) + 1 :]
