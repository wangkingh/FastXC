from __future__ import annotations

import platform
import shutil
from pathlib import Path
from typing import Iterable, Sequence


AUTO = "AUTO"
NONE = "NONE"

DEFAULT_EXECUTABLES: dict[str, tuple[str, ...]] = {
    "sac2spec": ("sac2spec",),
    "xc": ("xc_multi_channel", "xc_fast"),
    "pws": ("ncf_pws",),
    "tfpws": ("ncf_tfpws",),
}


def resolve_executable_root(root: str | None, *, ini_dir: str | Path | None = None) -> str:
    text = _normalize_token(root, default=AUTO)
    if text in {AUTO, NONE}:
        return text

    root_path = Path(text).expanduser()
    if ini_dir is not None and not root_path.is_absolute():
        root_path = Path(ini_dir).expanduser() / root_path
    return str(root_path)


def resolve_executable(
    *,
    value: str | None,
    default_names: str | Sequence[str],
    root: str = AUTO,
    ini_dir: str | Path | None = None,
) -> str:
    names = _normalize_names(default_names)
    text = _normalize_token(value, default=AUTO)
    root = _normalize_token(root, default=AUTO)

    if text == NONE:
        return NONE

    if root not in {AUTO, NONE}:
        root_path = Path(root).expanduser()
        candidates = names if text == AUTO else (text,)
        found = _find_under_root(root_path, candidates)
        return str(found if found is not None else root_path / candidates[0])

    if text == AUTO:
        found = find_packaged_executable(names)
        return str(found if found is not None else names[0])

    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)

    if any(sep in text for sep in ("/", "\\")):
        base = Path(ini_dir).expanduser() if ini_dir is not None else Path.cwd()
        return str((base / path).resolve())

    found = find_packaged_executable((text,))
    return str(found if found is not None else text)


def find_packaged_executable(names: Iterable[str]) -> Path | None:
    candidates = list(_candidate_names(names))
    for root in _binary_search_roots():
        found = _find_under_root(root, candidates)
        if found is not None:
            return found

    for name in candidates:
        hit = shutil.which(name)
        if hit:
            return Path(hit).expanduser().resolve()
    return None


def executable_report() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, names in DEFAULT_EXECUTABLES.items():
        if names == (NONE,):
            continue
        found = find_packaged_executable(names)
        rows.append((key, str(found) if found else "NOT FOUND"))
    return rows


def _normalize_token(value: str | None, *, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    upper = text.upper()
    if upper in {AUTO, "DEFAULT"}:
        return AUTO
    if upper in {NONE, "OFF", "DISABLED"}:
        return NONE
    return text


def _normalize_names(names: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, str):
        names = (names,)
    cleaned = tuple(name for name in names if name)
    return cleaned or (NONE,)


def _candidate_names(names: Iterable[str]) -> Iterable[str]:
    for name in names:
        if name == NONE:
            continue
        yield name


def _binary_search_roots() -> list[Path]:
    pkg_root = Path(__file__).resolve().parents[1]
    code_root = pkg_root.parent
    tags = _platform_tags()

    roots: list[Path] = []
    roots.append(code_root / "bin")
    for tag in tags:
        roots.append(pkg_root / "bin" / tag)
    roots.append(pkg_root / "bin")
    return roots


def _platform_tags() -> tuple[str, ...]:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("amd64", "x86_64")
    if system.startswith("linux"):
        return (f"linux-{machine}", "linux-x86_64", "linux")
    return (f"{system}-{machine}", system)


def _find_under_root(root: Path, names: Sequence[str]) -> Path | None:
    for name in _candidate_names(names):
        path = root / name
        if path.is_file():
            return path.resolve()
    return None
