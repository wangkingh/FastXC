from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractNcfResult:
    output_path: Path
    index_path: Path
    record_path: Path
    record_offset: int
    record_bytes: int
    row: dict[str, str]
    reversed_match: bool = False


def run(args: argparse.Namespace) -> ExtractNcfResult:
    return extract_ncf(
        input_path=args.input,
        workspace=args.workspace,
        timestamp=args.timestamp,
        source=args.source,
        receiver=args.receiver,
        component_pair=args.component_pair,
        output=args.output,
        src_network=args.src_network,
        rec_network=args.rec_network,
        src_location=args.src_location,
        rec_location=args.rec_location,
        allow_reverse=args.allow_reverse,
        dry_run=args.dry_run,
    )


def extract_ncf(
    *,
    input_path: str | Path | None = None,
    workspace: str | Path | None = None,
    timestamp: str | None,
    source: str,
    receiver: str,
    component_pair: str,
    output: str | Path,
    src_network: str | None = None,
    rec_network: str | None = None,
    src_location: str | None = None,
    rec_location: str | None = None,
    allow_reverse: bool = False,
    dry_run: bool = False,
) -> ExtractNcfResult:
    if bool(input_path) == bool(workspace):
        raise ValueError("provide exactly one of --input or --workspace")

    src_component, rec_component = parse_component_pair(component_pair)
    indexes = discover_ncf_indexes(input_path=input_path, workspace=workspace, timestamp=timestamp)
    matches: list[tuple[Path, dict[str, str], bool]] = []
    for index_path in indexes:
        for row in _iter_index_rows(index_path):
            if timestamp and not _timestamp_matches(row.get("timestamp", ""), timestamp):
                continue
            if _row_matches(
                row,
                source=source,
                receiver=receiver,
                src_component=src_component,
                rec_component=rec_component,
                src_network=src_network,
                rec_network=rec_network,
                src_location=src_location,
                rec_location=rec_location,
            ):
                matches.append((index_path, row, False))
            elif allow_reverse and _row_matches(
                row,
                source=receiver,
                receiver=source,
                src_component=rec_component,
                rec_component=src_component,
                src_network=rec_network,
                rec_network=src_network,
                src_location=rec_location,
                rec_location=src_location,
            ):
                matches.append((index_path, row, True))

    if not matches:
        raise FileNotFoundError(
            "No NCF record matched "
            f"source={source!r} receiver={receiver!r} component_pair={component_pair!r}"
        )
    if len(matches) > 1:
        summary = ", ".join(_match_summary(index, row) for index, row, _ in matches[:5])
        raise ValueError(f"Multiple NCF records matched ({len(matches)}): {summary}")

    index_path, row, reversed_match = matches[0]
    record_path = resolve_record_path(row, index_path=index_path, workspace=Path(workspace).expanduser() if workspace else None)
    record_offset = int(row[_offset_field(row)])
    record_bytes = int(row["bytes"])
    output_path = Path(output).expanduser().resolve()
    result = ExtractNcfResult(
        output_path=output_path,
        index_path=index_path,
        record_path=record_path,
        record_offset=record_offset,
        record_bytes=record_bytes,
        row=row,
        reversed_match=reversed_match,
    )
    if dry_run:
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("rb") as src:
        src.seek(record_offset)
        data = src.read(record_bytes)
    if len(data) != record_bytes:
        raise ValueError(
            f"Truncated NCF record: {record_path} offset={record_offset} "
            f"bytes={record_bytes} got={len(data)}"
        )
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return result


def parse_component_pair(component_pair: str) -> tuple[str, str]:
    parts = [item.strip() for item in component_pair.replace("/", "-").split("-") if item.strip()]
    if len(parts) != 2:
        raise ValueError("component pair must look like SRC-REC, e.g. BHE-BHZ")
    return parts[0], parts[1]


def discover_ncf_indexes(
    *,
    input_path: str | Path | None = None,
    workspace: str | Path | None = None,
    timestamp: str | None,
) -> list[Path]:
    if workspace is not None:
        root = Path(workspace).expanduser().resolve()
        candidates: list[Path] = []
        if timestamp:
            safe = _safe_timestamp(timestamp)
            candidates.extend(
                [
                    root / "sourcepack" / timestamp / "sourcepack_index.tsv",
                    root / "sourcepack" / safe / "sourcepack_index.tsv",
                ]
            )
        candidates.append(root / "sourcepack")
        for candidate in candidates:
            paths = _discover_from_input(candidate, timestamp=timestamp)
            if paths:
                return paths
        return _discover_from_input(root / "ncf" / "xcpack", timestamp=timestamp)

    if input_path is None:
        raise ValueError("provide --input or --workspace")
    paths = _discover_from_input(Path(input_path).expanduser(), timestamp=timestamp)
    if not paths:
        raise FileNotFoundError(f"No NCF index found for {input_path}")
    return paths


def resolve_record_path(row: dict[str, str], *, index_path: Path, workspace: Path | None = None) -> Path:
    raw = row.get("record_path") or row.get("pack_path")
    if not raw:
        raise ValueError(f"NCF index row has no record_path/pack_path: {row}")

    direct = Path(raw).expanduser()
    if direct.is_file():
        return direct.resolve()
    if not direct.is_absolute():
        relative = (index_path.parent / direct).resolve()
        if relative.is_file():
            return relative

    basename = direct.name or Path(raw.replace("\\", "/")).name
    names = _path_name_candidates(basename)
    search_dirs = _record_search_dirs(index_path, workspace)
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()

    raise FileNotFoundError(
        f"NCF record file not found for {index_path}: {raw}. "
        f"Tried names {names} under {', '.join(str(p) for p in search_dirs)}"
    )


def _discover_from_input(path: Path, *, timestamp: str | None) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        return []

    direct = path / "sourcepack_index.tsv"
    if direct.is_file():
        return [direct.resolve()]

    if path.name == "xcpack":
        paths = _timestamp_tsv_paths(path, timestamp)
        return [p.resolve() for p in paths]

    sourcepack_paths = sorted(path.glob("*/sourcepack_index.tsv"))
    if sourcepack_paths:
        if timestamp:
            safe = _safe_timestamp(timestamp)
            filtered = [
                p for p in sourcepack_paths if p.parent.name in {timestamp, safe} or _timestamp_in_file(p, timestamp)
            ]
            if filtered:
                return [p.resolve() for p in filtered]
        return [p.resolve() for p in sourcepack_paths]

    xcpack_dir = path / "xcpack"
    if xcpack_dir.is_dir():
        return [p.resolve() for p in _timestamp_tsv_paths(xcpack_dir, timestamp)]
    return []


def _iter_index_rows(index_path: Path):
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return
        _validate_fields(index_path, reader.fieldnames)
        for row in reader:
            yield row


def _validate_fields(index_path: Path, fields: list[str]) -> None:
    names = set(fields)
    required = {
        "timestamp",
        "src_station",
        "rec_station",
        "src_component",
        "rec_component",
        "bytes",
    }
    missing = required.difference(names)
    if missing:
        raise ValueError(f"NCF index missing fields {sorted(missing)}: {index_path}")
    if not ({"record_path", "record_offset"}.issubset(names) or {"pack_path", "offset"}.issubset(names)):
        raise ValueError(f"NCF index missing record path/offset fields: {index_path}")


def _row_matches(
    row: dict[str, str],
    *,
    source: str,
    receiver: str,
    src_component: str,
    rec_component: str,
    src_network: str | None,
    rec_network: str | None,
    src_location: str | None,
    rec_location: str | None,
) -> bool:
    return (
        _same(row.get("src_station"), source)
        and _same(row.get("rec_station"), receiver)
        and _same(row.get("src_component"), src_component)
        and _same(row.get("rec_component"), rec_component)
        and _optional_same(row.get("src_network"), src_network)
        and _optional_same(row.get("rec_network"), rec_network)
        and _optional_same(row.get("src_location"), src_location)
        and _optional_same(row.get("rec_location"), rec_location)
    )


def _timestamp_matches(value: str, wanted: str) -> bool:
    return value == wanted or _safe_timestamp(value) == _safe_timestamp(wanted)


def _safe_timestamp(text: str) -> str:
    return text.replace(":", "_")


def _timestamp_tsv_paths(xcpack_dir: Path, timestamp: str | None) -> list[Path]:
    if timestamp:
        names = {timestamp, _safe_timestamp(timestamp)}
        paths = [p for p in xcpack_dir.glob("*.tsv") if any(p.name.startswith(name) for name in names)]
        if paths:
            return sorted(paths)
    return sorted(p for p in xcpack_dir.glob("*.tsv") if not p.name.startswith("_"))


def _timestamp_in_file(index_path: Path, timestamp: str) -> bool:
    try:
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or "timestamp" not in reader.fieldnames:
                return False
            for row in reader:
                if _timestamp_matches(row.get("timestamp", ""), timestamp):
                    return True
    except OSError:
        return False
    return False


def _record_search_dirs(index_path: Path, workspace: Path | None) -> list[Path]:
    dirs: list[Path] = []
    if workspace is not None:
        workspace = workspace.expanduser().resolve()
        dirs.extend([workspace / "ncf" / "xcpack", workspace / "xcpack"])
    dirs.extend(
        [
            index_path.parent,
            index_path.parent / "xcpack",
            index_path.parent.parent / "xcpack",
            index_path.parent.parent / "ncf" / "xcpack",
            index_path.parent.parent.parent / "ncf" / "xcpack",
        ]
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        resolved = directory.expanduser().resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _path_name_candidates(name: str) -> list[str]:
    candidates = [name]
    if ":" in name:
        candidates.append(name.replace(":", "_"))
    if "_" in name:
        candidates.append(name.replace("_", ":"))
    return list(dict.fromkeys(candidates))


def _offset_field(row: dict[str, str]) -> str:
    return "record_offset" if "record_offset" in row else "offset"


def _same(left: str | None, right: str) -> bool:
    return (left or "").strip() == right.strip()


def _optional_same(left: str | None, right: str | None) -> bool:
    return right is None or _same(left, right)


def _match_summary(index_path: Path, row: dict[str, str]) -> str:
    return (
        f"{index_path.name}:"
        f"{row.get('src_station')}-{row.get('rec_station')}."
        f"{row.get('src_component')}-{row.get('rec_component')}@"
        f"{row.get(_offset_field(row))}"
    )

