from __future__ import annotations

import csv
import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .planner import (
    PathPlan,
    build_path_plan,
    filter_group_by_path_plan,
    station_time_rows_from_group,
    write_path_plan,
    write_timestamp_manifests,
)
from .source_scanner import organize_seisarrays


@dataclass(frozen=True)
class InventoryResult:
    root: Path
    sac_index: Path
    timestamp_index: Path
    allowed_paths: Path
    nsl_catalog: Path
    metadata: Path
    sac_row_count: int
    timestamp_count: int
    nsl_count: int
    allowed_path_count: int


def inventory_root(config: Any) -> Path:
    return Path(config.storage.output_dir).expanduser().resolve()


def sac_index_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / "manifest" / "sac_index.tsv"


def timestamp_index_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / "manifest" / "timestamp_index.tsv"


def timestamp_manifest_lists(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    manifest_root = root_path / "manifest"
    paths = sorted(manifest_root.glob("array*/timestamp_manifest_list.txt"))
    paths.extend(sorted(manifest_root.glob("seisarray*/timestamp_manifest_list.txt")))
    return [path for path in paths if path.is_file()]


def build_inventory(config: Any) -> InventoryResult:
    root = inventory_root(config)
    cpu_workers = config.device.cpu_workers
    files_by_group, _arrays_by_group = organize_seisarrays(
        config.seisarrays,
        config.time_filter,
        cpu_workers,
    )
    path_plan = build_path_plan(
        files_groups=files_by_group,
        distance_range=config.xcorr.distance_range,
        azimuth_range=config.xcorr.azimuth_range,
        double_array=config.is_double_array,
        autocorr_mode=config.xcorr.autocorr_mode,
        group_pair_mode=config.xcorr.group_pair_mode,
        external_geo_tsv_path=config.geometry.external_geo_tsv,
    )
    write_path_plan(path_plan, root / "path_plan")
    _raise_if_empty_path_plan(path_plan, config, root)

    filtered_by_group = {
        group_id: filter_group_by_path_plan(files_group, group_id, path_plan)
        for group_id, files_group in files_by_group.items()
    }
    first_group_id = next(iter(filtered_by_group), "1")
    first_group = filtered_by_group.get(first_group_id, {})
    stas1, times1 = station_time_rows_from_group(first_group)

    config.stas1 = stas1
    config.stas2 = []
    config.times1 = times1
    config.times2 = []
    config.files_group1 = first_group
    config.files_group2 = {}
    config.files_groups = filtered_by_group
    config.path_plan = path_plan
    config.allowed_path_ids = path_plan.allowed_path_ids

    manifest_dir = root / "manifest"
    _clean_generated_manifests(manifest_dir)
    for group_id, files_group in filtered_by_group.items():
        if not files_group:
            continue
        write_timestamp_manifests(
            files_group,
            group_id,
            path_plan,
            manifest_dir / f"seisarray{group_id}",
        )

    sac_index, sac_row_count = write_sac_index(root)
    timestamp_index, timestamp_count = write_timestamp_index(root)
    metadata = write_inventory_metadata(config)

    return InventoryResult(
        root=root,
        sac_index=sac_index,
        timestamp_index=timestamp_index,
        allowed_paths=root / "path_plan" / "allowed_paths.tsv",
        nsl_catalog=root / "path_plan" / "nsl_catalog.tsv",
        metadata=metadata,
        sac_row_count=sac_row_count,
        timestamp_count=timestamp_count,
        nsl_count=len(path_plan.nodes),
        allowed_path_count=len(path_plan.paths),
    )


def require_inventory(config: Any) -> None:
    root = inventory_root(config)
    missing: list[Path] = []
    allowed_paths = root / "path_plan" / "allowed_paths.tsv"
    sac_index = sac_index_path(root)
    timestamp_index = timestamp_index_path(root)
    if not allowed_paths.is_file():
        missing.append(allowed_paths)
    if not sac_index.is_file():
        missing.append(sac_index)
    if not timestamp_index.is_file():
        missing.append(timestamp_index)
    if missing:
        preview = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "FastXC inventory is not prepared yet. Run "
            f"`fastxc prepare {config.ini_path}` first.\nMissing:\n  {preview}"
        )

    empty = [
        f"{label}: {path}"
        for label, path in (
            ("allowed paths", allowed_paths),
            ("SAC index", sac_index),
            ("timestamp index", timestamp_index),
        )
        if _count_data_rows(path) == 0
    ]
    if empty:
        preview = "\n  ".join(empty)
        raise ValueError(
            "FastXC inventory has no runnable work. Run `fastxc prepare "
            f"{config.ini_path}` after fixing input filters or station geometry.\n"
            f"Empty table(s):\n  {preview}"
        )


def ensure_sac_index(root: str | Path) -> Path:
    path = sac_index_path(root)
    if path.is_file():
        return path
    path, _row_count = write_sac_index(root)
    return path


def ensure_timestamp_index(root: str | Path) -> Path:
    path = timestamp_index_path(root)
    if path.is_file():
        return path
    path, _timestamp_count = write_timestamp_index(root)
    return path


def write_sac_index(root: str | Path) -> tuple[Path, int]:
    root_path = Path(root).expanduser().resolve()
    index_path = sac_index_path(root_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["timestamp", "nsl_id", "network", "station", "location", "component", "sac_path"])
        for manifest_path in _timestamp_manifest_paths(root_path):
            timestamp = manifest_path.stem
            for nsl_id, network, station, location, component, sac_path in _timestamp_manifest_rows(manifest_path):
                writer.writerow([timestamp, nsl_id, network, station, location, component, sac_path])
                row_count += 1

    return index_path, row_count


def write_timestamp_index(root: str | Path) -> tuple[Path, int]:
    root_path = Path(root).expanduser().resolve()
    index_path = timestamp_index_path(root_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    counts: OrderedDict[str, int] = OrderedDict()
    sac_index = ensure_sac_index(root_path)
    with sac_index.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames and "timestamp" in reader.fieldnames:
            for row in reader:
                timestamp = row.get("timestamp", "").strip()
                if timestamp:
                    counts[timestamp] = counts.get(timestamp, 0) + 1

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["timestamp", "sac_row_count"])
        for timestamp, row_count in counts.items():
            writer.writerow([timestamp, row_count])

    return index_path, len(counts)


def write_inventory_metadata(config: Any) -> Path:
    root = inventory_root(config)
    root.mkdir(parents=True, exist_ok=True)

    snapshot = root / "config.snapshot.ini"
    try:
        shutil.copy2(config.ini_path, snapshot)
    except OSError:
        pass

    sac_index = ensure_sac_index(root)
    timestamp_index = ensure_timestamp_index(root)
    allowed_paths = root / "path_plan" / "allowed_paths.tsv"
    nsl_catalog = root / "path_plan" / "nsl_catalog.tsv"
    metadata = {
        "schema": "fastxc-inventory-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config.ini_path).expanduser().resolve()),
        "config_snapshot": str(snapshot),
        "inventory_root": str(root),
        "sac_index": {
            "path": str(sac_index),
            "row_count": _count_data_rows(sac_index),
        },
        "timestamp_index": {
            "path": str(timestamp_index),
            "timestamp_count": _count_data_rows(timestamp_index),
        },
        "path_plan": {
            "allowed_paths": str(allowed_paths),
            "allowed_path_count": _count_data_rows(allowed_paths),
            "nsl_catalog": str(nsl_catalog),
            "nsl_count": _count_data_rows(nsl_catalog),
        },
    }

    meta_path = root / "inventory.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta_path


def _raise_if_empty_path_plan(plan: PathPlan, config: Any, root: Path) -> None:
    if plan.paths:
        return

    pair_config = plan.config
    path_plan_dir = root / "path_plan"
    external_geo_tsv = getattr(getattr(config, "geometry", None), "external_geo_tsv", "NONE")
    lines = [
        "FastXC path planning produced 0 allowed paths; aborting before SAC2SPEC.",
        f"Path plan files were written for inspection: {path_plan_dir}",
        f"NSL nodes: {len(plan.nodes)}",
        f"Pair checks: {plan.pair_checks_total}",
        f"Nodes missing lat/lon: {plan.missing_geo_node_count}",
        f"SAC header geo parse errors: {plan.sac_header_parse_error_count}",
        f"External geo TSV: {external_geo_tsv}",
        f"External geo rows: {plan.external_geo_row_count}",
        f"External geo nodes updated: {plan.external_geo_node_fill_count}",
        f"External geo conflict nodes: {plan.external_geo_conflict_node_count}",
        f"Distance filter enabled: {pair_config.use_distance_filter}",
        f"Azimuth filter enabled: {pair_config.use_azimuth_filter}",
        f"Group pair mode: {pair_config.group_pair_mode}",
        f"Autocorr only: {pair_config.autocorr_only}",
    ]

    if pair_config.use_distance_filter or pair_config.use_azimuth_filter:
        lines.append("Pairs with missing lat/lon cannot pass active distance or azimuth filters.")
    if _external_geo_enabled(external_geo_tsv) and plan.external_geo_row_count == 0:
        lines.append("External geo TSV was provided, but no data rows were loaded.")
    if (
        _external_geo_enabled(external_geo_tsv)
        and plan.external_geo_row_count > 0
        and plan.external_geo_node_fill_count == 0
    ):
        lines.append(
            "External geo TSV was parsed but did not update any station; check station, network, and location matching."
        )

    raise ValueError("\n".join(lines))


def _external_geo_enabled(path: object) -> bool:
    return str(path or "NONE").strip().upper() not in {"", "NONE"}


def _clean_generated_manifests(manifest_dir: Path) -> None:
    if not manifest_dir.exists():
        return
    for stale in list(manifest_dir.glob("array*")) + list(manifest_dir.glob("seisarray*")):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
    index_path = manifest_dir / "sac_index.tsv"
    if index_path.exists():
        index_path.unlink()
    timestamp_index = manifest_dir / "timestamp_index.tsv"
    if timestamp_index.exists():
        timestamp_index.unlink()


def _timestamp_manifest_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for list_path in timestamp_manifest_lists(root):
        for raw_line in list_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            raw = raw_line.strip()
            if not raw or raw.startswith("#"):
                continue
            manifest_path = Path(raw).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = list_path.parent / manifest_path
            manifest_path = manifest_path.resolve()
            key = manifest_path.as_posix()
            if key in seen:
                continue
            seen.add(key)
            paths.append(manifest_path)
    return paths


def _timestamp_manifest_rows(manifest_path: Path) -> Iterable[tuple[str, str, str, str, str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#") or row[0] in {"gnsl_id", "nsl_id"}:
                continue
            if len(row) < 7:
                raise ValueError(f"Malformed timestamp manifest row in {manifest_path}: {row}")
            sac_path = Path(row[6]).expanduser()
            if not sac_path.is_absolute():
                sac_path = manifest_path.parent / sac_path
            yield row[0], row[2], row[3], row[4], row[5], sac_path.resolve().as_posix()


def _count_data_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index == 0 and line.lower().startswith(("path_id", "gnsl_id", "nsl_id", "timestamp")):
                continue
            if line.strip():
                count += 1
    return count
