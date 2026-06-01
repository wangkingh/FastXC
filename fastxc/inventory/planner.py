from __future__ import annotations

import csv
import logging
import math
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

GNSLKey = tuple[str, str, str, str]

_HEADER_BYTES = 632
_FLOAT_BLOCK_BYTES = 70 * 4
_NVHDR_OFFSET = _FLOAT_BLOCK_BYTES + 6 * 4
_STLA_OFFSET = 31 * 4
_STLO_OFFSET = 32 * 4
_STEL_OFFSET = 33 * 4
_SAC_UNDEFINED = -12345.0
_C_FLT_MAX = 3.4028234663852886e38


@dataclass
class GeoNode:
    gnsl_id: int
    group: str
    network: str
    station: str
    location: str
    representative_file_path: str
    components: set[str] = field(default_factory=set)
    lat: float | None = None
    lon: float | None = None
    ele: float | None = None
    sac_geo_parse_ok: bool = True

    @property
    def gnsl(self) -> GNSLKey:
        return (self.group, self.network, self.station, self.location)

    @property
    def id_text(self) -> str:
        return f"{self.gnsl_id:04d}"


@dataclass(frozen=True)
class PathRecord:
    path_id: int
    src_gnsl_id: int
    rec_gnsl_id: int
    src_gnsl: GNSLKey
    rec_gnsl: GNSLKey
    great_circle_deg: float | None
    distance_km: float | None
    azimuth_deg: float | None
    back_azimuth_deg: float | None

    @property
    def path_id_text(self) -> str:
        return f"{self.path_id:08d}"


@dataclass(frozen=True)
class PairFilterConfig:
    allow_autocorr: bool = True
    group_pair_mode: str = "all"
    use_distance_filter: bool = False
    min_distance_km: float = 0.0
    max_distance_km: float = 50000.0
    use_azimuth_filter: bool = False
    azimuth_ranges_deg: tuple[tuple[float, float], ...] = ((0.0, 360.0),)


@dataclass
class PathPlan:
    nodes: list[GeoNode]
    paths: list[PathRecord]
    pair_checks_total: int
    missing_geo_node_count: int
    sac_header_parse_error_count: int
    external_geo_row_count: int
    external_geo_node_fill_count: int
    external_geo_conflict_node_count: int
    config: PairFilterConfig

    @property
    def node_by_gnsl(self) -> dict[GNSLKey, GeoNode]:
        return {node.gnsl: node for node in self.nodes}

    @property
    def retained_gnsls(self) -> set[GNSLKey]:
        return {path.src_gnsl for path in self.paths} | {path.rec_gnsl for path in self.paths}

    @property
    def allowed_path_ids(self) -> set[int]:
        return {path.path_id for path in self.paths}


def build_path_plan(
    files_group1: Dict | None = None,
    files_group2: Optional[Dict] = None,
    *,
    files_groups: Optional[dict[str, Dict]] = None,
    distance_range: str = "-1/50000",
    azimuth_range: str = "-1/360",
    double_array: bool = False,
    allow_autocorr: bool = True,
    group_pair_mode: str | None = None,
    external_geo_tsv_path: str | Path | None = None,
) -> PathPlan:
    """Build unique GNSL nodes and retained canonical GNSL paths."""
    group_map = _normalize_files_groups(files_group1, files_group2, files_groups)
    nodes = collect_geo_nodes_from_groups(files_groups=group_map)
    enrich_nodes_from_sac_headers(nodes)
    external_geo_stats = inject_external_geo(nodes, external_geo_tsv_path)

    config = parse_pair_filter_config(
        distance_range=distance_range,
        azimuth_range=azimuth_range,
        double_array=double_array,
        allow_autocorr=allow_autocorr,
        group_pair_mode=group_pair_mode,
    )
    paths, pair_checks_total = build_valid_paths(nodes, config)

    return PathPlan(
        nodes=nodes,
        paths=paths,
        pair_checks_total=pair_checks_total,
        missing_geo_node_count=sum(1 for node in nodes if node.lat is None or node.lon is None),
        sac_header_parse_error_count=sum(1 for node in nodes if not node.sac_geo_parse_ok),
        **external_geo_stats,
        config=config,
    )


def collect_geo_nodes_from_groups(
    files_group1: Dict | None = None,
    files_group2: Optional[Dict] = None,
    *,
    files_groups: Optional[dict[str, Dict]] = None,
) -> list[GeoNode]:
    node_map: OrderedDict[GNSLKey, GeoNode] = OrderedDict()
    group_map = _normalize_files_groups(files_group1, files_group2, files_groups)
    for group_name, files_group in group_map.items():
        for key, info in files_group.items():
            gnsl = _gnsl_from_group_entry(group_name, key, info)
            components = {str(component) for component in info.get("component", [])}
            representative = _choose_representative_path(info.get("path", []))
            if representative is None:
                continue

            node = node_map.get(gnsl)
            if node is None:
                node = GeoNode(
                    gnsl_id=0,
                    group=gnsl[0],
                    network=gnsl[1],
                    station=gnsl[2],
                    location=gnsl[3],
                    representative_file_path=representative,
                    components=set(),
                )
                node_map[gnsl] = node
            else:
                node.representative_file_path = min(
                    node.representative_file_path,
                    representative,
                )
            node.components.update(components)

    nodes = sorted(node_map.values(), key=lambda item: item.gnsl)
    if len(nodes) > 9999:
        raise ValueError("GNSL node count exceeds 9999; 4-digit IDs are not enough.")

    for index, node in enumerate(nodes, start=1):
        node.gnsl_id = index
    return nodes


def enrich_nodes_from_sac_headers(nodes: Iterable[GeoNode]) -> None:
    for node in nodes:
        geo = read_station_geo_from_sac_header(node.representative_file_path)
        node.lat = geo.lat
        node.lon = geo.lon
        node.ele = geo.ele
        node.sac_geo_parse_ok = geo.parse_ok
        if not geo.parse_ok:
            logger.warning(
                "Could not read SAC geo for %s.%s.%s.%s: %s",
                node.group,
                node.network,
                node.station,
                node.location,
                geo.error_message,
            )


def inject_external_geo(
    nodes: Iterable[GeoNode],
    external_geo_tsv_path: str | Path | None,
) -> dict[str, int]:
    exact_rows, station_rows, row_count = load_external_geo(external_geo_tsv_path)
    if row_count == 0:
        return {
            "external_geo_row_count": 0,
            "external_geo_node_fill_count": 0,
            "external_geo_conflict_node_count": 0,
        }

    fill_count = 0
    conflict_count = 0
    for node in nodes:
        geo = exact_rows.get(node.gnsl) or station_rows.get(node.station)
        if geo is None:
            continue

        updated = False
        conflicted = False
        for field_name in ("lat", "lon", "ele"):
            incoming = geo.get(field_name)
            if incoming is None:
                continue
            current = getattr(node, field_name)
            if current is not None and abs(current - incoming) > 1e-6:
                conflicted = True
            if current is None or abs(current - incoming) > 1e-6:
                setattr(node, field_name, incoming)
                updated = True

        if updated:
            fill_count += 1
        if conflicted:
            conflict_count += 1

    return {
        "external_geo_row_count": row_count,
        "external_geo_node_fill_count": fill_count,
        "external_geo_conflict_node_count": conflict_count,
    }


def load_external_geo(
    external_geo_tsv_path: str | Path | None,
) -> tuple[dict[GNSLKey, dict[str, float | None]], dict[str, dict[str, float | None]], int]:
    if external_geo_tsv_path is None or str(external_geo_tsv_path).strip().upper() == "NONE":
        return {}, {}, 0

    path = Path(external_geo_tsv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"external geo TSV file not found: {path}")

    exact_rows: dict[GNSLKey, dict[str, float | None]] = {}
    station_rows: dict[str, dict[str, float | None]] = {}
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return {}, {}, 0
        for row_number, row in enumerate(reader, start=2):
            station = _required_cell(row, "station", row_number)
            geo = {
                "lat": _required_float_cell(row, "lat", row_number),
                "lon": _required_float_cell(row, "lon", row_number),
                "ele": _optional_float_cell(row.get("ele")),
            }
            row_count += 1

            group = (row.get("group") or "").strip()
            network = (row.get("network") or "").strip()
            location = (row.get("location") or "").strip()
            if group and network and location:
                exact_rows[(group, network, station, location)] = geo
            else:
                if station in station_rows:
                    raise ValueError(
                        f"duplicate station-only external geo row for {station!r}"
                    )
                station_rows[station] = geo

    return exact_rows, station_rows, row_count


def build_valid_paths(
    nodes: list[GeoNode],
    config: PairFilterConfig,
) -> tuple[list[PathRecord], int]:
    paths: list[PathRecord] = []
    pair_checks_total = 0

    for src_index, src in enumerate(nodes):
        rec_start = src_index if config.allow_autocorr else src_index + 1
        for rec in nodes[rec_start:]:
            pair_checks_total += 1
            record = _build_path_record_if_valid(src, rec, config)
            if record is not None:
                paths.append(record)

    paths.sort(key=lambda path: (path.src_gnsl_id, path.rec_gnsl_id))
    return paths, pair_checks_total


def filter_group_by_path_plan(files_group: Dict, group_name: str, plan: PathPlan) -> Dict:
    retained = plan.retained_gnsls
    return {
        key: info
        for key, info in files_group.items()
        if _gnsl_from_group_entry(group_name, key, info) in retained
    }


def station_time_rows_from_group(files_group: Dict) -> tuple[list[str], list]:
    stations = []
    times = []
    for key in files_group:
        if not isinstance(key, tuple) or len(key) < 2:
            continue
        stations.append(key[0])
        times.append(key[1])
    return stations, times


def write_path_plan(plan: PathPlan, output_dir: str | Path) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    node_path = output_path / "nsl_catalog.tsv"
    with node_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "nsl_id",
                "group",
                "network",
                "station",
                "location",
                "lat",
                "lon",
                "ele",
                "components",
                "representative_file_path",
            ]
        )
        for node in plan.nodes:
            writer.writerow(
                [
                    node.id_text,
                    node.group,
                    node.network,
                    node.station,
                    node.location,
                    _float_cell(node.lat),
                    _float_cell(node.lon),
                    _float_cell(node.ele),
                    ",".join(sorted(node.components)),
                    Path(node.representative_file_path).as_posix(),
                ]
            )

    path_path = output_path / "allowed_paths.tsv"
    with path_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "path_id",
                "src_nsl_id",
                "rec_nsl_id",
                "src_nsl",
                "rec_nsl",
                "great_circle_deg",
                "distance_km",
                "azimuth_deg",
                "back_azimuth_deg",
            ]
        )
        for path in plan.paths:
            writer.writerow(
                [
                    path.path_id_text,
                    f"{path.src_gnsl_id:04d}",
                    f"{path.rec_gnsl_id:04d}",
                    _gnsl_text(path.src_gnsl),
                    _gnsl_text(path.rec_gnsl),
                    _float_cell(path.great_circle_deg),
                    _float_cell(path.distance_km),
                    _float_cell(path.azimuth_deg),
                    _float_cell(path.back_azimuth_deg),
                ]
            )

    id_path = output_path / "allowed_path_ids.txt"
    id_path.write_text(
        "".join(f"{path.path_id_text}\n" for path in plan.paths),
        encoding="utf-8",
    )


def write_timestamp_manifests(
    files_group: Dict,
    group_name: str,
    plan: PathPlan,
    output_dir: str | Path,
) -> Path:
    """Write per-timestamp SAC manifests retained by one path plan."""
    manifest_root = Path(output_dir).expanduser().resolve()
    shard_root = manifest_root / "by_timestamp"
    shard_root.mkdir(parents=True, exist_ok=True)

    rows_by_timestamp: OrderedDict[str, list[list[str]]] = OrderedDict()
    for key, info in files_group.items():
        timestamp = _timestamp_from_group_key(key)
        if timestamp is None:
            continue
        gnsl = _gnsl_from_group_entry(group_name, key, info)
        node = plan.node_by_gnsl.get(gnsl)
        if node is None or gnsl not in plan.retained_gnsls:
            continue

        timestamp_text = _format_timestamp(timestamp)
        rows_by_timestamp.setdefault(timestamp_text, [])
        for component, sac_path in _component_path_rows(info):
            rows_by_timestamp[timestamp_text].append(
                [
                    node.id_text,
                    node.group,
                    node.network,
                    node.station,
                    node.location,
                    component,
                    Path(sac_path).expanduser().resolve().as_posix(),
                ]
            )

    index_rows: list[list[str]] = []
    shard_paths: list[Path] = []
    for timestamp_text, rows in rows_by_timestamp.items():
        rows.sort(key=lambda row: (row[0], _component_order(row[5]), row[5], row[6]))
        shard_path = shard_root / f"{timestamp_text}.tsv"
        with shard_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                [
                    "nsl_id",
                    "group",
                    "network",
                    "station",
                    "location",
                    "component",
                    "sac_path",
                ]
            )
            writer.writerows(rows)
        shard_paths.append(shard_path)
        index_rows.append(
            [
                timestamp_text,
                shard_path.relative_to(manifest_root).as_posix(),
                str(len(rows)),
            ]
        )

    index_path = manifest_root / "timestamp_index.tsv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["timestamp", "shard_path", "row_count"])
        writer.writerows(index_rows)

    list_path = manifest_root / "timestamp_manifest_list.txt"
    list_path.write_text(
        "".join(f"{path.as_posix()}\n" for path in shard_paths),
        encoding="utf-8",
    )

    return index_path


def parse_pair_filter_config(
    *,
    distance_range: str,
    azimuth_range: str,
    double_array: bool,
    allow_autocorr: bool,
    group_pair_mode: str | None = None,
) -> PairFilterConfig:
    use_distance, min_distance, max_distance = _parse_distance_range(distance_range)
    use_azimuth, azimuth_ranges = _parse_azimuth_range(azimuth_range)
    return PairFilterConfig(
        allow_autocorr=allow_autocorr,
        group_pair_mode=_normalize_group_pair_mode(group_pair_mode, double_array),
        use_distance_filter=use_distance,
        min_distance_km=min_distance,
        max_distance_km=max_distance,
        use_azimuth_filter=use_azimuth,
        azimuth_ranges_deg=azimuth_ranges,
    )


def _normalize_files_groups(
    files_group1: Dict | None,
    files_group2: Optional[Dict],
    files_groups: Optional[dict[str, Dict]],
) -> OrderedDict[str, Dict]:
    if files_groups is not None:
        return OrderedDict(
            (str(group_name), files_group or {})
            for group_name, files_group in sorted(
                files_groups.items(),
                key=lambda item: _group_sort_key(str(item[0])),
            )
        )

    group_map: OrderedDict[str, Dict] = OrderedDict()
    if files_group1 is not None:
        group_map["1"] = files_group1
    if files_group2:
        group_map["2"] = files_group2
    return group_map


def _group_sort_key(group_name: str) -> tuple[int, str]:
    return (int(group_name), group_name) if group_name.isdigit() else (999, group_name)


def _normalize_group_pair_mode(mode: str | None, double_array: bool) -> str:
    value = (mode or "auto").strip().lower()
    if value == "auto":
        return "cross_group_only" if double_array else "all"
    if value in {"all", "both"}:
        return "all"
    if value in {"intra", "within", "same", "same_group_only"}:
        return "same_group_only"
    if value in {"inter", "between", "cross", "cross_group_only"}:
        return "cross_group_only"
    raise ValueError(f"unsupported group_pair_mode: {mode!r}")


@dataclass(frozen=True)
class SacStationGeoResult:
    lat: float | None
    lon: float | None
    ele: float | None
    parse_ok: bool
    error_message: str | None = None


def read_station_geo_from_sac_header(file_path: str) -> SacStationGeoResult:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        return SacStationGeoResult(None, None, None, False, f"file not found: {path}")

    raw = path.read_bytes()[:_HEADER_BYTES]
    if len(raw) < _HEADER_BYTES:
        return SacStationGeoResult(
            None,
            None,
            None,
            False,
            "file too small to contain a complete SAC header",
        )

    endian = _detect_sac_endian(raw)
    if endian is None:
        return SacStationGeoResult(
            None,
            None,
            None,
            False,
            "unable to determine SAC header endianness",
        )

    try:
        lat = _parse_optional_sac_float(
            struct.unpack_from(f"{endian}f", raw, _STLA_OFFSET)[0],
            kind="lat",
        )
        lon = _parse_optional_sac_float(
            struct.unpack_from(f"{endian}f", raw, _STLO_OFFSET)[0],
            kind="lon",
        )
        ele = _parse_optional_sac_float(
            struct.unpack_from(f"{endian}f", raw, _STEL_OFFSET)[0],
            kind="ele",
        )
    except struct.error as exc:
        return SacStationGeoResult(None, None, None, False, str(exc))

    return SacStationGeoResult(lat, lon, ele, True)


def _build_path_record_if_valid(
    src: GeoNode,
    rec: GeoNode,
    config: PairFilterConfig,
) -> PathRecord | None:
    if src.gnsl == rec.gnsl and not config.allow_autocorr:
        return None
    if not _passes_group_rule(src.group, rec.group, config.group_pair_mode):
        return None

    distance_km = None
    azimuth_deg = None
    back_azimuth_deg = None
    great_circle_deg = None
    if src.lat is None or src.lon is None or rec.lat is None or rec.lon is None:
        if config.use_distance_filter or config.use_azimuth_filter:
            return None
    else:
        great_circle_deg, azimuth_deg, back_azimuth_deg, distance_km = distkm_az_baz_rudoe(
            src.lon,
            src.lat,
            rec.lon,
            rec.lat,
        )

    if config.use_distance_filter:
        if distance_km is None:
            return None
        if distance_km < config.min_distance_km or distance_km > config.max_distance_km:
            return None

    if config.use_azimuth_filter:
        if azimuth_deg is None or back_azimuth_deg is None:
            return None
        if not (
            azimuth_in_ranges(azimuth_deg, config.azimuth_ranges_deg)
            or azimuth_in_ranges(back_azimuth_deg, config.azimuth_ranges_deg)
        ):
            return None

    return PathRecord(
        path_id=src.gnsl_id * 10000 + rec.gnsl_id,
        src_gnsl_id=src.gnsl_id,
        rec_gnsl_id=rec.gnsl_id,
        src_gnsl=src.gnsl,
        rec_gnsl=rec.gnsl,
        great_circle_deg=great_circle_deg,
        distance_km=distance_km,
        azimuth_deg=azimuth_deg,
        back_azimuth_deg=back_azimuth_deg,
    )


def _gnsl_from_group_entry(group_name: str, key, info: Dict) -> GNSLKey:
    if isinstance(key, tuple):
        station = str(key[0])
        network = str(key[2]) if len(key) >= 3 else _first_text(info.get("network"), "VV")
        location = str(key[3]) if len(key) >= 4 else _first_text(info.get("location"), "00")
    else:
        station = str(key)
        network = _first_text(info.get("network"), "VV")
        location = _first_text(info.get("location"), "00")
    return (group_name, network, station, location)


def _choose_representative_path(paths) -> str | None:
    cleaned = [Path(path).expanduser().resolve().as_posix() for path in paths or []]
    return min(cleaned) if cleaned else None


def _timestamp_from_group_key(key):
    if isinstance(key, tuple) and len(key) >= 2:
        return key[1]
    return None


def _format_timestamp(timestamp) -> str:
    if hasattr(timestamp, "strftime"):
        return timestamp.strftime("%Y.%j.%H%M")
    return str(timestamp)


def _component_path_rows(info: Dict) -> list[tuple[str, str]]:
    components = info.get("component", [])
    paths = info.get("path", [])
    return [
        (str(component), str(path))
        for component, path in zip(components, paths)
    ]


def _component_order(component: str) -> int:
    order = {"E": 0, "N": 1, "Z": 2}
    return order.get(component, 99)


def _first_text(values, default: str) -> str:
    if isinstance(values, (list, tuple)) and values:
        return str(values[0])
    if values not in (None, ""):
        return str(values)
    return default


def _required_cell(row: dict[str, str | None], field_name: str, row_number: int) -> str:
    value = (row.get(field_name) or "").strip()
    if not value:
        raise ValueError(f"external geo row {row_number}: missing {field_name!r}")
    return value


def _required_float_cell(
    row: dict[str, str | None],
    field_name: str,
    row_number: int,
) -> float:
    value = _optional_float_cell(row.get(field_name))
    if value is None:
        raise ValueError(f"external geo row {row_number}: invalid {field_name!r}")
    return value


def _optional_float_cell(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _parse_distance_range(value: str) -> tuple[bool, float, float]:
    lo, hi = _parse_range_pair(value, "distance_range")
    if lo < 0.0 and hi >= 50000.0:
        return False, 0.0, 50000.0
    return True, max(0.0, lo), hi


def _parse_azimuth_range(value: str) -> tuple[bool, tuple[tuple[float, float], ...]]:
    lo, hi = _parse_range_pair(value, "azimuth_range")
    if lo < 0.0 and hi >= 360.0:
        return False, ((0.0, 360.0),)
    return True, ((max(0.0, lo), min(360.0, hi)),)


def _parse_range_pair(value: str, name: str) -> tuple[float, float]:
    try:
        lo_text, hi_text = str(value).strip().split("/", 1)
        lo = float(lo_text)
        hi = float(hi_text)
    except Exception as exc:
        raise ValueError(f"{name} must be 'low/high', got {value!r}") from exc
    if hi < lo and lo >= 0.0:
        raise ValueError(f"{name}: lower {lo} > upper {hi}")
    return lo, hi


def _passes_group_rule(src_group: str, rec_group: str, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "same_group_only":
        return src_group == rec_group
    if mode == "cross_group_only":
        return src_group != rec_group
    raise ValueError(f"unsupported group_pair_mode: {mode!r}")


def _detect_sac_endian(raw: bytes) -> str | None:
    best_endian = None
    best_score = -1
    for endian in ("<", ">"):
        try:
            nvhdr = struct.unpack_from(f"{endian}i", raw, _NVHDR_OFFSET)[0]
            stla = struct.unpack_from(f"{endian}f", raw, _STLA_OFFSET)[0]
            stlo = struct.unpack_from(f"{endian}f", raw, _STLO_OFFSET)[0]
        except struct.error:
            continue

        score = 0
        if 0 < nvhdr <= 20:
            score += 2
        if _is_plausible_sac_float(stla, kind="lat"):
            score += 1
        if _is_plausible_sac_float(stlo, kind="lon"):
            score += 1
        if score > best_score:
            best_score = score
            best_endian = endian

    return best_endian if best_score > 0 else None


def _parse_optional_sac_float(value: float, kind: str) -> float | None:
    if _is_undefined(value) or not math.isfinite(value):
        return None
    if kind == "lat" and not (-90.0 <= value <= 90.0):
        return None
    if kind == "lon" and not (-180.0 <= value <= 180.0):
        return None
    return float(value)


def _is_plausible_sac_float(value: float, kind: str) -> bool:
    if _is_undefined(value):
        return True
    if not math.isfinite(value):
        return False
    if kind == "lat":
        return -90.0 <= value <= 90.0
    if kind == "lon":
        return -180.0 <= value <= 180.0
    return True


def _is_undefined(value: float) -> bool:
    return abs(value - _SAC_UNDEFINED) <= 1e-3


def distkm_az_baz_rudoe(
    evlo: float,
    evla: float,
    stlo: float,
    stla: float,
) -> tuple[float, float, float, float]:
    """Return (gcarc_deg, azimuth_deg, back_azimuth_deg, distance_km).

    This mirrors the XC C implementation in cal_dist.c so the Python path plan
    and the CUDA XC stage share one geometry convention.
    """
    if evlo == stlo and evla == stla:
        return 0.0, 0.0, 0.0, 0.0

    deg2rad = math.pi / 180.0
    rad2deg = 180.0 / math.pi
    earth_radius_km = 6378.137
    earth_flattening = 1.0 / 298.257223563
    ec2 = 2.0 * earth_flattening - earth_flattening * earth_flattening
    one_minus_ec2 = 1.0 - ec2

    evla_for_rad = evla if evla != 0.0 else 1.0e-10
    evla_rad = deg2rad * evla_for_rad
    evlo_rad = deg2rad * evlo
    if evla == 90.0 or evla == -90.0:
        evla_geocent = evla * deg2rad
    else:
        evla_geocent = math.atan(one_minus_ec2 * math.tan(evla_rad))

    d = math.sin(evlo_rad)
    e = -math.cos(evlo_rad)
    f = -math.cos(evla_geocent)
    c = math.sin(evla_geocent)
    a = f * e
    b = -f * d
    g = -c * e
    h = c * d

    stla_for_rad = stla if stla != 0.0 else 1.0e-10
    stla_rad = stla_for_rad * deg2rad
    stlo_rad = stlo * deg2rad
    if stla == 90.0 or stla == -90.0:
        stla_geocent = stla * deg2rad
    else:
        stla_geocent = math.atan(one_minus_ec2 * math.tan(stla_rad))

    d1 = math.sin(stlo_rad)
    e1 = -math.cos(stlo_rad)
    f1 = -math.cos(stla_geocent)
    c1 = math.sin(stla_geocent)
    a1 = f1 * e1
    b1 = -f1 * d1
    g1 = -c1 * e1
    h1 = c1 * d1

    sc = a * a1 + b * b1 + c * c1
    sd = 0.5 * math.sqrt(
        (
            (a - a1) ** 2
            + (b - b1) ** 2
            + (c - c1) ** 2
        )
        * (
            (a + a1) ** 2
            + (b + b1) ** 2
            + (c + c1) ** 2
        )
    )
    gcarc = math.atan2(sd, sc) * rad2deg
    if gcarc < 0.0:
        gcarc += 360.0

    ss = (a1 - d) ** 2 + (b1 - e) ** 2 + c1**2 - 2.0
    sc = (a1 - g) ** 2 + (b1 - h) ** 2 + (c1 - f) ** 2 - 2.0
    azimuth = math.atan2(ss, sc) * rad2deg
    if azimuth < 0.0 and abs(azimuth) < 1.0e-8:
        azimuth = 0.0
    if azimuth < 0.0:
        azimuth += 360.0

    ss = (a - d1) ** 2 + (b - e1) ** 2 + c**2 - 2.0
    sc = (a - g1) ** 2 + (b - h1) ** 2 + (c - f1) ** 2 - 2.0
    back_azimuth = math.atan2(ss, sc) * rad2deg
    if back_azimuth < 0.0 and abs(back_azimuth) < 1.0e-8:
        back_azimuth = 0.0
    if back_azimuth < 0.0:
        back_azimuth += 360.0

    if stla_rad > 0.0:
        t1 = stla_rad
        p1 = stlo_rad
        t2 = evla_rad
        p2 = evlo_rad
        costhk, sinthk, tanthk = _rudoe_trig(evla, t2)
        costhi, sinthi, tanthi = _rudoe_trig(stla, t1)
    else:
        t1 = evla_rad
        p1 = evlo_rad
        t2 = stla_rad
        p2 = stlo_rad
        costhk, sinthk, tanthk = _rudoe_trig(stla, t2)
        costhi, sinthi, tanthi = _rudoe_trig(evla, t1)

    ellipsoid_e = ec2 / one_minus_ec2
    e1_rudoe = 1.0 + ellipsoid_e
    al = tanthi / (e1_rudoe * tanthk) + ec2 * math.sqrt(
        (e1_rudoe + tanthi**2) / (e1_rudoe + tanthk**2)
    )
    dl = p1 - p2
    a12 = math.atan2(math.sin(dl), (al - math.cos(dl)) * sinthk)
    cosa12 = math.cos(a12)
    sina12 = math.sin(a12)
    e1_rudoe = ellipsoid_e * ((costhk * cosa12) ** 2 + sinthk**2)
    e2_rudoe = e1_rudoe * e1_rudoe
    e3_rudoe = e1_rudoe * e2_rudoe

    c0 = 1.0 + 0.25 * e1_rudoe - (3.0 / 64.0) * e2_rudoe + (5.0 / 256.0) * e3_rudoe
    c2 = -(1.0 / 8.0) * e1_rudoe + (13.0 / 32.0) * e2_rudoe - (15.0 / 1024.0) * e3_rudoe
    c4 = -(1.0 / 256.0) * e2_rudoe + (3.0 / 1024.0) * e3_rudoe

    v1 = earth_radius_km / math.sqrt(1.0 - ec2 * sinthk**2)
    v2 = earth_radius_km / math.sqrt(1.0 - ec2 * sinthi**2)
    z1 = v1 * one_minus_ec2 * sinthk
    z2 = v2 * one_minus_ec2 * sinthi
    x2 = v2 * costhi * math.cos(dl)
    y2 = v2 * costhi * math.sin(dl)
    e1_plus_1 = e1_rudoe + 1.0
    sqrt_e1_plus_1 = math.sqrt(e1_plus_1)
    u1 = math.atan2(tanthk, sqrt_e1_plus_1 * cosa12)
    u2 = math.atan2(
        v1 * sinthk + e1_plus_1 * (z2 - z1),
        sqrt_e1_plus_1 * (x2 * cosa12 - y2 * sinthk * sina12),
    )
    b0 = v1 * math.sqrt(1.0 + ellipsoid_e * (costhk * cosa12) ** 2) / e1_plus_1
    du = u2 - u1
    if abs(du) > math.pi:
        du = (math.pi * 2.0 - abs(du)) if du > 0.0 else (abs(du) - math.pi * 2.0)
    pdist = b0 * (
        c2 * (math.sin(2.0 * u2) - math.sin(2.0 * u1))
        + c4 * (math.sin(4.0 * u2) - math.sin(4.0 * u1))
    )
    distance_km = abs(b0 * c0 * du + pdist)

    return gcarc, azimuth, back_azimuth, distance_km


def _rudoe_trig(latitude_deg: float, latitude_rad: float) -> tuple[float, float, float]:
    if latitude_deg == 90.0:
        return 0.0, 1.0, _C_FLT_MAX
    if latitude_deg == -90.0:
        return 0.0, -1.0, -_C_FLT_MAX
    cos_lat = math.cos(latitude_rad)
    sin_lat = math.sin(latitude_rad)
    return cos_lat, sin_lat, sin_lat / cos_lat


def azimuth_in_ranges(
    azimuth_deg: float,
    ranges_deg: tuple[tuple[float, float], ...],
) -> bool:
    azimuth_deg = azimuth_deg % 360.0
    for start_deg, end_deg in ranges_deg:
        if start_deg == 0.0 and end_deg == 360.0:
            return True
        start_norm = start_deg % 360.0
        end_norm = end_deg % 360.0
        if start_norm <= end_norm:
            if start_norm <= azimuth_deg <= end_norm:
                return True
        elif azimuth_deg >= start_norm or azimuth_deg <= end_norm:
            return True
    return False


def _float_cell(value: float | None) -> str:
    return "" if value is None else f"{value:.7g}"


def _gnsl_text(gnsl: GNSLKey) -> str:
    return ".".join(gnsl)
