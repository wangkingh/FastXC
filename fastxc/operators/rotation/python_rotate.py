from __future__ import annotations

import concurrent.futures
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import BinaryIO, Iterable, Sequence

import numpy as np
from tqdm import tqdm

from fastxc.inventory.planner import distkm_az_baz_rudoe
from fastxc.io import SacHeader, encode_sac_record, read_sac, read_sac_record, read_sac_record_from, write_sac
from fastxc.io.sourcepack import SOURCEPACK_INDEX_HEADER

log = logging.getLogger(__name__)

ENZ_ORDER = ("E-E", "E-N", "E-Z", "N-E", "N-N", "N-Z", "Z-E", "Z-N", "Z-Z")
RTZ_ORDER = ("R-R", "R-T", "R-Z", "T-R", "T-T", "T-Z", "Z-R", "Z-T", "Z-Z")
_SAC_COMPONENT_NAMES = ("RR", "RT", "RZ", "TR", "TT", "TZ", "ZR", "ZT", "ZZ")
_SAC_NULL = -12345.0


@dataclass(frozen=True)
class RotateResult:
    input_list: Path
    output_list: Path
    output_paths: tuple[Path, ...]
    sample_count: int


def build_rotate_matrix(azimuth_deg: float, back_azimuth_deg: float) -> np.ndarray:
    """Build the ENZ-tensor to RTZ-tensor rotation matrix.

    Source-side horizontal rotation:
        Rs =  Es * sin(az) + Ns * cos(az)
        Ts =  Es * cos(az) - Ns * sin(az)

    Receiver-side horizontal rotation follows the historical FastXC convention
    where receiver radial points from receiver back to source:
        Rr = -Er * sin(baz) - Nr * cos(baz)
        Tr = -Er * cos(baz) + Nr * sin(baz)

    The 9x9 tensor matrix below is just the outer-product expansion of those
    source and receiver basis transforms, with Z passing through unchanged.
    """

    az = np.float32(np.deg2rad(azimuth_deg))
    baz = np.float32(np.deg2rad(back_azimuth_deg))
    sin_az = np.sin(az, dtype=np.float32)
    cos_az = np.cos(az, dtype=np.float32)
    sin_baz = np.sin(baz, dtype=np.float32)
    cos_baz = np.cos(baz, dtype=np.float32)

    matrix = np.zeros((9, 9), dtype=np.float32)

    matrix[0, 0] = -sin_az * sin_baz
    matrix[0, 1] = -sin_az * cos_baz
    matrix[0, 3] = -cos_az * sin_baz
    matrix[0, 4] = -cos_az * cos_baz

    matrix[1, 0] = -sin_az * cos_baz
    matrix[1, 1] = sin_az * sin_baz
    matrix[1, 3] = -cos_az * cos_baz
    matrix[1, 4] = cos_az * sin_baz

    matrix[2, 2] = sin_az
    matrix[2, 5] = cos_az

    matrix[3, 0] = -cos_az * sin_baz
    matrix[3, 1] = -cos_az * cos_baz
    matrix[3, 3] = sin_az * sin_baz
    matrix[3, 4] = sin_az * cos_baz

    matrix[4, 0] = -cos_az * cos_baz
    matrix[4, 1] = cos_az * sin_baz
    matrix[4, 3] = sin_az * cos_baz
    matrix[4, 4] = -sin_az * sin_baz

    matrix[5, 2] = cos_az
    matrix[5, 5] = -sin_az

    matrix[6, 6] = -sin_baz
    matrix[6, 7] = -cos_baz

    matrix[7, 6] = -cos_baz
    matrix[7, 7] = sin_baz

    matrix[8, 8] = 1.0
    return matrix


def rotate_enz_to_rtz(
    enz_data: np.ndarray,
    azimuth_deg: float,
    back_azimuth_deg: float,
) -> np.ndarray:
    enz = np.asarray(enz_data, dtype=np.float32)
    if enz.ndim != 2 or enz.shape[0] != 9:
        raise ValueError(f"ENZ data must have shape (9, npts), got {enz.shape}")
    return build_rotate_matrix(azimuth_deg, back_azimuth_deg) @ enz


def _is_defined(value: float) -> bool:
    return np.isfinite(value) and abs(value - _SAC_NULL) > 1.0e-3


def _geometry_from_header(header: SacHeader) -> tuple[float, float, float, float]:
    evlo = header.get_float("evlo")
    evla = header.get_float("evla")
    stlo = header.get_float("stlo")
    stla = header.get_float("stla")
    if all(_is_defined(v) for v in (evlo, evla, stlo, stla)):
        return distkm_az_baz_rudoe(evlo, evla, stlo, stla)

    gcarc = header.get_float("gcarc")
    az = header.get_float("az")
    baz = header.get_float("baz")
    dist = header.get_float("dist")
    if all(_is_defined(v) for v in (az, baz)):
        return gcarc, az, baz, dist

    raise ValueError("Cannot determine azimuth/back-azimuth from SAC header")


def _read_list(path: str | Path, expected: int = 9) -> list[Path]:
    path = Path(path).expanduser().resolve()
    rows = [Path(line.strip()).expanduser().resolve() for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} paths, got {len(rows)}")
    return rows


def rotate_sac_files(
    input_paths: Sequence[str | Path],
    output_paths: Sequence[str | Path],
    *,
    dry_run: bool = False,
) -> RotateResult:
    if len(input_paths) != 9 or len(output_paths) != 9:
        raise ValueError("rotate_sac_files expects exactly 9 input and 9 output paths")

    inputs = tuple(Path(p).expanduser().resolve() for p in input_paths)
    outputs = tuple(Path(p).expanduser().resolve() for p in output_paths)
    if dry_run:
        return RotateResult(Path("<direct>"), Path("<direct>"), outputs, 0)

    headers: list[SacHeader] = []
    traces: list[np.ndarray] = []
    for path in inputs:
        header, data = read_sac(path)
        headers.append(header)
        traces.append(data)

    template = headers[8]
    npts = template.npts
    delta = template.get_float("delta")
    for path, header in zip(inputs, headers):
        if header.npts != npts:
            raise ValueError(f"{path}: npts mismatch ({header.npts} vs {npts})")
        if not np.isclose(header.get_float("delta"), delta, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"{path}: delta mismatch ({header.get_float('delta')} vs {delta})")

    gcarc, az, baz, dist = _geometry_from_header(template)
    enz = np.stack(traces, axis=0).astype(np.float32, copy=False)
    rtz = rotate_enz_to_rtz(enz, az, baz).astype(np.float32, copy=False)

    base_header = template.copy()
    base_header.set_float("gcarc", float(gcarc))
    base_header.set_float("az", float(az))
    base_header.set_float("baz", float(baz))
    base_header.set_float("dist", float(dist))
    base_header.set_float("cmpaz", float(baz + 180.0 if baz - 180.0 < 0.0 else baz - 180.0))
    base_header.set_float("cmpinc", 90.0)

    for i, (component, out_path) in enumerate(zip(_SAC_COMPONENT_NAMES, outputs)):
        out_header = base_header.copy()
        out_header.set_text("kcmpnm", component)
        write_sac(out_path, out_header, rtz[i])

    return RotateResult(Path("<direct>"), Path("<direct>"), outputs, npts)


def rotate_from_lists(
    input_list: str | Path,
    output_list: str | Path,
    *,
    dry_run: bool = False,
) -> RotateResult:
    input_list = Path(input_list).expanduser().resolve()
    output_list = Path(output_list).expanduser().resolve()
    outputs = tuple(_read_list(output_list))
    if dry_run:
        return RotateResult(input_list, output_list, outputs, 0)

    inputs = _read_list(input_list)
    result = rotate_sac_files(inputs, outputs, dry_run=False)
    return RotateResult(input_list, output_list, outputs, result.sample_count)


def discover_rotate_list_pairs(output_dir: str | Path) -> list[tuple[Path, Path]]:
    rotate_root = Path(output_dir).expanduser().resolve() / "rotate_list"
    if not rotate_root.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for in_list in sorted(rotate_root.glob("*/*/enz_list.txt")):
        out_list = in_list.with_name("rtz_list.txt")
        if out_list.is_file():
            pairs.append((in_list, out_list))
    return pairs


def _rotate_one(task: tuple[Path, Path, bool]) -> RotateResult:
    in_list, out_list, dry_run = task
    return rotate_from_lists(in_list, out_list, dry_run=dry_run)


def rotate_many_from_lists(
    list_pairs: Iterable[tuple[str | Path, str | Path]],
    *,
    max_workers: int = 1,
    dry_run: bool = False,
) -> list[RotateResult]:
    tasks = [
        (Path(in_list).expanduser().resolve(), Path(out_list).expanduser().resolve(), dry_run)
        for in_list, out_list in list_pairs
    ]
    if not tasks:
        return []

    max_workers = max(1, int(max_workers))
    if max_workers == 1 or len(tasks) == 1:
        return [
            _rotate_one(task)
            for task in tqdm(
                tasks,
                desc="Rotate",
                unit="pair",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            )
        ]

    results: list[RotateResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_rotate_one, task) for task in tasks]
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Rotate",
            unit="pair",
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
        ):
            results.append(fut.result())

    return sorted(results, key=lambda item: str(item.input_list))


def rotate_sourcepack_stack(
    output_dir: str | Path,
    method: str,
    *,
    dry_run: bool = False,
) -> list[RotateResult]:
    output_dir = Path(output_dir).expanduser().resolve()
    method = method.strip().lower()
    input_dir_name = f"{method}_sourcepack"
    output_dir_name = f"rtz_{method}_sourcepack"
    legacy_dir_name = f"rtz_{method}"
    pack_name = f"rtz_{method}.pack"
    storage_kind = f"rtz_{method}_pack"

    index_path = output_dir / "stack" / input_dir_name / "STACK" / "sourcepack_index.tsv"
    if not index_path.is_file():
        raise FileNotFoundError(f"{method} sourcepack index not found: {index_path}")

    groups = _read_rotation_groups(index_path)
    out_dir = output_dir / "stack" / output_dir_name / "STACK"
    pack_path = out_dir / pack_name
    out_index = out_dir / "sourcepack_index.tsv"
    if dry_run:
        return [
            RotateResult(index_path, out_index, tuple(_legacy_rtz_path(output_dir, legacy_dir_name, rows[0], tag) for tag in RTZ_ORDER), 0)
            for _, rows in sorted(groups.items(), key=_rotation_group_sort_key)
        ]

    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_pack = tmp_dir / pack_path.name
    tmp_index = tmp_dir / out_index.name

    results: list[RotateResult] = []
    skipped = 0
    handles: dict[Path, BinaryIO] = {}
    try:
        with tmp_pack.open("wb") as pack_handle, tmp_index.open("w", encoding="utf-8", newline="") as index_handle:
            writer = csv.DictWriter(index_handle, fieldnames=SOURCEPACK_INDEX_HEADER, delimiter="\t")
            writer.writeheader()
            for pair_key, rows in tqdm(
                sorted(groups.items(), key=_rotation_group_sort_key),
                desc="Rotate",
                unit="pair",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            ):
                by_component = {f"{row['src_component']}-{row['rec_component']}": row for row in rows}
                if any(tag not in by_component for tag in ENZ_ORDER):
                    skipped += 1
                    continue

                headers: list[SacHeader] = []
                traces: list[np.ndarray] = []
                for tag in ENZ_ORDER:
                    row = by_component[tag]
                    header, data = _read_sourcepack_row(row, handles)
                    headers.append(header)
                    traces.append(data)

                template = headers[8]
                npts = template.npts
                delta = template.get_float("delta")
                for row, header in zip((by_component[tag] for tag in ENZ_ORDER), headers):
                    if header.npts != npts:
                        raise ValueError(f"{row['record_path']}: npts mismatch ({header.npts} vs {npts})")
                    if not np.isclose(header.get_float("delta"), delta, rtol=0.0, atol=1.0e-7):
                        raise ValueError(f"{row['record_path']}: delta mismatch ({header.get_float('delta')} vs {delta})")

                gcarc, az, baz, dist = _geometry_from_header(template)
                rtz = rotate_enz_to_rtz(np.stack(traces, axis=0), az, baz).astype(np.float32, copy=False)

                base_header = template.copy()
                base_header.set_float("gcarc", float(gcarc))
                base_header.set_float("az", float(az))
                base_header.set_float("baz", float(baz))
                base_header.set_float("dist", float(dist))
                base_header.set_float("cmpaz", float(baz + 180.0 if baz - 180.0 < 0.0 else baz - 180.0))
                base_header.set_float("cmpinc", 90.0)

                output_paths: list[Path] = []
                for i, tag in enumerate(RTZ_ORDER):
                    src_component, rec_component = tag.split("-")
                    out_header = base_header.copy()
                    out_header.set_text("kcmpnm", _SAC_COMPONENT_NAMES[i])
                    record_offset = pack_handle.tell()
                    data = encode_sac_record(out_header, rtz[i])
                    pack_handle.write(data)

                    row = _rotated_index_row(
                        rows[0],
                        src_component=src_component,
                        rec_component=rec_component,
                        pack_path=pack_path,
                        record_offset=record_offset,
                        record_bytes=len(data),
                        final_pair_path=_legacy_rtz_path(output_dir, legacy_dir_name, rows[0], tag),
                        header=out_header,
                        gcarc=gcarc,
                        az=az,
                        baz=baz,
                        dist=dist,
                        storage_kind=storage_kind,
                    )
                    writer.writerow(row)
                    output_paths.append(Path(row["final_pair_path"]))

                results.append(RotateResult(index_path, out_index, tuple(output_paths), npts))
    finally:
        _close_binary_handles(handles)

    (tmp_dir / "_SUCCESS").write_text(
        f"pairs\t{len(results)}\nskipped\t{skipped}\npack\t{pack_path}\nindex\t{out_index}\n",
        encoding="utf-8",
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.replace(out_dir)
    if skipped:
        log.warning("Skipped %d pair(s) with incomplete ENZ components.", skipped)
    return results


def rotate_linearstack_sourcepack(
    output_dir: str | Path,
    *,
    dry_run: bool = False,
) -> list[RotateResult]:
    return rotate_sourcepack_stack(output_dir, "linearstack", dry_run=dry_run)


def _read_sourcepack_row(
    row: dict[str, str],
    handles: dict[Path, BinaryIO],
) -> tuple[SacHeader, np.ndarray]:
    record_path = Path(row["record_path"]).expanduser()
    handle = handles.get(record_path)
    if handle is None:
        handle = record_path.open("rb")
        handles[record_path] = handle
    return read_sac_record_from(handle, record_path, int(row["record_offset"]), int(row["bytes"]))


def _close_binary_handles(handles: dict[Path, BinaryIO]) -> None:
    for handle in handles.values():
        handle.close()


def _read_rotation_groups(index_path: Path) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return groups
        required = {
            "source_key",
            "receiver_key",
            "record_path",
            "record_offset",
            "bytes",
            "src_component",
            "rec_component",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Sourcepack index missing fields {sorted(missing)}: {index_path}")
        for row in reader:
            groups.setdefault((_row_path_id(row), row["source_key"], row["receiver_key"]), []).append(row)
    return groups


def _rotation_group_sort_key(item: tuple[tuple[str, str, str], list[dict[str, str]]]) -> tuple[str, str, str]:
    pair_key, rows = item
    row = rows[0] if rows else {}
    return (_row_path_id(row), pair_key[1], pair_key[2])


def _legacy_rtz_path(output_dir: Path, legacy_dir_name: str, row: dict[str, str], component_pair: str) -> Path:
    net_pair = f"{row['src_network']}-{row['rec_network']}"
    sta_pair = f"{row['src_station']}-{row['rec_station']}"
    pair_name = f"{net_pair}.{sta_pair}"
    return output_dir / "stack" / legacy_dir_name / pair_name / f"{pair_name}.{component_pair}.sac"


def _row_path_id(row: dict[str, str]) -> str:
    path_id = row.get("path_id", "").strip()
    if path_id:
        return path_id.zfill(8) if path_id.isdigit() else path_id
    try:
        return f"{int(row['src_id']) * 10000 + int(row['rec_id']):08d}"
    except (KeyError, ValueError):
        return ""


def _component_slot(src_component: str, rec_component: str) -> int:
    return _component_rank(src_component) * 3 + _component_rank(rec_component)


def _component_rank(component: str) -> int:
    tail = component.upper()[-1:] if component else ""
    return {"E": 0, "1": 0, "R": 0, "N": 1, "2": 1, "T": 1, "Z": 2, "3": 2}.get(tail, 100)


def _rotated_index_row(
    template: dict[str, str],
    *,
    src_component: str,
    rec_component: str,
    pack_path: Path,
    record_offset: int,
    record_bytes: int,
    final_pair_path: Path,
    header: SacHeader,
    gcarc: float,
    az: float,
    baz: float,
    dist: float,
    storage_kind: str,
) -> dict[str, str]:
    row = {field: "" for field in SOURCEPACK_INDEX_HEADER}
    for key in (
        "path_id",
        "source_key",
        "receiver_key",
        "src_id",
        "rec_id",
        "src_network",
        "src_station",
        "src_location",
        "rec_network",
        "rec_station",
        "rec_location",
    ):
        row[key] = template.get(key, "")
    row.update(
        {
            "timestamp": "ROTATE",
            "path_id": _row_path_id(template),
            "component_slot": str(_component_slot(src_component, rec_component)),
            "src_component": src_component,
            "rec_component": rec_component,
            "npts": str(header.npts),
            "dt": f"{header.get_float('delta'):.9g}",
            "dist": f"{dist:.9g}",
            "az": f"{az:.9g}",
            "baz": f"{baz:.9g}",
            "record_path": pack_path.as_posix(),
            "record_offset": str(record_offset),
            "bytes": str(record_bytes),
            "storage_kind": storage_kind,
            "final_pair_path": final_pair_path.as_posix(),
        }
    )
    return row
