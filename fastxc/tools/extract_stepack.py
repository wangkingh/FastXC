from __future__ import annotations

import argparse
import csv
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


STEPACK_HEADER_FORMAT = "<8sIIIIQQQIIIIffQQQQ64s64s"
STEPACK_HEADER_SIZE = struct.calcsize(STEPACK_HEADER_FORMAT)
NSLC_ENTRY_FORMAT = "<IIff16s32s16s16s32s"
NSLC_ENTRY_SIZE = struct.calcsize(NSLC_ENTRY_FORMAT)
STEPACK_MAGIC = b"FXCSTPK"
STEPACK_VERSION = 3
STEPACK_LAYOUT_PITCHED_STEP_NSLC_FREQ = 2


@dataclass(frozen=True)
class StepackHeader:
    pack_path: Path
    version: int
    header_size: int
    nslc_entry_size: int
    layout: int
    batch_seq: int
    start_group: int
    group_count: int
    worker_id: int
    nstep: int
    nslc_count: int
    nspec: int
    dt: float
    df: float
    nslc_table_bytes: int
    payload_offset: int
    payload_bytes: int
    pitch_step_bytes: int
    first_timestamp: str


@dataclass(frozen=True)
class StepackFragment:
    timestamp: str
    tsv_path: Path
    pack_path: Path
    nslc_start: int
    nslc_count: int
    batch_nslc_count: int
    nspec: int
    nstep: int
    dt: float
    df: float
    payload_offset: int
    pitch_step_bytes: int
    nslc_step_bytes: int


@dataclass(frozen=True)
class NslcEntry:
    file_index: int
    nsl_id: int
    stla: float
    stlo: float
    network: str
    station: str
    location: str
    component: str
    fragment: StepackFragment

    @property
    def key(self) -> str:
        return f"{self.network}.{self.station}.{self.location}.{self.component}"


@dataclass(frozen=True)
class ExtractedStepack:
    output_path: Path
    timestamp: str
    station: str
    component_count: int
    nstep: int
    nspec: int
    plot_output_path: Path | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract one station's SAC2SPEC spectra from FastXC stepack output."
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    result = run(args)
    print(
        f"Wrote {result.output_path} "
        f"({result.component_count} component(s), {result.nstep} step(s), {result.nspec} frequency bin(s))."
    )
    if result.plot_output_path is not None:
        print(f"Wrote {result.plot_output_path}")
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", help="FastXC workspace containing stepack/")
    parser.add_argument("--stepack", help="stepack directory or a single stepack TSV")
    parser.add_argument("--timestamp", required=True, help="timestamp to extract, e.g. 2011.356.0000")
    parser.add_argument("--station", required=True, help="station code or full key fragment")
    parser.add_argument("--network", help="optional network filter")
    parser.add_argument("--location", help="optional location filter")
    parser.add_argument(
        "--components",
        default="ALL",
        help="comma-separated components to extract; default ALL keeps all matched components",
    )
    parser.add_argument(
        "--component-match",
        choices=("exact", "tail", "auto"),
        default="auto",
        help="component matching mode for --components",
    )
    parser.add_argument("-O", "--output", required=True, help="output .mat path")
    parser.add_argument("--no-compress", action="store_true", help="disable scipy savemat compression")
    parser.add_argument("--plot", action="store_true", help="also write a quick-look PNG after exporting .mat")
    parser.add_argument("--plot-output", help="output PNG path; defaults beside the .mat file")
    parser.add_argument("--quantity", choices=("amplitude", "power", "phase", "real", "imag"), default="amplitude")
    parser.add_argument("--db", action="store_true", help="plot amplitude/power in dB")
    parser.add_argument("--min-frequency", type=float, default=0.0, help="minimum frequency in Hz")
    parser.add_argument("--max-frequency", type=float, help="maximum frequency in Hz")
    parser.add_argument("--smooth-step", type=float, default=0.35, help="Gaussian smoothing sigma along step axis")
    parser.add_argument("--smooth-frequency", type=float, default=1.0, help="Gaussian smoothing sigma along frequency-bin axis")
    parser.add_argument("--no-smooth", action="store_true", help="disable plot smoothing")
    parser.add_argument("--plot-title", default="AUTO", help="figure title for --plot")
    parser.add_argument("--dpi", type=int, default=180, help="output figure DPI for --plot")


def run(args: argparse.Namespace) -> ExtractedStepack:
    if bool(args.workspace) == bool(args.stepack):
        raise ValueError("provide exactly one of --workspace or --stepack")
    stepack_input = Path(args.stepack).expanduser() if args.stepack else Path(args.workspace).expanduser() / "stepack"
    result = extract_stepack_to_mat(
        stepack_input,
        timestamp=args.timestamp,
        station=args.station,
        output=args.output,
        network=args.network,
        location=args.location,
        components=args.components,
        component_match=args.component_match,
        compress=not args.no_compress,
    )
    if not _should_plot(args):
        return result

    plot_path = _plot_extracted_stepack(result.output_path, args)
    return ExtractedStepack(
        result.output_path,
        result.timestamp,
        result.station,
        result.component_count,
        result.nstep,
        result.nspec,
        plot_path,
    )


def extract_stepack_to_mat(
    stepack_input: str | Path,
    *,
    timestamp: str,
    station: str,
    output: str | Path,
    network: str | None = None,
    location: str | None = None,
    components: str | Iterable[str] = "ALL",
    component_match: str = "auto",
    compress: bool = True,
) -> ExtractedStepack:
    fragments = [
        fragment
        for fragment in discover_stepack_fragments(stepack_input)
        if fragment.timestamp == timestamp
    ]
    if not fragments:
        available = sorted({fragment.timestamp for fragment in discover_stepack_fragments(stepack_input)})
        preview = ", ".join(available[:12])
        suffix = "..." if len(available) > 12 else ""
        raise FileNotFoundError(f"Timestamp {timestamp!r} not found. Available: {preview}{suffix}")

    entries = []
    for fragment in fragments:
        entries.extend(read_fragment_nslc_entries(fragment))
    selected = select_nslc_entries(
        entries,
        station=station,
        network=network,
        location=location,
        components=components,
        component_match=component_match,
    )
    if not selected:
        raise FileNotFoundError(f"No NSLC entries matched station={station!r} timestamp={timestamp!r}")

    shape = _common_shape(selected)
    spectra = np.empty((len(selected), shape[0], shape[1]), dtype=np.complex64)
    for index, entry in enumerate(selected):
        spectra[index, :, :] = read_entry_spectra(entry)

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_stepack_mat(output_path, timestamp, selected, spectra, compress=compress)
    return ExtractedStepack(output_path, timestamp, selected[0].station, len(selected), shape[0], shape[1])


def _should_plot(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "plot", False) or getattr(args, "plot_output", None))


def _plot_extracted_stepack(mat_path: Path, args: argparse.Namespace) -> Path:
    from fastxc.tools.plot_stepack_mat import plot_stepack_mat

    output = getattr(args, "plot_output", None) or default_plot_output_path(
        mat_path,
        quantity=args.quantity,
        db=args.db,
    )
    return plot_stepack_mat(
        mat_path,
        output=output,
        max_frequency=args.max_frequency,
        min_frequency=args.min_frequency,
        quantity=args.quantity,
        db=args.db,
        smooth_step=0.0 if args.no_smooth else args.smooth_step,
        smooth_frequency=0.0 if args.no_smooth else args.smooth_frequency,
        dpi=args.dpi,
        title=args.plot_title,
    )


def default_plot_output_path(mat_path: str | Path, *, quantity: str = "amplitude", db: bool = False) -> Path:
    path = Path(mat_path).expanduser().resolve()
    suffix = f".{quantity}{'_db' if db and quantity in {'amplitude', 'power'} else ''}.png"
    if path.suffix:
        return path.with_suffix(suffix)
    return path.with_name(path.name + suffix)


def discover_stepack_fragments(stepack_input: str | Path) -> list[StepackFragment]:
    root = Path(stepack_input).expanduser().resolve()
    if root.is_dir():
        tsv_paths = sorted(root.glob("*.tsv"))
    elif root.is_file() and root.suffix.lower() == ".tsv":
        tsv_paths = [root]
    else:
        raise FileNotFoundError(f"Stepack input not found: {root}")
    if not tsv_paths:
        raise FileNotFoundError(f"No stepack TSV files found under {root}")

    fragments: list[StepackFragment] = []
    for tsv_path in tsv_paths:
        fragments.extend(_read_stepack_tsv(tsv_path))
    return fragments


def read_fragment_nslc_entries(fragment: StepackFragment) -> list[NslcEntry]:
    header = read_stepack_header(fragment.pack_path)
    _validate_fragment_header(fragment, header)

    entries: list[NslcEntry] = []
    with fragment.pack_path.open("rb") as handle:
        for file_index in range(fragment.nslc_start, fragment.nslc_start + fragment.nslc_count):
            handle.seek(header.header_size + file_index * header.nslc_entry_size)
            raw = handle.read(header.nslc_entry_size)
            if len(raw) != header.nslc_entry_size:
                raise EOFError(f"{fragment.pack_path}: truncated NSLC entry at index {file_index}")
            entry = _parse_nslc_entry(raw, fragment)
            if entry.file_index != file_index:
                raise ValueError(
                    f"{fragment.pack_path}: NSLC entry index mismatch "
                    f"({entry.file_index} != {file_index})"
                )
            entries.append(entry)
    return entries


def read_entry_spectra(entry: NslcEntry) -> np.ndarray:
    fragment = entry.fragment
    out = np.empty((fragment.nstep, fragment.nspec), dtype=np.complex64)
    byte_count = fragment.nslc_step_bytes
    expected = fragment.nspec * 8
    if byte_count != expected:
        raise ValueError(
            f"Unexpected nslc_step_bytes for {entry.key}: {byte_count} != nspec*8 ({expected})"
        )
    with fragment.pack_path.open("rb") as handle:
        for step_index in range(fragment.nstep):
            offset = (
                fragment.payload_offset
                + step_index * fragment.pitch_step_bytes
                + entry.file_index * fragment.nslc_step_bytes
            )
            handle.seek(offset)
            raw = handle.read(byte_count)
            if len(raw) != byte_count:
                raise EOFError(
                    f"{fragment.pack_path}: truncated spectrum for {entry.key} "
                    f"step={step_index} offset={offset}"
                )
            floats = np.frombuffer(raw, dtype="<f4")
            out[step_index, :] = floats.reshape(-1, 2)[:, 0] + 1j * floats.reshape(-1, 2)[:, 1]
    return out


def select_nslc_entries(
    entries: Iterable[NslcEntry],
    *,
    station: str,
    network: str | None,
    location: str | None,
    components: str | Iterable[str],
    component_match: str,
) -> list[NslcEntry]:
    station_query = station.strip()
    component_queries = _component_queries(components)
    matched = [
        entry
        for entry in entries
        if _matches_station(entry, station_query)
        and _optional_equal(entry.network, network)
        and _optional_equal(entry.location, location)
    ]
    if component_queries:
        matched = [
            entry
            for entry in matched
            if _matches_component(entry.component, component_queries, component_match)
        ]
    return sorted(matched, key=lambda entry: (_component_sort_key(entry.component), entry.component, entry.key))


def save_stepack_mat(
    output_path: Path,
    timestamp: str,
    entries: list[NslcEntry],
    spectra: np.ndarray,
    *,
    compress: bool,
) -> None:
    try:
        from scipy.io import savemat
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("scipy is required to write MATLAB .mat files") from exc

    fragment = entries[0].fragment
    frequency = np.arange(fragment.nspec, dtype=np.float32) * np.float32(fragment.df)
    savemat(
        output_path,
        {
            "spectra": spectra,
            "frequency_hz": frequency,
            "step_index": np.arange(fragment.nstep, dtype=np.int32),
            "timestamp": timestamp,
            "dt": np.float32(fragment.dt),
            "df": np.float32(fragment.df),
            "components": np.array([entry.component for entry in entries], dtype=object),
            "networks": np.array([entry.network for entry in entries], dtype=object),
            "stations": np.array([entry.station for entry in entries], dtype=object),
            "locations": np.array([entry.location for entry in entries], dtype=object),
            "nsl_id": np.array([entry.nsl_id for entry in entries], dtype=np.int32),
            "stla": np.array([entry.stla for entry in entries], dtype=np.float32),
            "stlo": np.array([entry.stlo for entry in entries], dtype=np.float32),
            "source_pack": np.array([entry.fragment.pack_path.as_posix() for entry in entries], dtype=object),
        },
        do_compression=compress,
    )


def read_stepack_header(pack_path: str | Path) -> StepackHeader:
    path = Path(pack_path).expanduser().resolve()
    with path.open("rb") as handle:
        raw = handle.read(STEPACK_HEADER_SIZE)
    if len(raw) != STEPACK_HEADER_SIZE:
        raise EOFError(f"{path}: truncated stepack header")
    values = struct.unpack(STEPACK_HEADER_FORMAT, raw)
    magic = values[0].split(b"\0", 1)[0]
    if magic != STEPACK_MAGIC:
        raise ValueError(f"{path}: unsupported stepack magic {magic!r}")
    header = StepackHeader(
        pack_path=path,
        version=values[1],
        header_size=values[2],
        nslc_entry_size=values[3],
        layout=values[4],
        batch_seq=values[5],
        start_group=values[6],
        group_count=values[7],
        worker_id=values[8],
        nstep=values[9],
        nslc_count=values[10],
        nspec=values[11],
        dt=values[12],
        df=values[13],
        nslc_table_bytes=values[14],
        payload_offset=values[15],
        payload_bytes=values[16],
        pitch_step_bytes=values[17],
        first_timestamp=_fixed_string(values[18]),
    )
    if header.version != STEPACK_VERSION or header.layout != STEPACK_LAYOUT_PITCHED_STEP_NSLC_FREQ:
        raise ValueError(f"{path}: unsupported stepack version/layout {header.version}/{header.layout}")
    if header.header_size != STEPACK_HEADER_SIZE or header.nslc_entry_size != NSLC_ENTRY_SIZE:
        raise ValueError(f"{path}: unexpected stepack ABI sizes")
    return header


def _read_stepack_tsv(tsv_path: Path) -> list[StepackFragment]:
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return []
        required = {
            "timestamp",
            "pack_path",
            "nstep",
            "nslc_start",
            "nslc_count",
            "batch_nslc_count",
            "nspec",
            "dt",
            "df",
            "payload_offset",
            "pitch_step_bytes",
            "nslc_step_bytes",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{tsv_path}: missing stepack TSV fields {sorted(missing)}")
        return [_fragment_from_row(tsv_path, row) for row in reader]


def _fragment_from_row(tsv_path: Path, row: dict[str, str]) -> StepackFragment:
    pack_path = _resolve_pack_path(row["pack_path"], tsv_path)
    return StepackFragment(
        timestamp=row["timestamp"],
        tsv_path=tsv_path,
        pack_path=pack_path,
        nslc_start=int(row["nslc_start"]),
        nslc_count=int(row["nslc_count"]),
        batch_nslc_count=int(row["batch_nslc_count"]),
        nspec=int(row["nspec"]),
        nstep=int(row["nstep"]),
        dt=float(row["dt"]),
        df=float(row["df"]),
        payload_offset=int(row["payload_offset"]),
        pitch_step_bytes=int(row["pitch_step_bytes"]),
        nslc_step_bytes=int(row["nslc_step_bytes"]),
    )


def _resolve_pack_path(text: str, tsv_path: Path) -> Path:
    path = Path(text).expanduser()
    if path.is_file():
        return path.resolve()
    local = tsv_path.parent / path.name
    if local.is_file():
        return local.resolve()
    if not path.is_absolute():
        rel = tsv_path.parent / path
        if rel.is_file():
            return rel.resolve()
    raise FileNotFoundError(f"Stepack pack file not found for {tsv_path}: {text}")


def _parse_nslc_entry(raw: bytes, fragment: StepackFragment) -> NslcEntry:
    values = struct.unpack(NSLC_ENTRY_FORMAT, raw[:NSLC_ENTRY_SIZE])
    return NslcEntry(
        file_index=values[0],
        nsl_id=values[1],
        stla=values[2],
        stlo=values[3],
        network=_fixed_string(values[4]),
        station=_fixed_string(values[5]),
        location=_fixed_string(values[6]),
        component=_fixed_string(values[7]),
        fragment=fragment,
    )


def _validate_fragment_header(fragment: StepackFragment, header: StepackHeader) -> None:
    checks = {
        "nstep": (fragment.nstep, header.nstep),
        "nspec": (fragment.nspec, header.nspec),
        "batch_nslc_count": (fragment.batch_nslc_count, header.nslc_count),
        "payload_offset": (fragment.payload_offset, header.payload_offset),
        "pitch_step_bytes": (fragment.pitch_step_bytes, header.pitch_step_bytes),
    }
    for name, (left, right) in checks.items():
        if left != right:
            raise ValueError(f"{fragment.pack_path}: TSV/header mismatch for {name}: {left} != {right}")
    if fragment.nslc_start < 0 or fragment.nslc_count < 1:
        raise ValueError(f"{fragment.tsv_path}: invalid nslc range")
    if fragment.nslc_start + fragment.nslc_count > header.nslc_count:
        raise ValueError(f"{fragment.tsv_path}: nslc range exceeds batch table")


def _common_shape(entries: list[NslcEntry]) -> tuple[int, int]:
    shapes = {(entry.fragment.nstep, entry.fragment.nspec, entry.fragment.dt, entry.fragment.df) for entry in entries}
    if len(shapes) != 1:
        raise ValueError(f"Selected NSLC entries have inconsistent spectrum shapes: {sorted(shapes)}")
    nstep, nspec, _, _ = next(iter(shapes))
    return int(nstep), int(nspec)


def _component_queries(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        if value.strip().upper() == "ALL":
            return ()
        items = value.split(",")
    else:
        items = value
    return tuple(item.strip() for item in items if item.strip())


def _matches_station(entry: NslcEntry, query: str) -> bool:
    if query == entry.station:
        return True
    return query in {
        f"{entry.network}.{entry.station}",
        f"{entry.network}.{entry.station}.{entry.location}",
        entry.key,
    }


def _matches_component(component: str, queries: tuple[str, ...], mode: str) -> bool:
    upper = component.upper()
    exact = {query.upper() for query in queries}
    if mode in {"exact", "auto"} and upper in exact:
        return True
    if mode in {"tail", "auto"}:
        return any(len(query) == 1 and upper.endswith(query.upper()) for query in queries)
    return False


def _component_sort_key(component: str) -> int:
    tail = component.upper()[-1:] if component else ""
    return {"E": 0, "1": 0, "N": 1, "2": 1, "Z": 2, "3": 2, "R": 3, "T": 4}.get(tail, 100)


def _optional_equal(value: str, query: str | None) -> bool:
    return query is None or value == query


def _fixed_string(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="ignore").strip()


if __name__ == "__main__":
    raise SystemExit(main())
