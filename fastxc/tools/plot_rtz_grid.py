from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fastxc.io import SacHeader, read_sac


PREFERRED_COMPONENT_ORDERS = (
    ("R", "T", "Z"),
    ("E", "N", "Z"),
)
SAC_NULL = -12345.0
_PAIR_RE = re.compile(r"\.([A-Za-z0-9_]+)-([A-Za-z0-9_]+)\.ncf\.sac$", re.IGNORECASE)


@dataclass(frozen=True)
class RtzTrace:
    receiver_key: str
    distance_km: float
    lag: np.ndarray
    data: np.ndarray


@dataclass(frozen=True)
class RtzGridResult:
    output_path: Path
    source_key: str
    receiver_count: int
    trace_count: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot a single-component or 3x3 virtual-source gather from unpacked FastXC SAC output."
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    result = run(args)
    print(f"Wrote {result.output_path}")
    print(
        f"Source {result.source_key}: plotted {result.receiver_count} receiver(s), "
        f"{result.trace_count} trace(s)."
    )
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-I",
        "--input",
        required=True,
        help="unpacked result directory, e.g. workspace/result_ncf/ncf_linear_BHZ or ncf_linear_RTZ",
    )
    parser.add_argument("--source", required=True, help="virtual source station or key, e.g. 45002 or VV.45002")
    parser.add_argument("-O", "--output", help="output PNG path; defaults under <input>/plots")
    parser.add_argument("--title", default="AUTO", help="figure title")
    parser.add_argument("--receiver", action="append", default=[], help="receiver station/key to include; repeatable")
    parser.add_argument("--lag-window", type=float, default=20.0, help="half window in seconds around zero lag")
    parser.add_argument("--scale", type=float, default=0.0, help="amplitude scale in km; 0 means auto")
    parser.add_argument("--max-receivers", type=int, default=0, help="maximum receivers to draw after sorting; 0 draws all")
    parser.add_argument("--sample-stride", type=int, default=1, help="draw every Nth receiver after sorting")
    parser.add_argument("--min-distance", type=float, help="minimum receiver distance in km")
    parser.add_argument("--max-distance", type=float, help="maximum receiver distance in km")
    parser.add_argument("--linewidth", type=float, default=0.55, help="trace line width")
    parser.add_argument("--dpi", type=int, default=180, help="output figure DPI")


def run(args: argparse.Namespace) -> RtzGridResult:
    return plot_unpacked_rtz_grid(
        args.input,
        source=args.source,
        output=args.output,
        title=args.title,
        receivers=args.receiver,
        lag_window=args.lag_window,
        scale=args.scale,
        max_receivers=args.max_receivers,
        sample_stride=args.sample_stride,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        linewidth=args.linewidth,
        dpi=args.dpi,
    )


def plot_unpacked_rtz_grid(
    input_dir: str | Path,
    *,
    source: str,
    output: str | Path | None = None,
    title: str = "AUTO",
    receivers: Iterable[str] = (),
    lag_window: float = 20.0,
    scale: float = 0.0,
    max_receivers: int = 0,
    sample_stride: int = 1,
    min_distance: float | None = None,
    max_distance: float | None = None,
    linewidth: float = 0.55,
    dpi: int = 180,
) -> RtzGridResult:
    if sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")
    if max_receivers < 0:
        raise ValueError("--max-receivers must be >= 0")
    if lag_window < 0:
        raise ValueError("--lag-window must be >= 0")

    root = Path(input_dir).expanduser().resolve()
    source_dir = _resolve_source_dir(root, source)
    source_key = source_dir.name
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else root / "plots" / f"{source_key}.component_grid.png"
    )

    receiver_filters = tuple(item.strip() for item in receivers if item.strip())
    receiver_dirs = _discover_receiver_dirs(source_dir, receiver_filters)
    pairs_by_receiver = _read_receiver_summaries(receiver_dirs)
    pairs_by_receiver = [
        item
        for item in pairs_by_receiver
        if _distance_in_range(item, min_distance=min_distance, max_distance=max_distance)
    ]
    pairs_by_receiver.sort(key=lambda item: (item[0], item[1].name))
    if sample_stride > 1:
        pairs_by_receiver = pairs_by_receiver[::sample_stride]
    if max_receivers > 0 and len(pairs_by_receiver) > max_receivers:
        indexes = np.linspace(0, len(pairs_by_receiver) - 1, max_receivers)
        pairs_by_receiver = [pairs_by_receiver[int(round(index))] for index in indexes]
    if not pairs_by_receiver:
        raise ValueError(f"No receivers with valid component SAC files found for source {source_key}")

    source_components, receiver_components = _infer_component_layout(pairs_by_receiver)
    traces = _load_grid_traces(
        pairs_by_receiver,
        source_components=source_components,
        receiver_components=receiver_components,
        lag_window=lag_window,
    )
    if not any(traces.values()):
        raise ValueError(f"No component traces found for source {source_key}")

    layout_label = _layout_label(source_components, receiver_components)
    figure_title = (
        f"{source_key} {layout_label} virtual-source gather"
        if title.upper() == "AUTO"
        else title
    )
    _plot_grid(
        traces,
        source_components=source_components,
        receiver_components=receiver_components,
        title=figure_title,
        output_path=output_path,
        scale=scale,
        linewidth=linewidth,
        dpi=dpi,
    )

    receiver_count = len({trace.receiver_key for values in traces.values() for trace in values})
    trace_count = sum(len(values) for values in traces.values())
    return RtzGridResult(output_path, source_key, receiver_count, trace_count)


def _resolve_source_dir(root: Path, source: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")
    matches = [path for path in root.iterdir() if path.is_dir() and _matches_station_key(path.name, source)]
    if not matches:
        available = ", ".join(path.name for path in sorted(root.iterdir()) if path.is_dir())
        raise FileNotFoundError(f"Source {source!r} not found under {root}. Available sources: {available}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Source {source!r} matched multiple directories: {names}")
    return matches[0]


def _discover_receiver_dirs(source_dir: Path, receiver_filters: tuple[str, ...]) -> list[Path]:
    receiver_dirs = [path for path in source_dir.iterdir() if path.is_dir()]
    if not receiver_filters:
        return sorted(receiver_dirs, key=lambda path: path.name)
    selected = [
        path
        for path in receiver_dirs
        if any(_matches_station_key(path.name, query) for query in receiver_filters)
    ]
    missing = [
        query
        for query in receiver_filters
        if not any(_matches_station_key(path.name, query) for path in selected)
    ]
    if missing:
        raise FileNotFoundError(f"Receiver(s) not found for {source_dir.name}: {', '.join(missing)}")
    return sorted(selected, key=lambda path: path.name)


def _read_receiver_summary(receiver_dir: Path) -> tuple[float, Path, dict[tuple[str, str], Path]]:
    pairs: dict[tuple[str, str], Path] = {}
    for path in receiver_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".sac":
            continue
        match = _PAIR_RE.search(path.name)
        if match is None:
            continue
        pair = (match.group(1).upper(), match.group(2).upper())
        pairs[pair] = path
    if not pairs:
        raise ValueError(f"No component SAC files found under {receiver_dir}")

    distance = _read_first_valid_distance(pairs.values())
    return distance, receiver_dir, pairs


def _read_receiver_summaries(receiver_dirs: list[Path]) -> list[tuple[float, Path, dict[tuple[str, str], Path]]]:
    summaries: list[tuple[float, Path, dict[tuple[str, str], Path]]] = []
    for receiver_dir in receiver_dirs:
        try:
            summaries.append(_read_receiver_summary(receiver_dir))
        except ValueError:
            continue
    return summaries


def _infer_component_layout(
    receiver_summaries: list[tuple[float, Path, dict[tuple[str, str], Path]]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    source_components = _sort_components(
        {pair[0] for _, _, pairs in receiver_summaries for pair in pairs}
    )
    receiver_components = _sort_components(
        {pair[1] for _, _, pairs in receiver_summaries for pair in pairs}
    )
    layout_size = len(source_components) * len(receiver_components)
    if layout_size in {1, 9}:
        return source_components, receiver_components
    raise ValueError(
        "Unsupported component layout: "
        f"{len(source_components)} source component(s) x {len(receiver_components)} receiver component(s). "
        "Expected single-component 1x1 or three-component 3x3 output."
    )


def _sort_components(components: set[str]) -> tuple[str, ...]:
    for preferred in PREFERRED_COMPONENT_ORDERS:
        if components == set(preferred):
            return preferred
    return tuple(sorted(components, key=_component_sort_key))


def _component_sort_key(component: str) -> tuple[int, str]:
    tail = component.upper()[-1:] if component else ""
    rank = {"E": 0, "1": 0, "R": 0, "N": 1, "2": 1, "T": 1, "Z": 2, "3": 2}.get(tail, 99)
    return rank, component


def _layout_label(source_components: tuple[str, ...], receiver_components: tuple[str, ...]) -> str:
    if len(source_components) == 1 and len(receiver_components) == 1:
        return f"{source_components[0]}-{receiver_components[0]}"
    return f"{''.join(source_components)} x {''.join(receiver_components)}"


def _load_grid_traces(
    receiver_summaries: list[tuple[float, Path, dict[tuple[str, str], Path]]],
    *,
    source_components: tuple[str, ...],
    receiver_components: tuple[str, ...],
    lag_window: float,
) -> dict[tuple[str, str], list[RtzTrace]]:
    traces: dict[tuple[str, str], list[RtzTrace]] = {
        (src, rec): [] for src in source_components for rec in receiver_components
    }
    for distance, receiver_dir, pairs in receiver_summaries:
        for pair, path in pairs.items():
            if pair not in traces:
                continue
            header, data = read_sac(path)
            lag = _lag_axis(header, data.size)
            data = np.asarray(data, dtype=np.float32)
            if lag_window > 0:
                keep = np.abs(lag) <= lag_window
                if not np.any(keep):
                    continue
                lag = lag[keep]
                data = data[keep]
            traces[pair].append(
                RtzTrace(
                    receiver_key=receiver_dir.name,
                    distance_km=distance,
                    lag=lag,
                    data=data,
                )
            )
    for values in traces.values():
        values.sort(key=lambda item: (item.distance_km, item.receiver_key))
    return traces


def _plot_grid(
    traces: dict[tuple[str, str], list[RtzTrace]],
    *,
    source_components: tuple[str, ...],
    receiver_components: tuple[str, ...],
    title: str,
    output_path: Path,
    scale: float,
    linewidth: float,
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("matplotlib is required to plot RTZ grids") from exc

    all_distances = np.array(
        [trace.distance_km for values in traces.values() for trace in values],
        dtype=np.float32,
    )
    amp_scale = scale if scale > 0 else _auto_scale_km(all_distances)
    y_min, y_max = _distance_limits(all_distances, amp_scale)

    row_count = len(source_components)
    col_count = len(receiver_components)
    figsize = (7.5, 5.5) if row_count == 1 and col_count == 1 else (4.5 * col_count, 3.5 * row_count)
    fig, axes = plt.subplots(row_count, col_count, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    fig.suptitle(title, fontsize=15)

    for row_index, src_component in enumerate(source_components):
        for col_index, rec_component in enumerate(receiver_components):
            ax = axes[row_index][col_index]
            pair = (src_component, rec_component)
            for trace in traces[pair]:
                normalized = _normalize(trace.data)
                ax.plot(
                    trace.lag,
                    trace.distance_km + normalized * amp_scale,
                    color="#1f77b4",
                    linewidth=linewidth,
                    alpha=0.9,
                )
            ax.set_title(f"{src_component}-{rec_component}", fontsize=11)
            ax.axvline(0.0, color="0.58", linewidth=0.6)
            ax.grid(True, axis="y", color="0.88", linewidth=0.45)
            ax.set_ylim(y_min, y_max)
            if col_index == 0:
                ax.set_ylabel("Distance (km)")
            if row_index == row_count - 1:
                ax.set_xlabel("Lag (s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _read_distance(path: Path) -> float:
    header, _ = read_sac(path)
    distance = header.get_float("dist")
    if _is_defined(distance):
        return float(distance)
    gcarc = header.get_float("gcarc")
    if _is_defined(gcarc):
        return float(gcarc) * 111.1949
    raise ValueError(f"{path}: SAC header does not contain valid dist/gcarc")


def _read_first_valid_distance(paths: Iterable[Path]) -> float:
    last_error: ValueError | None = None
    for path in paths:
        try:
            return _read_distance(path)
        except ValueError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("No SAC files available for distance lookup")


def _lag_axis(header: SacHeader, npts: int) -> np.ndarray:
    delta = header.get_float("delta")
    if not _is_defined(delta) or delta <= 0:
        raise ValueError("SAC header does not contain a valid delta")
    begin = header.get_float("b")
    if _is_defined(begin):
        return begin + np.arange(npts, dtype=np.float32) * float(delta)
    return (np.arange(npts, dtype=np.float32) - (npts - 1) / 2.0) * float(delta)


def _matches_station_key(name: str, query: str) -> bool:
    query = query.strip()
    if not query:
        return False
    if name == query:
        return True
    return "." not in query and name.split(".")[-1] == query


def _distance_in_range(
    item: tuple[float, Path, dict[tuple[str, str], Path]],
    *,
    min_distance: float | None,
    max_distance: float | None,
) -> bool:
    distance = item[0]
    if min_distance is not None and distance < min_distance:
        return False
    if max_distance is not None and distance > max_distance:
        return False
    return True


def _is_defined(value: float) -> bool:
    return np.isfinite(value) and abs(float(value) - SAC_NULL) > 1.0e-3


def _normalize(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(data, dtype=np.float32)
    return data / scale


def _auto_scale_km(distances: np.ndarray) -> float:
    finite = np.unique(np.round(distances[np.isfinite(distances)], decimals=6))
    if finite.size >= 2:
        diffs = np.diff(np.sort(finite))
        diffs = diffs[diffs > 0]
        if diffs.size:
            return max(float(np.median(diffs)) * 0.55, 0.08)
    if finite.size == 1:
        return max(abs(float(finite[0])) * 0.05, 0.5)
    return 1.0


def _distance_limits(distances: np.ndarray, amp_scale: float) -> tuple[float, float]:
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return -1.0, 1.0
    pad = max(float(amp_scale) * 1.7, 0.25)
    return float(np.min(finite)) - pad, float(np.max(finite)) + pad


if __name__ == "__main__":
    raise SystemExit(main())
