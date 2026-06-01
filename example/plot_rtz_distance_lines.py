from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fastxc.io import read_sac_record


RTZ_COMPONENTS = ("R", "T", "Z")
DEFAULT_PAIRS = (("Z", "Z"), ("Z", "R"), ("R", "Z"))
PAIR_COLORS = {
    ("Z", "Z"): "black",
    ("Z", "R"): "#d55e00",
    ("R", "Z"): "#0072b2",
}
RTZ_INDEXES = {
    "linear": "stack/rtz_linearstack_sourcepack/STACK/sourcepack_index.tsv",
    "pws": "stack/rtz_pws_sourcepack/STACK/sourcepack_index.tsv",
    "tfpws": "stack/rtz_tfpws_sourcepack/STACK/sourcepack_index.tsv",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RTZ SourcePack stacks as distance-offset line sections."
    )
    parser.add_argument("--workspace", help="FastXC workspace containing stack/rtz_*_sourcepack outputs")
    parser.add_argument("--index", help="single RTZ sourcepack_index.tsv")
    parser.add_argument("--method", default="linear", choices=sorted(RTZ_INDEXES), help="RTZ stack method for --workspace")
    parser.add_argument("--output", help="PNG path; defaults to <workspace>/plots/rtz_<method>_distance_lines.png")
    parser.add_argument("--title", default="AUTO", help="figure title")
    parser.add_argument("--pairs", default="ZZ,ZR,RZ", help="component pairs to overlay, e.g. ZZ,ZR,RZ or Z-Z,Z-R,R-Z")
    parser.add_argument("--lag-window", type=float, default=20.0, help="half window in seconds around zero lag; 0 disables cropping")
    parser.add_argument("--scale", type=float, default=0.0, help="amplitude scale in km; 0 means auto")
    parser.add_argument("--linewidth", type=float, default=0.65, help="trace line width")
    args = parser.parse_args()

    if bool(args.workspace) == bool(args.index):
        parser.error("provide exactly one of --workspace or --index")

    if args.workspace:
        workspace = Path(args.workspace).expanduser().resolve()
        index = workspace / RTZ_INDEXES[args.method]
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else workspace / "plots" / f"rtz_{args.method}_zz_zr_rz_distance_lines.png"
        )
        title = f"RTZ {args.method.upper()} ZZ/ZR/RZ distance-offset line plot" if args.title.upper() == "AUTO" else args.title
    else:
        index = Path(args.index).expanduser().resolve()
        if not args.output:
            parser.error("--output is required with --index")
        output = Path(args.output).expanduser().resolve()
        title = "RTZ ZZ/ZR/RZ distance-offset line plot" if args.title.upper() == "AUTO" else args.title

    rows = read_rows(index)
    pairs = parse_pairs(args.pairs)
    traces = load_rtz_traces(rows, pairs)
    plot_rtz_lines(traces, pairs, title, output, lag_window=args.lag_window, scale=args.scale, linewidth=args.linewidth)
    print(f"Wrote {output}")


def read_rows(index_path: Path) -> list[dict[str, str]]:
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Empty index: {index_path}")
        required = {
            "src_component",
            "rec_component",
            "dist",
            "dt",
            "record_path",
            "record_offset",
            "bytes",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{index_path} is missing required fields: {', '.join(sorted(missing))}")
        return list(reader)


def parse_pairs(value: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for raw in value.split(","):
        item = raw.strip().upper().replace("-", "")
        if not item:
            continue
        if len(item) != 2 or item[0] not in RTZ_COMPONENTS or item[1] not in RTZ_COMPONENTS:
            raise ValueError(f"Invalid component pair: {raw!r}")
        pair = (item[0], item[1])
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("--pairs must contain at least one RTZ component pair")
    return tuple(pairs)


def pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}{pair[1]}"


def load_rtz_traces(
    rows: list[dict[str, str]],
    pairs: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], list[tuple[float, np.ndarray, float]]]:
    grouped: dict[tuple[str, str], list[tuple[float, np.ndarray, float]]] = {
        pair: [] for pair in pairs
    }

    for row in rows:
        key = (row["src_component"], row["rec_component"])
        if key not in grouped:
            continue
        header, data = read_sac_record(row["record_path"], int(row["record_offset"]), int(row["bytes"]))
        dt = float(header.get_float("delta"))
        if not np.isfinite(dt) or dt <= 0:
            dt = float(row["dt"])
        grouped[key].append((float(row["dist"]), np.asarray(data, dtype=np.float32), dt))

    for key, values in grouped.items():
        if not values:
            raise ValueError(f"No traces for component pair {key[0]}-{key[1]}")
        values.sort(key=lambda item: item[0])
    return grouped


def plot_rtz_lines(
    traces: dict[tuple[str, str], list[tuple[float, np.ndarray, float]]],
    pairs: tuple[tuple[str, str], ...],
    title: str,
    output: Path,
    *,
    lag_window: float,
    scale: float,
    linewidth: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title(title, fontsize=14)

    all_distances = np.array(
        [distance for values in traces.values() for distance, _, _ in values],
        dtype=np.float32,
    )
    amp_scale = scale if scale > 0 else auto_scale_km(all_distances)

    for pair in pairs:
        src, rec = pair
        color = PAIR_COLORS.get(pair)
        label = pair_label(pair)
        values = traces[pair]
        for line_index, (distance, data, dt) in enumerate(values):
            trace = normalize(data)
            lag = (np.arange(trace.size, dtype=np.float32) - (trace.size - 1) / 2.0) * dt
            if lag_window > 0:
                keep = np.abs(lag) <= lag_window
                if not np.any(keep):
                    raise ValueError(f"No samples within +/-{lag_window:g} s for component pair {src}-{rec}")
                lag = lag[keep]
                trace = trace[keep]
            ax.plot(
                lag,
                distance + trace * amp_scale,
                color=color,
                linewidth=linewidth,
                alpha=0.9,
                label=label if line_index == 0 else None,
            )

    ax.axvline(0.0, color="0.55", linewidth=0.65)
    ax.grid(True, axis="y", color="0.88", linewidth=0.5)
    ax.set_ylabel("Distance (km)")
    ax.set_xlabel("Lag (s)")
    ax.legend(loc="upper right", frameon=True)

    finite_distances = all_distances[np.isfinite(all_distances)]
    if finite_distances.size:
        pad = max(float(amp_scale) * 1.5, 0.25)
        ax.set_ylim(float(np.nanmin(finite_distances)) - pad, float(np.nanmax(finite_distances)) + pad)
    if lag_window > 0:
        ax.set_xlim(-lag_window, lag_window)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def auto_scale_km(distances: np.ndarray) -> float:
    unique = np.unique(np.round(distances[np.isfinite(distances)], decimals=6))
    if unique.size >= 2:
        diffs = np.diff(np.sort(unique))
        diffs = diffs[diffs > 0]
        if diffs.size:
            return max(float(np.median(diffs)) * 0.38, 0.05)
    if unique.size == 1:
        return max(abs(float(unique[0])) * 0.06, 0.5)
    return 1.0


def normalize(data: np.ndarray) -> np.ndarray:
    scale = float(np.nanmax(np.abs(data)))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(data, dtype=np.float32)
    return data / scale


if __name__ == "__main__":
    main()
