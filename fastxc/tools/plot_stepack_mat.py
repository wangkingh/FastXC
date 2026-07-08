from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot spectra exported by fastxc extract-stepack.")
    add_arguments(parser)
    args = parser.parse_args(argv)
    output = plot_stepack_mat(
        args.input,
        output=args.output,
        max_frequency=args.max_frequency,
        min_frequency=args.min_frequency,
        quantity=args.quantity,
        db=args.db,
        smooth_step=0.0 if args.no_smooth else args.smooth_step,
        smooth_frequency=0.0 if args.no_smooth else args.smooth_frequency,
        dpi=args.dpi,
        title=args.title,
    )
    print(f"Wrote {output}")
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-I", "--input", required=True, help="input .mat file from extract-stepack")
    parser.add_argument("-O", "--output", required=True, help="output PNG path")
    parser.add_argument("--quantity", choices=("amplitude", "power", "phase", "real", "imag"), default="amplitude")
    parser.add_argument("--db", action="store_true", help="plot amplitude/power in dB")
    parser.add_argument("--min-frequency", type=float, default=0.0, help="minimum frequency in Hz")
    parser.add_argument("--max-frequency", type=float, help="maximum frequency in Hz")
    parser.add_argument("--smooth-step", type=float, default=0.35, help="Gaussian smoothing sigma along step axis")
    parser.add_argument(
        "--smooth-frequency",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma along frequency-bin axis",
    )
    parser.add_argument("--no-smooth", action="store_true", help="disable plot smoothing")
    parser.add_argument("--title", default="AUTO", help="figure title")
    parser.add_argument("--dpi", type=int, default=180, help="output figure DPI")


def plot_stepack_mat(
    mat_path: str | Path,
    *,
    output: str | Path,
    max_frequency: float | None = None,
    min_frequency: float = 0.0,
    quantity: str = "amplitude",
    db: bool = False,
    smooth_step: float = 0.35,
    smooth_frequency: float = 1.0,
    dpi: int = 180,
    title: str = "AUTO",
) -> Path:
    if smooth_step < 0 or smooth_frequency < 0:
        raise ValueError("smoothing sigmas must be >= 0")
    data = load_stepack_mat(mat_path)
    spectra = data["spectra"]
    frequency = data["frequency_hz"]
    components = data["components"]
    timestamp = data.get("timestamp", "")
    stations = data.get("stations", [])
    station = stations[0] if stations else ""

    keep = frequency >= float(min_frequency)
    if max_frequency is not None:
        keep &= frequency <= float(max_frequency)
    if not np.any(keep):
        raise ValueError("frequency range removed all bins")
    frequency = frequency[keep]
    spectra = spectra[:, :, keep]

    values, colorbar_label, cmap = _quantity_values(spectra, quantity=quantity, db=db)
    values = _smooth_plot_values(
        values,
        spectra,
        quantity=quantity,
        smooth_step=smooth_step,
        smooth_frequency=smooth_frequency,
    )
    output_path = Path(output).expanduser().resolve()
    _plot_values(
        values,
        frequency,
        components,
        title=_title_text(title, station=station, timestamp=timestamp, quantity=quantity, db=db),
        colorbar_label=colorbar_label,
        cmap=cmap,
        output_path=output_path,
        dpi=dpi,
    )
    return output_path


def load_stepack_mat(mat_path: str | Path) -> dict[str, object]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("scipy is required to read MATLAB .mat files") from exc

    path = Path(mat_path).expanduser().resolve()
    raw = loadmat(path, squeeze_me=True)
    if "spectra" not in raw:
        raise ValueError(f"{path}: missing 'spectra' variable")
    spectra = np.asarray(raw["spectra"], dtype=np.complex64)
    if spectra.ndim == 2:
        spectra = spectra[np.newaxis, :, :]
    if spectra.ndim != 3:
        raise ValueError(f"{path}: spectra must have shape component x step x frequency, got {spectra.shape}")

    frequency = np.asarray(raw.get("frequency_hz"), dtype=np.float32).reshape(-1)
    if frequency.size != spectra.shape[2]:
        raise ValueError(f"{path}: frequency_hz size does not match spectra frequency dimension")
    return {
        "spectra": spectra,
        "frequency_hz": frequency,
        "components": _string_list(raw.get("components"), expected=spectra.shape[0], default_prefix="C"),
        "timestamp": _string_scalar(raw.get("timestamp")),
        "stations": _string_list(raw.get("stations"), expected=spectra.shape[0], default_prefix="STA"),
    }


def _quantity_values(
    spectra: np.ndarray,
    *,
    quantity: str,
    db: bool,
) -> tuple[np.ndarray, str, str]:
    if quantity == "amplitude":
        values = np.abs(spectra)
        if db:
            values = 20.0 * np.log10(np.maximum(values, np.finfo(np.float32).tiny))
            return values, "Amplitude (dB)", "viridis"
        return values, "Amplitude", "viridis"
    if quantity == "power":
        values = np.abs(spectra) ** 2
        if db:
            values = 10.0 * np.log10(np.maximum(values, np.finfo(np.float32).tiny))
            return values, "Power (dB)", "viridis"
        return values, "Power", "viridis"
    if quantity == "phase":
        return np.angle(spectra), "Phase (rad)", "twilight"
    if quantity == "real":
        return spectra.real, "Real", "coolwarm"
    if quantity == "imag":
        return spectra.imag, "Imag", "coolwarm"
    raise ValueError(f"Unsupported quantity: {quantity}")


def _smooth_plot_values(
    values: np.ndarray,
    spectra: np.ndarray,
    *,
    quantity: str,
    smooth_step: float,
    smooth_frequency: float,
) -> np.ndarray:
    if smooth_step <= 0 and smooth_frequency <= 0:
        return values
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("scipy is required for smoothed stepack plots") from exc

    sigma = (0.0, float(smooth_step), float(smooth_frequency))
    if quantity == "phase":
        amplitude = np.abs(spectra)
        unit = np.divide(
            spectra,
            amplitude,
            out=np.zeros_like(spectra, dtype=np.complex64),
            where=amplitude > 0,
        )
        real = gaussian_filter(unit.real, sigma=sigma, mode="nearest")
        imag = gaussian_filter(unit.imag, sigma=sigma, mode="nearest")
        return np.angle(real + 1j * imag)
    return gaussian_filter(values, sigma=sigma, mode="nearest")


def _plot_values(
    values: np.ndarray,
    frequency: np.ndarray,
    components: list[str],
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    output_path: Path,
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("matplotlib is required to plot stepack spectra") from exc

    ncomponent = values.shape[0]
    fig_height = max(3.0, 2.25 * ncomponent)
    fig, axes = plt.subplots(ncomponent, 1, figsize=(11.0, fig_height), sharex=True, squeeze=False)
    finite = values[np.isfinite(values)]
    vmin = vmax = None
    if finite.size and cmap in {"viridis"}:
        vmin, vmax = np.nanpercentile(finite, [2.0, 98.0])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin = vmax = None

    image = None
    for index in range(ncomponent):
        ax = axes[index][0]
        label = components[index] if index < len(components) else f"C{index}"
        image = ax.imshow(
            values[index],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[float(frequency[0]), float(frequency[-1]), -0.5, values.shape[1] - 0.5],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(f"{label}\nstep")
        ax.grid(False)
    axes[-1][0].set_xlabel("Frequency (Hz)")
    fig.suptitle(title, fontsize=14)
    if image is not None:
        cbar = fig.colorbar(image, ax=[ax[0] for ax in axes], pad=0.012, fraction=0.035)
        cbar.set_label(colorbar_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _title_text(title: str, *, station: str, timestamp: str, quantity: str, db: bool) -> str:
    if title.upper() != "AUTO":
        return title
    suffix = " dB" if db and quantity in {"amplitude", "power"} else ""
    parts = [part for part in (station, timestamp, f"{quantity}{suffix}") if part]
    return " ".join(parts) if parts else "Stepack spectra"


def _string_scalar(value: object) -> str:
    if value is None:
        return ""
    array = np.asarray(value)
    if array.size == 0:
        return ""
    item = array.reshape(-1)[0]
    if isinstance(item, np.ndarray):
        return "".join(str(part) for part in item.reshape(-1)).strip()
    return str(item).strip()


def _string_list(value: object, *, expected: int, default_prefix: str) -> list[str]:
    if value is None:
        return [f"{default_prefix}{index}" for index in range(expected)]
    array = np.asarray(value, dtype=object).reshape(-1)
    out = [_string_scalar(item) for item in array]
    out = [item for item in out if item]
    if len(out) == expected:
        return out
    if len(out) == 1 and expected == 1:
        return out
    return [f"{default_prefix}{index}" for index in range(expected)]


if __name__ == "__main__":
    raise SystemExit(main())
