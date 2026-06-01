"""Filter coefficient generation for the SAC2SPEC stage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from scipy.signal import butter

log = logging.getLogger(__name__)


def design_filter(
    delta: float,
    bands: str | Sequence[str],
    output_file: str | Path,
    order: int = 2,
) -> None:
    """Design the broad-band filter plus each configured sub-band filter."""
    output_file = Path(output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(bands, str):
        band_tokens = bands.split()
    else:
        band_tokens = list(bands)

    try:
        band_pairs = [tuple(map(float, band.split("/"))) for band in band_tokens]
    except ValueError as exc:
        raise ValueError(f"Bad band specification: {bands}") from exc

    fs = 1.0 / delta
    fnyq = fs / 2.0

    overall_min = min(low for low, _high in band_pairs)
    overall_max = max(high for _low, high in band_pairs)

    if not (0 < overall_min < overall_max < fnyq):
        raise ValueError(
            f"Invalid overall band {overall_min}/{overall_max} "
            f"for Nyquist={fnyq:.4f} Hz"
        )

    def normalize(freq: float) -> float:
        return freq / fnyq

    coeffs: list[tuple[str, Sequence[float], Sequence[float]]] = []

    b_all, a_all = butter(order, [normalize(overall_min), normalize(overall_max)], "bandpass")
    coeffs.append((f"{overall_min}/{overall_max}", b_all, a_all))

    for low, high in band_pairs:
        b, a = butter(order, [normalize(low), normalize(high)], "bandpass")
        coeffs.append((f"{low}/{high}", b, a))

    try:
        with output_file.open("w", encoding="ascii") as handle:
            for tag, b, a in coeffs:
                handle.write(f"# {tag}\n")
                handle.write("\t".join(f"{value:.18e}" for value in b) + "\n")
                handle.write("\t".join(f"{value:.18e}" for value in a) + "\n")
        log.info("Filter written to %s", output_file)
    except OSError as exc:
        log.error("Error writing filter file %s: %s", output_file, exc)
        raise
