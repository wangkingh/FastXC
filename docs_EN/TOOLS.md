# FastXC Tool Commands

This page documents helper commands outside the normal FastXC main workflow.
They are useful for debugging, format conversion, manual export, recovering an
existing workspace, and inspecting intermediate results. For normal production
runs and compute-stage reruns, use the workflow commands described in the
README.

```bash
fastxc prepare config.ini
fastxc run config.ini
```

## Quick Reference

| Command | Purpose | Common input | Common output |
| --- | --- | --- | --- |
| `fastxc sac2dat` | Convert SAC to text DAT | Directory containing `.sac` files | `.dat` text files |
| `fastxc sourcepack` | Rebuild SourcePack indexes from XC pack output | `ncf/` or `ncf/xcpack/` | `sourcepack/<timestamp>/sourcepack_index.tsv` |
| `fastxc unpack` | Export ordinary SAC files from SourcePack | SourcePack directory or `sourcepack_index.tsv` | `.sac` files |
| `fastxc plot-rtz-grid` | Plot unpacked single-component or 3x3 virtual-source gathers | `result_ncf/ncf_*_BHZ` or `result_ncf/ncf_*_RTZ` | PNG |
| `fastxc extract-stepack` | Extract one station's StepPack spectra, optionally with a plot | `workspace/stepack` | `.mat`, optional PNG |
| `fastxc plot-stepack-mat` | Plot a StepPack `.mat` spectrum file | `.mat` from `extract-stepack` | PNG |

## `fastxc unpack`

`unpack` exports SourcePack records to traditional SAC files. The normal
`fastxc run` path can perform this automatically according to
`[advance.storage]`; this command is mainly for manual export or re-export.

```bash
fastxc unpack \
  -I workspace/stack/linearstack_sourcepack/STACK \
  -O workspace/manual_export/ncf_linear \
  -T 4
```

You can also point to one index file:

```bash
fastxc unpack \
  -I workspace/stack/rtz_linearstack_sourcepack/STACK/sourcepack_index.tsv \
  -O workspace/manual_export/ncf_linear_RTZ
```

Options:

- `-I, --input`: SourcePack directory or `sourcepack_index.tsv`.
- `-O, --output`: output SAC root directory.
- `-T, --threads`: parallel output writers, default `1`.

Output files are grouped by virtual source and receiver:

```text
manual_export/ncf_linear/VV.AAA/VV.BBB/VV-VV.AAA-BBB.Z-Z.ncf.SAC
```

## `fastxc sourcepack`

`sourcepack` manually builds SourcePack indexes from native XC `xcpack` output.
Use it when XC completed but SourcePack was interrupted, when you need to
rebuild indexes from an existing `ncf/xcpack/`, or when debugging SourcePack
sorting/index content.

```bash
fastxc sourcepack \
  -I workspace/ncf \
  -O workspace/sourcepack
```

You can point directly to `xcpack`:

```bash
fastxc sourcepack \
  -I workspace/ncf/xcpack \
  -O workspace/sourcepack
```

Options:

- `-I, --input`: XC output root or `xcpack` directory.
- `-O, --output`: SourcePack output directory.
- `--keep-order`: keep XC encounter order.
- `--sort`: sort by source/receiver/component; this is the default.

The indexes point to real payload inside `ncf/xcpack/*.xcpack`; they do not
copy cross-correlation data.

## `fastxc sac2dat`

`sac2dat` converts SAC files to text DAT files for quick inspection, plotting,
or scripts that only accept text.

```bash
fastxc sac2dat \
  -I /path/to/sac_dir \
  -O /path/to/dat_dir
```

This is suitable for small checks. For large results, keep SAC or SourcePack to
avoid expanding storage with text output.

## `fastxc plot-rtz-grid`

`plot-rtz-grid` reads already unpacked `result_ncf` SAC directories, selects one
virtual source, and plots virtual-source gathers ordered by distance. It infers
the layout from filenames ending in `src_component-rec_component.ncf.SAC`:

- single-component results, such as `BHZ-BHZ` or `Z-Z`, are drawn as one panel;
- three-component results, such as `R/T/Z x R/T/Z` or `E/N/Z x E/N/Z`, are
  drawn as a 3x3 grid.

```bash
fastxc plot-rtz-grid \
  -I workspace/result_ncf/ncf_linear_RTZ \
  --source 45002 \
  -O workspace/plots/rtz_grid_45002.png
```

The same command handles single-component output:

```bash
fastxc plot-rtz-grid \
  -I workspace/result_ncf/ncf_linear_BHZ \
  --source 45002 \
  -O workspace/plots/linear_BHZ_45002.png
```

Important options:

- `-I, --input`: unpacked SAC directory such as `ncf_linear_BHZ` or
  `ncf_linear_RTZ`.
- `--source`: virtual source station, or `NET.STA`.
- `-O, --output`: PNG output path; default is under the input directory.
- `--receiver`: include only selected receivers; can be repeated.
- `--lag-window`: plotting window around zero lag, in seconds.
- `--min-distance`, `--max-distance`: receiver distance filters in km.
- `--max-receivers`, `--sample-stride`: limit or thin plotted receivers.
- `--scale`: horizontal trace scale; default is automatic.

This tool reads exported SAC files, not SourcePack directly. Distance uses the
SAC `dist` header when available, or approximates from `gcarc`.

## `fastxc extract-stepack`

`extract-stepack` reads SAC2SPEC output under `workspace/stepack/` and extracts
one station's multi-step spectra for a given timestamp. It writes MATLAB-readable
`.mat` files for inspecting the intermediate result between SAC2SPEC and XC.

```bash
fastxc extract-stepack \
  --workspace workspace \
  --timestamp 2023.001.0000 \
  --station A7K2 \
  --components E,N,Z \
  -O workspace/plots/A7K2.2023.001.0000.stepack.mat
```

Options:

- `--workspace`: FastXC workspace; reads `stepack/*.tsv`.
- `--stepack`: direct stepack directory or one stepack TSV.
- `--timestamp`: timestamp to extract.
- `--station`: station name.
- `--network`, `--location`: optional filters.
- `--components`: comma-separated components; default `ALL`.
- `--component-match`: `exact`, `tail`, or `auto`.
- `--no-compress`: disable `.mat` compression.
- `--plot`: also generate a quick-look PNG after exporting `.mat`.
- `--plot-output`: PNG output path; default is beside the `.mat` file.

`auto` component matching first tries exact names, then tail matching. Thus
`--components E,N,Z` can match `BHE,BHN,BHZ`; use `--components BHZ` when you
only want one raw channel.

Add `--plot` for immediate inspection:

```bash
fastxc extract-stepack \
  --workspace workspace \
  --timestamp 2023.001.0000 \
  --station A7K2 \
  --components E,N,Z \
  -O workspace/plots/A7K2.2023.001.0000.stepack.mat \
  --plot \
  --max-frequency 1.0 \
  --db
```

The default PNG is written beside the `.mat`, for example
`A7K2.2023.001.0000.stepack.amplitude.png`. Plot options such as `--quantity`,
`--db`, frequency limits, smoothing, title, and DPI follow
`plot-stepack-mat`.

The `.mat` contains:

- `spectra`: complex spectra shaped as `component x step x frequency`.
- `frequency_hz`: frequency axis.
- `step_index`, `timestamp`: step number and timestamp.
- `components`, `networks`, `stations`, `locations`: channel metadata.
- `stla`, `stlo`: station coordinates.

## `fastxc plot-stepack-mat`

`plot-stepack-mat` plots `.mat` files exported by `extract-stepack`, drawing a
step-frequency image for each component. A light Gaussian smoothing is applied
by default to make the broad energy structure easier to inspect.

```bash
fastxc plot-stepack-mat \
  -I workspace/plots/A7K2.2023.001.0000.stepack.mat \
  -O workspace/plots/A7K2.2023.001.0000.stepack.png \
  --max-frequency 1.0 \
  --db
```

Options:

- `-I, --input`: `.mat` file from `extract-stepack`.
- `-O, --output`: PNG output path.
- `--quantity`: `amplitude`, `power`, `phase`, `real`, or `imag`.
- `--db`: plot amplitude/power in dB.
- `--min-frequency`, `--max-frequency`: frequency limits.
- `--smooth-step`, `--smooth-frequency`: smoothing strengths.
- `--no-smooth`: disable smoothing.

## Which Tool To Use

- Re-export final SAC: `fastxc unpack`.
- Rebuild SourcePack after completed XC: `fastxc sourcepack`.
- Convert a small SAC subset to text: `fastxc sac2dat`.
- Inspect single-component or RTZ/ENZ 3x3 virtual-source gathers:
  `fastxc plot-rtz-grid`.
- Inspect one station's SAC2SPEC spectra: `fastxc extract-stepack --plot`;
  replot an existing `.mat` with `fastxc plot-stepack-mat`.

The old BigSAC/extract toolchain is retired. Prefer SourcePack `unpack`,
`plot-rtz-grid`, and StepPack inspection tools.
