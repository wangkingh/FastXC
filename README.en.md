# FastXC

[中文主文档](README.md) | [Docs](docs/README.md) | [Configuration](docs/CONFIGURATION.md) | [Outputs](docs/OUTPUTS.md) | [Changelog](CHANGELOG.md)

The v2605 public cleanup, documentation rewrite, and release packaging for this
project were assisted by OpenAI Codex / GPT Pro.

FastXC is a Linux/HPC-first ambient-noise cross-correlation pipeline for SAC
waveforms. Python prepares reusable inventory tables and runtime inputs; the
native CUDA/C backends perform SAC2SPEC, cross-correlation, PWS/TF-PWS, and
related heavy compute.

The supported target is Linux or WSL with NVIDIA CUDA. Native Windows builds
are not supported.

## Get The Code

Clone the repository or unpack a release archive, then enter the project root:

```bash
git clone https://github.com/wangkingh/FastXC.git FastXC
cd FastXC

# Or unpack a release archive.
tar -xf FastXC*.tar.gz
cd FastXC*
```

## Requirements

- Python 3.10 or newer.
- Python dependencies: `numpy`, `pandas`, `scipy`, `tqdm`.
- NVIDIA CUDA Toolkit and a CUDA-capable GPU.
- GNU Make plus a C/CUDA compiler toolchain.
- Linux or WSL. Native Windows builds are not supported.

CUDA architecture is auto-detected. If detection fails, override it explicitly:

```bash
make install ARCH=sm_89
```

## Install

All commands below assume the current directory is the FastXC repository root,
the directory that contains `README.md`, `Makefile`, `fastxc/`, `native/`, and
`example/`.

From the repository root:

```bash
# Activate any Python >= 3.10 environment first.
make install
fastxc doctor
```

`make install` builds the supported native backends, stages them into the Python
package, and installs the editable `fastxc` command.

If you want to build the native binaries without installing the Python package:

```bash
make native-full      # build sac2spec, xc_fast, ncf_pws, and ncf_tfpws
make stage-binaries   # copy built binaries into fastxc/bin/<platform>
```

The native binaries are written to `bin/` and staged package binaries are
written under `fastxc/bin/<platform>/`.

## Command Styles

FastXC can be run in either style:

```bash
# Source-tree style; useful before installing the console command.
python -m fastxc.cli doctor config.ini
python -m fastxc.cli prepare config.ini
python -m fastxc.cli run config.ini

# Packaged CLI style; available after `make install` or `pip install -e .`.
fastxc doctor config.ini
fastxc prepare config.ini
fastxc run config.ini
```

Both styles call the same Python entry point. The packaged `fastxc` command is
shorter; the `python -m fastxc.cli ...` form is convenient when working directly
from a source checkout or a partially installed environment.

## Workflow

FastXC is organized around a reusable inventory:

```bash
fastxc prepare config.ini
fastxc run config.ini
```

`prepare` scans SAC files, assigns NSL IDs, builds the SAC index, filters
allowed station pairs, and writes the inventory metadata. `run` consumes that
prepared inventory and runs the staged FastXC backend.

## Quick Start

Create a starter configuration:

```bash
fastxc init -o config.ini
```

Edit `config.ini`, then run:

```bash
fastxc doctor config.ini
fastxc prepare config.ini
fastxc run config.ini
```

## Bundled Example

The repository includes a small anonymized three-component SAC dataset:

```text
example/
```

After installation and a successful `fastxc doctor`, enter the example
directory and run:

```bash
cd example
fastxc doctor config.ini
fastxc prepare config.ini
fastxc run config.ini
```

The example workspace is written under:

```text
workspace
```

Optional plotting is not part of the standard compute pipeline. Stay in the
`example/` directory. If the current environment does not have `matplotlib`,
install it first:

```bash
python -m pip install matplotlib
```

Then you can inspect the rotated RTZ stack as distance-offset line sections. By
default, the plot overlays only `ZZ`, `ZR`, and `RZ` with different colors so
phase differences are easy to inspect:

```bash
python plot_rtz_distance_lines.py \
  --workspace workspace \
  --lag-window 20 \
  --output workspace/plots/rtz_linear_zz_zr_rz_distance_lines.png
```

The plotter reads only RTZ results under `stack/rtz_*_sourcepack`. If PWS or
TF-PWS stacking is enabled, use `--method pws` or `--method tfpws`.

## Documentation

Longer project notes live under `docs/`:

- [Configuration](docs/CONFIGURATION.md) describes INI fields, path pattern
  rules, and common values.
- [Architecture](docs/ARCHITECTURE.md) describes the current
  `spack -> xcache -> sourcepack` data flow.
- [Outputs](docs/OUTPUTS.md) describes stage-by-stage workspace outputs.
- [Results](docs/RESULTS.md) explains public result artifacts and local output
  retention.

## Key Configuration Fields

Most users mainly edit:

```ini
[seisarray1]
sac_dir = /path/to/sac/root
pattern = {home}/{station}/{YYYY}/{station}.{YYYY}.{JJJ}.{*}.{component}.{suffix}
sta_list = NONE
component_list = E,N,Z

[time_filter]
time_start = 2017-09-01 00:00:00
time_end = 2017-09-30 01:00:00
time_list = NONE

[geometry]
external_geo_tsv = NONE

[compute]
sac_len = 86400
win_len = 3600
shift_len = AUTO
delta = 0.1
normalize = AUTO
bands = 0.1/2
whiten = AFTER
max_lag = 100
stack_flag = 100
workspace_dir = /path/to/output/workspace

[device]
gpu_list = 0
cpu_workers = 20

[advance.compute]
skip_step = -1
phase_only = False
distance_range = -1/50000
azimuth_range = -1/360
group_pair_mode = all
autocorr_mode = off
windows_per_xcache = AUTO
xcache_async_after_sac2spec = True
async_poll_sec = 5
xcache_cleanup_timestamp_spack = True
sourcepack_async_after_xc = True
pre_stack_size = 10
tfpws_band = FULL
tfpws_taper_hz = AUTO

[advance.storage]
unpack_enabled = True
unpack_target = ALL

[debug]
debug = False
```

`pattern` describes paths below `sac_dir`. It must start with `{home}` and
include `{station}`, `{component}` or `{channel}`, plus date fields such as
`{YYYY}` + `{JJJ}` or `{YYYY}` + `{MM}` + `{DD}`. Use `{*}` for an ignored
free-form segment and `{?}` for a short non-path segment. See
[Configuration](docs/CONFIGURATION.md) for the full pattern rules.

`sta_list` belongs to each `[seisarrayN]` or `[seisarrayN.source]` section.
`NONE` means that source is not filtered by station. A station list file has one
station code per line; blank lines and lines starting with `#` are ignored. When
multiple sources are folded into the same group, each source is filtered by its
own `sta_list` before the group merge.

Common field formats:

- `time_start/time_end`: `YYYY-MM-DD HH:MM:SS`.
- `time_list`: one timestamp per line, preferably `YYYY-MM-DD HH:MM:SS`; blank
  lines and `#` comments are ignored; `NONE` uses `time_start/time_end`.
- `bands`: space-separated `fmin/fmax` pairs, such as `0.2/0.4 0.1/0.2`.
- `shift_len`: seconds or `AUTO`; `AUTO` means `shift_len = win_len`.
- `max_lag`: maximum correlation lag in seconds.
- `stack_flag`: three 0/1 bits for linear/PWS/TF-PWS, for example `100` for
  linear only and `111` for all three.
- `distance_range`: `min/max` in km; `-1/50000` is effectively unlimited.
- `azimuth_range`: `min/max` in degrees; `-1/360` is effectively unlimited.
- `group_pair_mode`: `intra` for within-group pairs, `inter` for cross-group
  pairs, or `all` for both.
- `autocorr_mode`: autocorrelation path control. Use `off` to exclude station
  self-pairs, `include` to add self-pairs to normal station pairs, or `only`
  to keep only self-pairs.

`[geometry].external_geo_tsv` points to an optional external station-coordinate
TSV. `NONE` means coordinates are read from SAC header fields
`STLA/STLO/STEL`. Set it to a TSV path when SAC headers are missing coordinates,
the header coordinates are not trustworthy, or the same station code needs
different coordinates by `network` and `location`. The TSV must contain
`station`, `lat`, and `lon`; `latitude` and `longitude` are also accepted. Optional
columns include `ele`/`elevation`, `network`, and `location`. `group` is a
FastXC logical grouping concept and is ignored by external geometry matching.
When a row has both `network` and `location`, FastXC prefers the
`network + station + location` match; otherwise it falls back to matching by
`station`. Coordinates are decimal degrees, with west/south written as negative
values. Relative paths are resolved from the directory where you run `fastxc`;
use an absolute path or run from the configuration file's directory.

```text
station	lat	lon	network	location
A7K2	38.499907	-98.603619	KS	00
```

## Inventory

`fastxc prepare` writes reusable inventory artifacts under `workspace_dir`:

```text
inventory.meta.json
manifest/sac_index.tsv
path_plan/allowed_paths.tsv
path_plan/nsl_catalog.tsv
```

`fastxc run` writes compute artifacts such as:

```text
commands/
filter.txt
spack_by_timestamp/
xcache/
xcache/xcspec_index.tsv
ncf/
sourcepack/
stack/
stack/rtz_*_sourcepack/
result_ncf/
progress/
log/
```

`commands/` contains review copies of the native commands that Python runs.
See [Outputs](docs/OUTPUTS.md) for stage-by-stage artifact details.

Optional standalone tools are handled outside the main pipeline:

```bash
fastxc sac2dat -I /path/to/sac_dir -O /path/to/dat_dir
fastxc unpack -I /path/to/sourcepack_index.tsv -O /path/to/sac_dir
```

## Project Layout

```text
fastxc/              Python package, CLI, config parsing, workflow control
fastxc/inventory/    SAC inventory, source matching, NSL IDs, path planning
fastxc/system/       Executable discovery, logging setup, template export
fastxc/stages/       Pipeline stage orchestration
fastxc/adapters/     Native executable command adapters
fastxc/runtime/      Native subprocess execution and progress polling
fastxc/io/           Binary/index format readers and writers
fastxc/operators/    Python-native xcache, stacking, rotation, and filter operators
fastxc/resources/    Packaged starter config and static resources
docs/                Architecture and public project notes
native/sac2spec/     CUDA SAC2SPEC backend
native/xc/           CUDA cross-correlation backend
native/pws/          CUDA PWS backend
native/tfpws/        CUDA TF-PWS backend
tools/               Optional standalone utilities
configs/             Public smoke-test config
example/             Bundled example with config, anonymized data, and plotting helper
```

## Troubleshooting

Check executable discovery and config parsing:

```bash
fastxc doctor config.ini
```

Check native build settings:

```bash
make -C native print-config
```

Useful machine checks:

```bash
nvcc --version
nvidia-smi
make -C native print-config
fastxc doctor config.ini
```

## Author And Citation

For questions, suggestions, or contributions, please use
[GitHub Issues](https://github.com/wangkingh/FastXC/issues). For direct
discussion, contact the author at
[wkh16@mail.ustc.edu.cn](mailto:wkh16@mail.ustc.edu.cn).

If FastXC helps your research, please consider citing the FastXC method paper:

Wang et al. (2025). [High-performance CPU-GPU Heterogeneous Computing Method
for 9-Component Ambient Noise Cross-correlation](https://doi.org/10.1016/j.eqrea.2024.100357).
Earthquake Research Advances.

## Statement And Acknowledgements

FastXC was originally commissioned by the team of Prof. Weitao Wang at the
Institute of Geophysics, China Earthquake Administration, and developed with
teams led by Prof. Huimin Li, Prof. Guangzhong Sun, and Prof. Chao Wu at the
University of Science and Technology of China. Later optimization and public
packaging were continued by the author.

We thank colleagues and friends from the University of Science and Technology
of China, the Institute of Geophysics, China Earthquake Administration, the
Institute of Earthquake Forecasting, China Earthquake Administration, and the
Institute of Geology and Geophysics, Chinese Academy of Sciences, for testing,
trial runs, and feedback.

The v2605 public cleanup, documentation rewrite, and release packaging for this
project were assisted by OpenAI Codex / GPT Pro. The earlier README title
illustration was generated with ChatGPT.

## References

- Wang et al. (2025). [High-performance CPU-GPU Heterogeneous Computing Method
  for 9-Component Ambient Noise Cross-correlation](https://doi.org/10.1016/j.eqrea.2024.100357).
  Earthquake Research Advances.
- Bensen, G. D., et al. (2007). [Processing seismic ambient noise data to obtain
  reliable broad-band surface wave dispersion measurements](https://dx.doi.org/10.1111/j.1365-246x.2007.03374.x).
  Geophysical Journal International, 169(3), 1239-1260.
- Cupillard, P., et al. (2011). [The one-bit noise correlation: a theory based
  on the concepts of coherent and incoherent noise](https://doi.org/10.1111/j.1365-246X.2010.04923.x).
  Geophysical Journal International, 184(3), 1397-1414.
- Zhang, Y., et al. (2018). [3-D Crustal Shear-Wave Velocity Structure of the
  Taiwan Strait and Fujian, SE China, Revealed by Ambient Noise Tomography](https://doi.org/10.1029/2018JB015938).
  Journal of Geophysical Research: Solid Earth, 123(9), 8016-8031.

## License

FastXC is released under the MIT License. See [LICENSE](LICENSE).
