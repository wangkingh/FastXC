# FastXC INI Configuration

FastXC uses an INI file to describe input SAC data, time ranges, compute
parameters, device resources, and the output workspace. Start from the packaged
template, then edit the paths and fields that match your dataset:

```bash
fastxc init -o config.ini
```

Most users only need the core fields shown in the root README. This page
collects the full configuration model, path pattern rules, and common values.

## General Conventions

- Paths can be absolute or relative. Data and output paths are usually resolved
  from the command working directory; executable lookup also checks the source
  tree, packaged staged binaries, and the INI directory. Run
  `fastxc doctor config.ini` before long jobs.
- `NONE` means disabled or not provided.
- `AUTO` lets FastXC choose a default implementation or resource value.
- Booleans accept `True/False`, `yes/no`, `1/0`, and `on/off`.
- Data sources use `[seisarrayN]` or `[seisarrayN.source]`; the main compute
  parameters live under `[compute]`.
- Legacy `[arrayN]`, `[preprocess]`, `[xcorr]`, `[stack]`, and related
  compatibility sections are removed. Path planning now uses the current
  `[seisarrayN]` `files_groups` model instead of `files_group1/files_group2`.

## SAC Path Pattern

`[seisarrayN].pattern` describes the relative path of SAC files under
`sac_dir`. FastXC recursively scans `sac_dir`, normalizes each file path, and
parses station, component, date, and optional metadata from the pattern.

Example:

```ini
[seisarray1]
sac_dir = data
pattern = {home}/{station}/{YYYY}/{station}.{YYYY}.{JJJ}.{*}.{component}.{suffix}
component_list = E,N,Z
```

This matches paths such as:

```text
data/A7K2/2022/A7K2.2022.137.raw.E.sac
```

Common rules:

- `{home}` must appear at the beginning and represents the `sac_dir` root.
- `{station}` is required.
- `{component}` or `{channel}` is required. When `{channel}` is used, FastXC
  maps it to component internally.
- A date must be parsable. Prefer `{YYYY}` + `{JJJ}`, or `{YYYY}` + `{MM}` +
  `{DD}`. `{HH}` and `{MI}` can describe hour/minute files.
- `{network}` and `{location}` are optional; missing values default to `VV` and
  `00`.
- Repeated fields must have identical values.
- `{*}` matches arbitrary text and can cross path separators.
- `{?}` matches one short segment and does not cross `/`, dots, spaces, or
  underscores.
- Windows backslashes are normalized to `/`.

Built-in fields include:

```text
YYYY YY MM DD JJJ HH MI
network event station location component channel
sampleF quality locid suffix arrayID
label0 label1 label2 label3 label4 label5 label6 label7 label8 label9
```

When files do not match, check `sac_dir`, whether the pattern starts with
`{home}`, and whether `component_list` matches the component/channel names in
the filenames.

## `[seisarrayN]`

Each section describes one logical array or line. `N` must be `1` to `9` and
becomes the group id in GNSL records. A group can have multiple physical
sources, for example `[seisarray1]` and `[seisarray1.network_b]`; they are
merged into the same logical group.

Common fields:

| Field | Meaning |
| --- | --- |
| `sac_dir` | Root directory of SAC files. |
| `pattern` | Relative path pattern used to parse SAC metadata. |
| `component_list` | Expected components. One item keeps the original label; three items are mapped to `E,N,Z` order for unrotated three-component results. |
| `sta_list` | Optional station whitelist. Use `NONE` to disable. |
| `geo_csv` / `external_geo_tsv` | Optional external station geometry table. |

If geometry cannot be read from SAC headers, provide an external table with
network/station/location and coordinate columns. If external geometry injection
updates no stations and path filtering produces zero allowed paths, FastXC
fails before SAC2SPEC so the empty geometry problem does not reach XC.

## Pair Filtering

Pair filtering is configured under `[compute]`:

| Field | Meaning |
| --- | --- |
| `distance_range` | `min/max` distance in km. `-1/50000` is effectively unlimited. |
| `azimuth_range` | `min/max` azimuth in degrees. `-1/360` is effectively unlimited. |
| `group_pair_mode` | Which array groups may correlate with each other. |
| `allow_autocorr` | Whether station autocorrelation pairs are allowed. |
| `autocorr_mode` | How autocorrelation pairs are handled. |

The result is written to `path_plan/allowed_paths.tsv` and
`path_plan/allowed_path_ids.txt`.

## `[compute]`

Frequently edited fields:

| Field | Meaning |
| --- | --- |
| `start_date`, `end_date` | Time range. |
| `win_len`, `shift_len` | Window length and step in seconds. `shift_len = AUTO` means `shift_len = win_len`. |
| `fs` | Target sampling rate. |
| `freq_norm`, `time_norm`, `whiten` | Preprocessing and spectral normalization switches. |
| `max_lag` | Maximum correlation lag in seconds. |
| `stack_flag` | Three bits controlling linear/PWS/TF-PWS, for example `100` or `111`. |
| `workspace_dir` | Output workspace. |

The normal workflow is:

```bash
fastxc prepare config.ini
fastxc run config.ini
```

Stage aliases such as `fastxc sac2spec config.ini`, `fastxc xc config.ini`,
`fastxc stack config.ini`, and `fastxc rotate config.ini` can rerun part of an
existing prepared workspace.

## Storage And Result Naming

`[advance.storage].unpack_enabled = True` exports final SourcePack products to
SAC under `workspace_dir/result_ncf`. `unpack_target = ALL` exports both stack
and rotated products when available.

Single-component stack output keeps the original component label, for example
`ncf_linear_BHZ` or `ncf_linear_Z`. Unrotated three-component stack output uses
`ENZ`, for example `ncf_linear_ENZ`. Rotated output uses `RTZ`, for example
`ncf_linear_RTZ`.

The retired BigSAC export/extract path is no longer part of the current tool
surface. Inspect results through SourcePack `unpack`, `plot-rtz-grid`, or the
StepPack tools described in [Tools](TOOLS.md).

## Device And Native Binaries

CUDA architecture is auto-detected during native builds. Override it with
`CUDA_ARCH=sm_XX` when detection fails. The runtime locates packaged binaries
under `fastxc/bin/<platform>` after `make stage-binaries`, or from explicit
paths when configured.

If you use a Conda CUDA package, make sure it contains the full Toolkit,
including `nvcc`, `cufft.h`, and `libcufft.so`.

## Debugging

Use these commands before expensive runs:

```bash
fastxc doctor config.ini
fastxc prepare config.ini
```

Useful workspace files:

- `config.snapshot.ini`: parsed configuration snapshot.
- `inventory.meta.json`: inventory summary.
- `manifest/sac_index.tsv`: parsed SAC inventory.
- `path_plan/allowed_paths.tsv`: final allowed station-pair table.
- `commands/*.sh`: native commands emitted by Python.
- `log/fastxc-*.log` and `progress/*.tsv`: runtime diagnostics.
