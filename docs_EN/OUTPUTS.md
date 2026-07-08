# FastXC Stage Outputs

All FastXC runtime artifacts are written under `[compute].workspace_dir`.
Use a separate workspace for each experiment, dataset, or key parameter set to
avoid accidental overwrites.

## Overview

A typical workspace looks like this:

```text
workspace_dir/
  config.snapshot.ini
  inventory.meta.json
  filter.txt
  commands/
  log/
  progress/
  manifest/
  path_plan/
  stepack/
  ncf/
  sourcepack/
  stack/
  result_ncf/
```

Not every run creates every directory. For example,
`[advance.storage].unpack_enabled = False` skips final SAC export.

## `prepare`

`fastxc prepare config.ini` performs data discovery, pair filtering, and path
planning without launching heavy compute.

| Path | Meaning |
| --- | --- |
| `config.snapshot.ini` | Parsed configuration snapshot for reproducibility. |
| `inventory.meta.json` | Inventory metadata and configuration summary. |
| `manifest/sac_index.tsv` | Global SAC inventory with timestamp, NSL, component, and path fields. |
| `manifest/seisarray*/timestamp_index.tsv` | Timestamp index for one data source. |
| `manifest/seisarray*/by_timestamp/*.tsv` | Timestamp-split SAC manifests used by SAC2SPEC. |
| `path_plan/nsl_catalog.tsv` | GNSL node table with group/network/station/location, coordinates, and components. |
| `path_plan/allowed_paths.tsv` | Station path pairs allowed to enter XC. |
| `path_plan/allowed_path_ids.txt` | Path id whitelist consumed by native XC. |

If `allowed_paths.tsv` has an unexpected row count, first check `sta_list`,
`external_geo_tsv`, `distance_range`, `azimuth_range`, `group_pair_mode`, and
`autocorr_mode`.

The current prepare stage uses `files_groups` keyed by group id. The old
`files_group1/files_group2` compatibility entry points have been removed. If an
external geometry table parses but updates no stations, and filtering yields
zero allowed paths, FastXC fails before SAC2SPEC.

## `run`: SAC2SPEC

SAC2SPEC windows, preprocesses, and converts time-domain SAC data into spectra.

| Path | Meaning |
| --- | --- |
| `filter.txt` | Filter description for this frequency band and preprocessing setup. |
| `stepack/w<worker>.b<batch>.stepack` | Step-major binary spectra written by worker batch. |
| `stepack/w<worker>.b<batch>.tsv` | Timestamp slice index for the same batch. |
| `stepack/_SUCCESS` | SAC2SPEC completion marker. |
| `progress/sac2spec_progress.tsv` | SAC2SPEC progress profile. |

`stepack` is the direct input to XC. The binary payload is stored as
`[step][batch_nslc][freq]`, and the TSV sidecar records timestamp, pack path,
NSLC range, byte range, and pitch metadata.

## `run`: XC

XC performs cross-correlation and writes packed NCF output.

| Path | Meaning |
| --- | --- |
| `ncf/xcpack/*.xcpack` | Cross-correlation binary packs. |
| `ncf/xcpack/*.tsv` | Record index for each pack. |
| `ncf/xcpack/<timestamp>.done` | Completion marker for one timestamp. |
| `ncf/xcpack/_SUCCESS` | XC stage completion marker. |
| `progress/xc_progress.tsv` | Native XC and side-task progress. |

`xcpack` is not the final human browsing format. SourcePack, stack, and rotate
continue to reference these pack records. Native XC no longer writes BigSAC
result files; the final pair semantics use ordinary `.sac` final pair paths and
continue through SourcePack/stack/unpack.

## `run`: SourcePack

SourcePack does not copy the XC payload. It builds indexes over XC pack records
sorted by source, receiver, and component.

| Path | Meaning |
| --- | --- |
| `sourcepack/<timestamp>/sourcepack_index.tsv` | SourcePack index for one timestamp. |
| `sourcepack/<timestamp>/_SUCCESS` | Per-timestamp SourcePack marker. |
| `sourcepack/_SUCCESS` | SourcePack stage marker. |
| `progress/sourcepack_progress.tsv` | SourcePack build progress. |

`sourcepack_index.tsv` contains `record_path`, `record_offset`, and `bytes`,
which point to the real pack records. Immediately after XC, SourcePack is an
index view over `ncf/xcpack`.

## `run`: Stack

Stack reads SourcePack records from multiple timestamps and writes stack
SourcePack output.

| Path | Meaning |
| --- | --- |
| `stack/linearstack_sourcepack/STACK/linearstack.pack` | Linear-stack result pack. |
| `stack/linearstack_sourcepack/STACK/sourcepack_index.tsv` | Linear-stack index. |
| `stack/pws_sourcepack/STACK/` | PWS stack output when the second `stack_flag` bit is `1`. |
| `stack/tfpws_sourcepack/STACK/` | TF-PWS stack output when the third `stack_flag` bit is `1`. |
| `stack/*_sourcepack/manifests/*_inputs.txt` | SourcePack input lists used by that stack. |

After stack, SourcePack is a materialized product: its pack files contain newly
stacked traces. Linear stack currently writes one `linearstack.pack`; PWS and
TF-PWS may write worker shard packs such as `pws.w000.pack`.

## `run`: Rotate

Rotate converts ENZ stacks to RTZ stacks and keeps the SourcePack structure.

| Path | Meaning |
| --- | --- |
| `stack/rtz_linearstack_sourcepack/STACK/rtz_linearstack.pack` | RTZ output for linear stack. |
| `stack/rtz_linearstack_sourcepack/STACK/sourcepack_index.tsv` | RTZ linear-stack index. |
| `stack/rtz_pws_sourcepack/STACK/` | RTZ output for PWS when enabled. |
| `stack/rtz_tfpws_sourcepack/STACK/` | RTZ output for TF-PWS when enabled. |

The example plotting script reads these RTZ stacks:

```bash
python example/plot_rtz_distance_lines.py \
  --workspace workspace_dir \
  --lag-window 20
```

If results have already been exported under `result_ncf/`, use
`fastxc plot-rtz-grid` to plot single-component or RTZ/ENZ 3x3 virtual-source
gathers from the unpacked SAC files.

## `run`: Unpack

`[advance.storage].unpack_enabled = True` is the default. It exports
SourcePack or stack results to traditional SAC files. Set it to `False` when
you only want to keep compact SourcePack products. `unpack_target = ALL`
exports both stack and rotated products when available.

| Path | Meaning |
| --- | --- |
| `result_ncf/` | Fixed export directory: `workspace_dir/result_ncf`. |
| `ncf_<method>_<component_frame>/.../*.SAC` | SAC output directories depending on `unpack_target`, enabled stack methods, and component count. |

Single-component stack keeps the original component label, such as
`ncf_linear_BHZ` or `ncf_linear_Z`. Unrotated three-component stack uses `ENZ`,
such as `ncf_linear_ENZ`. Rotated output uses `RTZ`, such as
`ncf_linear_RTZ`.

## Runtime Audit Files

| Path | Meaning |
| --- | --- |
| `commands/sac2spec.sh` | Native SAC2SPEC command emitted by Python. |
| `commands/xc.sh` | Native XC command emitted by Python. |
| `log/fastxc-*.log` | Python and native backend logs. |
| `progress/*.tsv` | Long-running task progress profiles. |

These files are small and useful for reproduction and debugging.

## What Belongs In Git

Commit source code, example configs, and documentation. Do not commit large
runtime artifacts:

- `config.snapshot.ini`, `inventory.meta.json`, `filter.txt`
- `manifest/`
- `path_plan/`
- `workspace*/`
- `stepack/`
- `ncf/`
- `sourcepack/`
- `stack/`
- `rotate/`
- `result_ncf/`
- `ncf_<method>_<component_frame>/`
- native build products and `bin/`
- generated `plots/`
- private test configs, station tables, or machine-local paths

These paths are ignored in `.gitignore` for the public repository.
