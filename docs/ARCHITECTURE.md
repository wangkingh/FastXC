# FastXC Architecture

FastXC is now organized around explicit intermediate data formats instead of
large numbers of human-readable temporary files.

## Main Flow

```text
prepare
  -> manifest/sac_index.tsv
  -> path_plan/nsl_catalog.tsv
  -> path_plan/allowed_paths.tsv

sac2spec
  -> stepack/w<worker>.b<batch>.stepack + .tsv

xc
  -> ncf/xcpack/*.xcpack + *.tsv

sourcepack
  -> sourcepack/<timestamp>/sourcepack_index.tsv

stack
  -> stack/<method>_sourcepack/STACK/sourcepack_index.tsv

rotate
  -> stack/rtz_<method>_sourcepack/STACK/sourcepack_index.tsv

unpack
  -> ncf_<method>_<component_frame>/**/*.SAC
```

## Responsibility Boundaries

```text
inventory/
  SAC discovery, NSL IDs, station metadata, path IDs, allowed path tables

adapters/
  Python-to-native command construction

stages/
  High-level orchestration of one pipeline stage at a time

operators/
  FastXC-owned data transformations: sourcepack, linear stack, rotate

io/
  Binary/index format readers and writers

runtime/
  Subprocess execution, progress polling, logging, and side-task summaries
```

## Data Format Roles

- `stepack`: high-throughput SAC2SPEC output and native XC input. Each worker
  batch stores a header, NSLC table, and pitched `payload[step][nslc][freq]`
  spectra, with TSV sidecars describing virtual timestamp slices.
- `xcpack`: native XC output packs, written under GPU memory and job sharding.
- `sourcepack`: Source/receiver/component sorted indexes over pack records.
  Immediately after XC it is primarily an index view over `ncf/xcpack`
  records. After stack or rotate it is a materialized product whose own pack
  files contain the newly computed traces.

The normal pipeline keeps these binary/index formats until the final explicit
`unpack` step. Traditional SAC files are an export product, not an internal
working format.
