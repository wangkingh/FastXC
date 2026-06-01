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
  -> spack_by_timestamp/<timestamp>/*.spack + *.tsv

xcache
  -> xcache/<timestamp>.xcspec
  -> xcache/xcspec_index.tsv

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
  FastXC-owned data transformations: xcache, sourcepack, linear stack, rotate

io/
  Binary/index format readers and writers

runtime/
  Subprocess execution, progress polling, logging, and side-task summaries
```

## Data Format Roles

- `spack`: high-throughput SAC2SPEC output. It stores raw SEGSPEC records in
  larger append-only blocks plus TSV sidecars.
- `xcache`: native XC input. It is a timestamp-level, self-describing binary
  shard with `payload[step][file][freq]` layout.
- `xcpack`: native XC output packs, written under GPU memory and job sharding.
- `sourcepack`: Source/receiver/component sorted indexes over pack records.
  Stack and rotation consume SourcePack rather than scattered SAC files.

The normal pipeline keeps these binary/index formats until the final explicit
`unpack` step. Traditional SAC files are an export product, not an internal
working format.
