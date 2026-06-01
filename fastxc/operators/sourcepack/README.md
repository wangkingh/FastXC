# FastXC SourcePack

SourcePack is the FastXC-side index layer for native XC `PACK` output.

Input:

```text
ncf/xcpack/*.xcpack
ncf/xcpack/*.tsv
```

Output:

```text
sourcepack/<timestamp>/
  sourcepack_index.tsv
  _SUCCESS
```

FastXC writes one `sourcepack/<timestamp>/` directory for every timestamp that
XC attempted, including timestamps with zero records. Empty days contain a
header-only `sourcepack_index.tsv` plus `_SUCCESS`, so every run has a stable
timestamp layout.

During a normal pipeline run, SourcePack can materialize asynchronously while
native XC is still running. Native XC writes:

```text
ncf/xcpack/<timestamp>.done
```

When FastXC sees that marker, it builds only that timestamp directory. The
regular synchronous builder remains available and produces the same layout.

`sourcepack_index.tsv` groups records by `path_id` and 9-component slot, then
stores the byte offset back into the original native `.xcpack` file:

```text
path_id      = src_id * 10000 + rec_id
component   = E/N/Z or R/T/Z axis rank
slot         = src_axis * 3 + rec_axis
```

That makes each day’s sourcepack layout comparable even when a day is partial
or empty.

This stage does not change native XC math. It only reshapes XC pack output into
a lookup layout that stack and rotation code can scan by virtual source without
copying the binary NCF payload or creating hundreds of thousands of tiny files.

PACK-mode linear stack keeps the same shape:

```text
stack/linearstack_sourcepack/STACK/
  linearstack.pack
  sourcepack_index.tsv
  _SUCCESS
```

Linear stack reads timestamp indexes as sorted streams and merges them by
`path_id + slot`, so it does not need to build a large in-memory list of every
record before stacking.

PACK-mode PWS now uses the same stream contract through native `ncf_pws`:

```text
stack/pws_sourcepack/STACK/
  pws.w000.pack
  pws.w000.tsv
  pws.w001.pack
  pws.w001.tsv
  sourcepack_index.tsv
  _SUCCESS
```

FastXC passes a manifest of `sourcepack_index.tsv` files to `ncf_pws` through
`--sourcepack-list` and receives another SourcePack-shaped output. Native PWS
honors `-G` and `-M`: each virtual GPU worker writes its own shard pack/index,
then the shard indexes are merged into the standard `sourcepack_index.tsv`.

PACK-mode TFPWS uses the same stream contract through native `ncf_tfpws`:

```text
stack/tfpws_sourcepack/STACK/
  tfpws.w000.pack
  tfpws.w000.tsv
  tfpws.w001.pack
  tfpws.w001.tsv
  sourcepack_index.tsv
  _SUCCESS
```

TFPWS preserves the legacy pre-stack math: each group is a sum of daily records,
and the linear seed stack is normalized by the total number of records.

PACK-mode rotation reads whichever stack SourcePack outputs are enabled by
`stack_flag`. For example:

```text
stack/rtz_linearstack_sourcepack/STACK/
  rtz_linearstack.pack
  sourcepack_index.tsv
  _SUCCESS

stack/rtz_pws_sourcepack/STACK/
  rtz_pws.pack
  sourcepack_index.tsv
  _SUCCESS

stack/rtz_tfpws_sourcepack/STACK/
  rtz_tfpws.pack
  sourcepack_index.tsv
  _SUCCESS
```

Each rotated stack keeps SourcePack shape, so it can still be unpacked into
traditional SAC files for delivery or external tools.

The normal FastXC pipeline can run a final export step after stack and rotation
when `[advance.storage].unpack_enabled = True`. This is intentionally the last
step: it creates per-pair SAC files only after the machine-friendly SourcePack
workflow has finished. Pipeline unpack only exports stack or rotated stack
products; raw XC SourcePack is left for explicit debugging tools. Exported files
use one top-level directory per product:

```text
output_dir/ncf_linear_Z/<src_network>.<src_station>/<rec_network>.<rec_station>/
  <src_network>-<rec_network>.<src_station>-<rec_station>.<src_component>-<rec_component>.ncf.SAC

output_dir/ncf_pws_Z/<src_network>.<src_station>/<rec_network>.<rec_station>/
  <src_network>-<rec_network>.<src_station>-<rec_station>.<src_component>-<rec_component>.ncf.SAC

output_dir/ncf_tfpws_RTZ/<src_network>.<src_station>/<rec_network>.<rec_station>/
  <src_network>-<rec_network>.<src_station>-<rec_station>.<src_component>-<rec_component>.ncf.SAC
```

For manual inspection, sampling, or external delivery outside a full FastXC run,
use the standalone tool:

```text
tools/unpack.py -I workspace/stack/rtz_linearstack_sourcepack/STACK -O export/rtz_linearstack
```

The package CLI exposes the same operation:

```text
fastxc unpack -I workspace/sourcepack -O workspace/ncf_unpacked
```

Manual unpack is useful for inspection or external delivery outside a full run;
large exports can still be expensive because they materialize many pair files.
