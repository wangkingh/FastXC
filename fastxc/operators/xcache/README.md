# FastXC XCache

`xcache` repacks SAC2SPEC timestamp output into one step-major `.xcspec`
shard per timestamp.

## Input

The formal input is the SAC2SPEC workspace root with timestamp-local spack
output:

```text
<workspace>/spack_by_timestamp/<timestamp>/*.spack
<workspace>/spack_by_timestamp/<timestamp>/*.tsv
<workspace>/spack_by_timestamp/<timestamp>/_SUCCESS
<workspace>/spack_by_timestamp/_SUCCESS
```

FastXC can build xcache synchronously after SAC2SPEC, or asynchronously while
SAC2SPEC is still running. In async mode, it watches timestamp `_SUCCESS`
markers and consumes one completed timestamp at a time.

If `[xcache].cleanup_timestamp_spack = True`, a separate async spack sweeper
deletes `spack_by_timestamp/<timestamp>` only after xcache writes a cleanup
marker for that timestamp. The xcache builder creates data; the sweeper owns
deletion.

XCache normalizes timestamp text to:

```text
YYYYMMDDTHH:MM
```

For example, `2023.189.0809` and `2023-07-08T08:09` both become
`20230708T08:09` in the `.xcspec` header, file name, and index.

## Output

XCache writes flat files under:

```text
<workspace>/xcache/
  xcspec_index.tsv
  <timestamp>.xcspec
  <timestamp>.xcspec.json
```

By default, `windows_per_xcache = AUTO` writes one `.xcspec` containing all
windows from the source SEGSPEC files. If `windows_per_xcache` is set to a
positive integer, one timestamp is split into multiple shards named like:

```text
<timestamp>.w000000-000007.xcspec
```

Each shard still has the same binary layout. Its header `nstep` is the number
of windows in that shard, so native XC reads it exactly like a normal `.xcspec`.

## Binary Layout

One `.xcspec` file contains one timestamp:

```text
XCSPECHeader              256 bytes
XCSPECSourceEntry[]       file_count * 128 bytes
padding                   zero bytes to 4096-byte payload alignment
payload                   complex64[nstep][file_count][nspec]
```

The payload is step-major:

```text
payload[step][file_index][freq]
```

Each complex value is native `complex64_interleaved`:

```text
float32 real
float32 imag
```

The offset for one value is:

```text
payload_offset + ((step * file_count + file_index) * nspec + freq) * 8
```

The offset for one whole step is:

```text
offset = payload_offset + step * file_count * nspec * 8
bytes  = file_count * nspec * 8
```

## Header

All integer/float fields are little-endian. The fixed 256-byte header stores:

```text
offset  type     field
0       char[8]  magic = FXCXSPEC
8       uint32   version = 1
12      uint32   endian_tag = 0x01020304
16      uint32   header_size = 256
20      uint32   source_entry_size = 128
24      uint64   source_table_offset
32      uint32   source_count
36      uint32   file_count
40      uint64   payload_offset
48      uint32   layout = 1, step_file_freq
52      uint32   dtype = 1, complex64_interleaved
56      uint32   string_encoding = 1, ascii_nul_padded
64      char[64] timestamp
128     uint32   nstep
132     uint32   nspec
136     uint32   nfft
144     float32  dt
148     float32  df
152     uint64   step_bytes
160     uint64   payload_bytes
168     uint64   manifest_hash_u64
176     uint64   source_table_bytes
```

`step_bytes` is the whole contiguous block for one timestamp step:

```text
file_count * nspec * 8
```

`manifest_hash_u64` is the first 8 bytes of the SHA256 manifest hash stored as
a little-endian uint64. The full 64-character hex hash is stored in JSON and
`xcspec_index.tsv`.

## Source Entry

Each 128-byte source entry is fixed width:

```text
offset  type      field
0       uint32    file_index
4       uint32    nsl_id
8       float32   stla
12      float32   stlo
16      char[16]  network
32      char[32]  station
64      char[16]  location
80      char[16]  component
96      byte[32]  reserved
```

`file_index` is dense and starts at zero. `nsl_id` does not need to be dense.
Native code that still uses the old `gnsl_id` name should treat this field as
the same station identity.

## XCSPEC Index

`xcspec_index.tsv` contains one row per shard:

```text
timestamp
xcspec_path
manifest_path
file_count
nstep
nspec
nfft
dt
df
payload_offset
step_bytes
manifest_hash
window_start
window_count
source_nstep
```

## JSON

The `.json` manifest is for inspection and debugging. Native XC should not need
it for computation.

The builder reuses an existing shard when the JSON `manifest_hash` matches the
current source metadata and the `.xcspec` size is exactly what the header/layout
requires. Writes use `.tmp` files followed by rename to avoid half-written
artifacts.

Source `.SEGSPEC` files are left in place. Delete them outside this builder
after native XC no longer needs the original timestamp speclists.
