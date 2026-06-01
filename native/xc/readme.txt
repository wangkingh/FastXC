XC Native Read/Write Model
==========================

This backend computes cross correlations from XCache `.xcspec` shards.
The current production-oriented data path is:

  SAC / spack -> xcache -> native XC -> xcpack -> host finalize


Input
-----

XC accepts either an index of timestamp shards or one shard directly:

  -I <xcspec_index.tsv>
  --timestamp <one.xcspec>

Each `.xcspec` file contains:

  header
  source table
  step-major complex spectrum payload

The payload layout is optimized for reading one time step as a contiguous
block.  For small shards XC may cache the whole payload in host memory;
otherwise workers read step blocks with `pread`.

`-P <allowed_paths.tsv>` is the pair policy table.  It maps source/receiver
station ids to the canonical pair metadata used by the output header and
final path.


GPU Workers and Memory
----------------------

`-G` is a virtual GPU worker list.  Repeated GPU ids are allowed:

  -G 0
  -G 0,1
  -G 0,0

Every entry is one worker.  `-G 0,0` means two workers share physical GPU 0.

`-M` is a per-worker MiB memory budget list matching `-G`.  A value of 0 means
automatic budget.  Automatic budget is:

  0.90 * current_free_gpu_memory / workers_on_same_physical_gpu

The memory budget is used for block/job sizing.  It does not reserve memory by
itself.  Resident workers then allocate their working buffers once and reuse
them across timestamps.

Host RAM is also considered during automatic block sizing.  XC reads
`MemAvailable` from `/proc/meminfo` when possible, uses 80% of that value as
the total host budget, and divides it across the configured GPU workers.  This
budget covers the resident per-worker host staging, index, CC, and lazy-write
buffers; optional xcspec payload caching still falls back to `pread` if its
allocation fails.


Output Modes
------------

`--write-mode append`
  Compatibility mode.  Native XC writes final pair `.bigsac` files directly.
  Existing files are appended to.

`--write-mode aggregate`
  Compatibility mode.  Native XC reads an existing final `.bigsac`, adds the
  new trace, and writes the aggregate result back.

`--write-mode pack`
  Recommended high-throughput mode.  Native XC writes job-local pack files and
  a TSV sidecar instead of creating one small file per pair.


Pack Output
-----------

Pack output is written under:

  <output_root>/xcpack/

Files are grouped by timestamp, block-i/block-j job, worker, and rolling part:

  <timestamp>.i<anchor_block>.j<target_begin_block>.w<worker_id>.p<part>.xcpack
  <timestamp>.i<anchor_block>.j<target_begin_block>.w<worker_id>.p<part>.tsv

Each binary record in `.xcpack` is:

  [SACHEAD][float trace payload]

The TSV sidecar records:

  timestamp
  worker_id
  anchor_block
  target_begin_block
  target_end_block
  block_size
  anchor_begin
  anchor_end
  target_begin
  target_end
  pack_path
  offset
  bytes
  source and receiver metadata
  npts
  dt
  dist
  az
  baz
  final_pair_path

`final_pair_path` is the old `.bigsac` destination path.  Pack mode keeps this
path in the sidecar but does not materialize the file in native code.  A host
finalize step should group TSV rows by `final_pair_path`, seek/read records by
`pack_path + offset + bytes`, and then perform append or aggregate semantics.

Native pack writing buffers records before flushing:

  binary buffer: 32 MiB
  TSV buffer:     4 MiB
  pack part:      4 GiB default rolling limit

Completion markers:

  <output_root>/xcpack/<timestamp>.done
  <output_root>/xcpack/_SUCCESS


Progress and Logs
-----------------

Runtime logs are structured and written to stderr.  The default log level is
INFO.  Use:

  FASTXC_LOG_LEVEL=ERROR|WARN|INFO|DEBUG
  XC_LOG_LEVEL=ERROR|WARN|INFO|DEBUG

Progress, if requested, is written separately with:

  --progress <file>

The progress sidecar is for job monitoring and is not part of the pack data
format.
