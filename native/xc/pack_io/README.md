# XC Pack Output

`pack_io` implements the native side of `--write-mode pack`.

Native responsibility:

- write correlation records sequentially into `.xcpack` files;
- group pack files by timestamp, block-i/block-j job, worker, and rolling part;
- write one TSV row per record with block job coordinates, `pack_path`, `offset`, `bytes`, station metadata, and `final_pair_path`;
- buffer binary records before flushing to `.xcpack` files, and buffer TSV rows before flushing sidecars;
- roll pack files by part before a record would exceed the configured pack size;
- emit timestamp-level `.done` markers and a run-level `_SUCCESS` marker.

Pack file names:

```text
<timestamp>.i<anchor_block>.j<target_begin_block>.w<worker_id>.p<part>.xcpack
<timestamp>.i<anchor_block>.j<target_begin_block>.w<worker_id>.p<part>.tsv
```

Each binary record is:

```text
[SACHEAD][float trace payload]
```

The TSV sidecar keeps `final_pair_path`, which is the final `.bigsac` path that
append/aggregate mode would have written directly.  The final pair-level append
or aggregate materialization is intentionally left to a host-side finalize step.
