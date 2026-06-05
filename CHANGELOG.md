# Changelog

All notable project changes are recorded here. Dates use `YYYY-MM-DD`.

## [Unreleased] - 2026-06-04

### Added

- Added `docs/` as the stable public documentation home:
  - `docs/ARCHITECTURE.md`
  - `docs/RESULTS.md`
- Kept private validation matrices and large result artifacts outside the public
  repository.

### Changed

- Retired the Python XCache/SPACK mainline. SAC2SPEC now writes `stepack/`
  worker-batch files, native XC reads that workspace directly, and SourcePack
  receives timestamp expectations from the inventory manifest.
- Removed the native backup directories and the `xc_only_read` comparison
  benchmark from the maintained build tree.
- Updated the English and Chinese READMEs with direct links to public
  architecture and result-artifact policy documentation.
- Clarified the current pack/index result model in README, Outputs, and agent
  notes: XC-side SourcePack is an index over `ncf/xcpack`, while stack/rotate
  SourcePack outputs are materialized products.
- Reworked the Chinese README into a public user guide and moved migration
  rationale, cleanup notes, and architecture-change context into changelog/docs
  instead of the top-level usage page.
- Reduced public test-suite configs to one self-contained smoke test based on
  bundled `example/data`.
- Removed private/local testing notes and machine-specific result paths from
  the public documentation surface.
- Simplified native build documentation and Makefile targets so `native` and
  `native-full` build all CUDA backends, including TF-PWS, in one pass.
- Cleaned the user-facing `fastxc unpack` help text so it describes the current
  SourcePack-to-SAC export path instead of the retired BigSAC wording.
- Grouped the bundled example into `example/`, with
  data, INI, generated workspace, and optional plot outputs under one tree.
- Kept engineering switches under `advance.compute` and `advance.storage`,
  while removing retired XC write-mode knobs from the public config surface.
- Kept `sac_len` in `[compute]` because it describes the input window
  geometry, and removed duplicate/retired write-mode fields.
- Removed the obsolete `stack` and `rotate` executable slots from current INI
  templates and configs; stacking and rotation are Python-native FastXC stages.
- Consolidated the public INI layout so `[compute]` carries the core compute
  fields (`max_lag`, `stack_flag`, `workspace_dir`), while advanced path
  filtering, SourcePack, and PWS/TF-PWS tuning live under
  `advance.compute`.
- Renamed the phase-only and pre-stack controls to
  `advance.compute.phase_only` and `advance.compute.pre_stack_size`.
- Moved final SAC export settings to `advance.storage`.
- Further reduced public advanced settings: SourcePack construction and sorting
  are always enabled, all async polling uses one `advance.compute.async_poll_sec`,
  and final SAC export is always written to `workspace_dir/result_ncf`.
- Simplified external station geometry matching so optional
  `external_geo_tsv` rows match by `network + station + location` or by
  `station`; the logical FastXC group no longer participates in geometry
  lookup.
- Removed old INI compatibility for retired `[arrayN]`, `[preprocess]`,
  `[xcorr]`, `[stack]`, `[storage]`, `[unpack]`, `advance.xcache`,
  `advance.sourcepack`, and `advance.stack` layouts.
- Added `shift_len = AUTO`, which resolves to `win_len`.

### Notes

- Entries below this section include historical architecture transitions. Some
  earlier notes mention retired MiniXC, BigSAC, APPEND, or AGGREGATE branches
  that are no longer part of the current PACK/SourcePack mainline.

## 2026-05-30

### Changed

- Consolidated the production data path around:
  - timestamp-local SAC2SPEC spack output
  - async xcache materialization
  - native XC pack output
  - async SourcePack indexing
  - SourcePack-based linear/PWS/TF-PWS/rotation
  - explicit final unpack
- Split FastXC orchestration into stages, adapters, operators, runtime, system,
  inventory, and io packages.
- Kept PACK/SourcePack as the formal mainline and retired APPEND/AGGREGATE and
  BigSAC branches from the normal workflow.

### Validated

- Locked the private test-suite reference run outside the public repository.
- Recorded the TLWF 01-line one-month all-stack reference:
  - prepare: 81.51 s
  - run: 266.21 s
  - SAC2SPEC: 68.34 s
  - XC: 50.04 s
  - PWS: 30.05 s
  - TFPWS: 40.02 s
- Verified AUTO vs 4096 MiB hash consistency for final XC and stack/rotate
  binary outputs.

## 2026-05-22

### Added

- Added package-level console command:
  - `fastxc`
- Added the two-stage workflow:
  - `fastxc prepare config.ini`
  - `fastxc run config.ini`
- Added MiniXC as a second compute engine using the same prepared workspace.
- Added an experimental MiniXC CMake build path for native Windows/MSVC/CUDA
  compatibility work. The CMake build can produce CUDA or CPU-only MiniXC
  executables and writes to `bin` by default.
- Added MiniXC CMake presets for CPU, `sm_80`, and `sm_89` builds.
- Added `seisarrayN` configuration sections with shared logical group IDs.
- Added `[time_filter]` as a top-level time selection section.
- Added `group_pair_mode = intra|inter|all` for path generation.
- Added reusable workspace outputs:
  - `time_stamps.tsv`
  - `by_timestamp/*.tsv`
  - `path_plan/allowed_paths.tsv`
  - `path_plan/gnsl_nodes.tsv`
  - `workspace.meta.json`
- Added path metadata to allowed paths, including distance, azimuth, and
  back-azimuth.
- Added automatic CUDA architecture detection with `ARCH=auto`.
- Added top-level build targets for native backends, binary staging, and editable
  Python installation.
- Added MATLAB NoiseCorr reference folders for algorithm comparison:
  - `NoiseCorr-2016Jul-v4.2`
  - `noise_corr_multi_components`
- Added a stabilization note for the current frozen baseline:
  - `docs/VERSION_FREEZE_2026-05-22.md`
- Added a source-shard MPI improvement plan:
  - `docs/MPI_SOURCE_SHARD_PLAN.md`
- Added a fixed anonymized 1 Hz three-component example dataset and config:
  - `example/data/`
  - `example/config.ini`
- Added `docs/MINIXC_WINDOWS.md` describing native Windows support as a MiniXC
  subset while keeping Linux/WSL as the recommended production target.

### Changed

- Replaced the old one-shot Python-centered flow with reusable `xcprepare` and
  compute stages.
- Simplified user-facing configuration around:
  - `[seisarrayN]`
  - `[time_filter]`
  - `[geometry]`
  - `[executables]`
  - `[preprocess]`
  - `[xcorr]`
  - `[stack]`
  - `[device]`
  - `[storage]`
  - `[debug]`
- Default preprocessing now resolves `normalize = AUTO` to:
  - `RUN-ABS` for one band
  - `RUN-ABS-MF` for multiple bands
- Default spectral whitening mode is `AFTER`.
- Default stack behavior is linear-only with `stack_flag = 100`.
- XC native execution now owns timestamp iteration and GPU worker scheduling.
- SAC2SPEC now consumes timestamp manifests directly instead of a separate
  generated SAC2SPEC path list.
- The old XC list-generation concept was replaced by timestamp-based SEGSPEC
  lists for the native XC stage.
- Linear stacking and rotation are Python implementations designed for easy
  parallel execution.
- PWS and TF-PWS are optional stack steps instead of default pipeline steps.
- `cmd_deployer`/command execution responsibilities were simplified around
  one native command per major backend.
- Native progress reporting was moved toward sidecar/progress-aware execution.
- README and Chinese README were rewritten to reflect the current architecture.
- README now documents the bundled three-component example commands.
- Cleaned intermediate local and G-drive test outputs into waste directories
  instead of deleting them.
- Simplified the public Python CLI to `init`, `doctor`, `prepare`, and `run`.
  Development helpers were removed from the public command surface.

### Removed

- Removed user-facing reliance on:
  - Python-side multi-GPU task tables
  - separate generated SAC2SPEC input lists
  - BigSAC concatenation as a mandatory Python stage
  - `dry_run`
  - `log_file_path`
  - `gpu_task_num`
  - `gpu_mem_info`
  - `source_info_file`
- Removed outdated README content that referenced old directory names and old
  array configuration fields.

### Notes

- The default recommended entry point is now `fastxc`.
- `fastxc prepare` replaces the earlier public `fastxc xcprepare` command name.
- `fastxc run config.ini` defaults to the staged FastXC engine. Use
  `fastxc run config.ini --engine mini` for MiniXC.
- The public CLI no longer writes a local `debug.py` replay launcher during
  normal runs.
- WSL/Linux is the primary native build target. Windows support depends on
  staging compatible native binaries for the target platform.
- The README now documents the current Windows CUDA portability boundary:
  Python/workspace logic is mostly portable, MiniXC is the most promising native
  Windows target, and the older FastXC CUDA backends still need build-system and
  POSIX-thread adaptation.
- Current freeze validation anchors the cleaned build around the final
  `0101 -> 0103` three-component case. WSL FastXC and WSL MiniXC match at
  numerical noise level; WSL/Windows MiniXC differences are documented as
  CUDA/cuFFT-version drift rather than a known memory-layout bug.

## 2025-05-07

### Added

- Added build-system guard for top-level `MODE`; allowed values are `par` and
  `seq`.
- Added distinct debug and release NVCC flag sets.

### Changed

- Unified formatting and recursive CUDA build goals.
- Updated binary lookup paths to the staged `bin/` layout.

### Fixed

- Fixed several buffer-size and uninitialized-pointer warnings in native code.

### Removed

- Removed redundant NVCC flag combinations in debug builds.

## 2025-05-03

### Fixed

- Fixed a queue deadlock in dispatcher logic.
- Improved round-robin task dispatch.
- Added globally unique `-Q` IDs for worker commands.

## 2025-05-02

### Changed

- Refactored the core codebase and rewrote the changelog structure.
