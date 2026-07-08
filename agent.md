# FastXC Maintainer Notes

This document is a concise handoff guide for maintainers and coding agents.
User-facing installation and usage instructions belong in the Chinese primary
`README.md` and the auxiliary English `README.en.md`; architecture details
belong in `docs_EN/ARCHITECTURE.md` and `docs_ZH/ARCHITECTURE.md`.

## LLM Quick Start

If you know nothing about this repository, start from the repository root, the
directory that contains `README.md`, `Makefile`, `fastxc/`, `native/`, and
`example/`.

Use this path to verify that the program works exactly as a public user would:

```bash
make install
fastxc doctor

cd example
fastxc doctor config.ini
fastxc prepare config.ini
fastxc run config.ini

python -m pip install matplotlib
python plot_rtz_distance_lines.py \
  --workspace workspace \
  --lag-window 20 \
  --output workspace/plots/rtz_linear_zz_zr_rz_distance_lines.png
```

The important distinction is directory context:

- Build and installation commands run from the repository root.
- The bundled example runs from `example/`, because `example/config.ini` uses
  paths relative to that directory.
- The root-level public smoke INI,
  `configs/test_suite/public_smoke_1hz_kansas.ini`, is for quick maintainer
  checks from the repository root. It points at the same bundled data.

Expected smoke/example shape:

- `fastxc doctor` finds `sac2spec`, `xc_fast`, `ncf_pws`, and `ncf_tfpws`.
- `fastxc prepare` writes `workspace/manifest/` and `workspace/path_plan/`.
- `fastxc run` writes `workspace/stepack/`, `workspace/ncf/`,
  `workspace/sourcepack/`, and `workspace/stack/`.
- The example plot is written under `example/workspace/plots/`.

Do not commit generated `workspace/` contents, native build outputs, local
plots, private INI files, machine hostnames, or absolute local data paths.

## Project Scope

FastXC is a Linux/WSL/HPC-oriented ambient-noise cross-correlation pipeline for
SAC waveforms. Python owns configuration parsing, inventory generation,
workflow orchestration, and machine-friendly binary/index formats. Native
CUDA/C backends own the heavy compute stages.

The supported runtime target is Linux or WSL with NVIDIA CUDA. Native Windows
builds are not supported.

## Main Workflow

The normal user workflow is:

```bash
fastxc prepare config.ini
fastxc run config.ini
```

The current compute path is:

```text
prepare
  -> manifest/sac_index.tsv
  -> path_plan/nsl_catalog.tsv
  -> path_plan/allowed_paths.tsv

sac2spec
  -> stepack/w<worker>.b<batch>.stepack
  -> stepack/w<worker>.b<batch>.tsv

xc
  -> ncf/xcpack/*.xcpack
  -> ncf/xcpack/*.tsv

sourcepack
  -> sourcepack/<timestamp>/sourcepack_index.tsv

stack
  -> stack/linearstack_sourcepack/STACK/sourcepack_index.tsv
  -> stack/pws_sourcepack/STACK/sourcepack_index.tsv
  -> stack/tfpws_sourcepack/STACK/sourcepack_index.tsv

rotate
  -> stack/rtz_linearstack_sourcepack/STACK/sourcepack_index.tsv
  -> stack/rtz_pws_sourcepack/STACK/sourcepack_index.tsv
  -> stack/rtz_tfpws_sourcepack/STACK/sourcepack_index.tsv
```

SAC is the input and optional final export format. Internal stages should prefer
PACK/SourcePack data rather than materializing large numbers of SAC files.

SourcePack has two roles in the current pipeline:

- After XC, `sourcepack/<timestamp>/sourcepack_index.tsv` is mostly an index
  view over records stored in `ncf/xcpack/*.xcpack`.
- After stack or rotate, `stack/*_sourcepack/STACK/` is a materialized product:
  its pack files contain newly computed stack/RTZ traces, and
  `sourcepack_index.tsv` points into those packs.

## Module Boundaries

```text
fastxc/config_parser/  INI loading, schema, and compatibility handling
fastxc/inventory/      SAC discovery, NSL IDs, time filtering, path planning
fastxc/system/         executable discovery, logging, template export
fastxc/stages/         workflow stage orchestration
fastxc/adapters/       native command construction
fastxc/runtime/        subprocess execution, progress files, command review
fastxc/io/             SAC and SourcePack readers/writers
fastxc/operators/      Python-native sourcepack, stacking, rotation
native/                CUDA/C backends
configs/               public smoke-test config
example/               bundled example config, anonymized data, and plotting helper
```

Keep algorithmic changes inside the module that owns the relevant behavior.
Avoid moving responsibilities across layers unless the boundary itself is the
bug.

## Build Commands

From the repository root:

```bash
make install
fastxc doctor
```

`make install` builds all supported native backends, stages the binaries into
the package, and installs the editable `fastxc` command.

Useful lower-level targets:

```bash
make native-full      # all native backends
make stage-binaries   # copy built binaries into fastxc/bin/<platform>
make veryclean-native # remove native build outputs and binaries
```

If CUDA architecture detection fails, pass an explicit architecture:

```bash
make install ARCH=sm_89
```

## Public Smoke Test

The only public test-suite INI is:

```text
configs/test_suite/public_smoke_1hz_kansas.ini
```

It uses only `example/data` and writes under ignored `example/workspace`.
Run:

```bash
fastxc doctor configs/test_suite/public_smoke_1hz_kansas.ini
fastxc prepare configs/test_suite/public_smoke_1hz_kansas.ini
fastxc run configs/test_suite/public_smoke_1hz_kansas.ini
```

Private result matrices, local data paths, large result directories, and
machine-specific performance notes should not be committed to the public
repository.

## Repository Hygiene

Do not commit generated workspaces, cached binary products, native build
outputs, Python caches, or local result plots. The `.gitignore` file is written
to keep the common FastXC outputs out of Git:

```text
config.snapshot.ini
inventory.meta.json
filter.txt
manifest/
path_plan/
workspace*/
stepack/
ncf/
sourcepack/
stack/
rotate/
result_ncf/
*.stepack
*.xcpack
*.pack
```

`example/data` is intentionally kept because it makes
the public smoke test self-contained. `example/workspace/.gitkeep` is also kept
so the generated example workspace location is visible while its contents stay
ignored.

## Development Rules

- Prefer existing config schema and local helpers over adding parallel parsing
  conventions.
- Keep native command generation in `fastxc/adapters/`; keep process execution
  in `fastxc/runtime/`.
- Keep SourcePack indexes sorted and stable where downstream streaming stack
  code depends on merge order.
- Preserve `PACK`/SourcePack as the mainline output path for XC, stacking, PWS,
  TF-PWS, and rotation.
- Treat `SAC`/`sacio` code in native backends as inherited external-style
  support code unless a task explicitly asks to change it.
- Add compatibility only when it protects a real existing config or workflow.
- Keep public docs free of private hostnames, absolute local data paths, and
  large local result references.
- Do not rewrite historical design notes in `CHANGELOG.md` or `changelog/`
  just because they mention retired paths. Update them only when a current
  user-facing instruction is wrong.

## Quick Orientation Checklist

When returning to the project:

```bash
rg -n "TODO|FIXME|sourcepack|stack_flag|gpu_memory_mib" fastxc native configs
make -C native print-config
fastxc doctor configs/test_suite/public_smoke_1hz_kansas.ini
```

For architecture context, read `docs_EN/ARCHITECTURE.md`.
