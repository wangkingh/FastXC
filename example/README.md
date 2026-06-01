# FastXC 1 Hz Kansas Example

This directory contains the bundled public SAC subset, example INI, generated
workspace location, and optional plot output location for a small FastXC smoke
run.

After `make install` and `fastxc doctor` succeed from the repository root, run
this example from inside the `example/` directory:

```bash
cd example
fastxc doctor config.ini
fastxc prepare config.ini
fastxc run config.ini
```

Optional RTZ line plot, still from inside `example/`:

```bash
python -m pip install matplotlib
```

```bash
python plot_rtz_distance_lines.py \
  --workspace workspace \
  --lag-window 20 \
  --output workspace/plots/rtz_linear_zz_zr_rz_distance_lines.png
```

## Dataset

Processing applied:

- low-pass filtered at `0.45 Hz` before downsampling;
- downsampled from `10 Hz` to `1 Hz`;
- station names anonymized to four-character letter/digit names;
- timestamps moved to `2022.137` through `2022.143`;
- station coordinates translated as a group to Kansas;
- SAC headers updated for `delta`, `npts`, `b/e`, extrema, station name,
  component name, time fields, network/location, and station coordinates.

Subset:

- stations: `A7K2`, `B4Q8`, `C9M3`, `D2X6`
- network/location: `KS/00`
- year: `2022`
- Julian days: `137` to `143`
- components: `E`, `N`, `Z`
- files: `84` SAC files
- size: about `28 MB`
- optional station coordinate table: `station_geo.tsv`

Recommended config: `config.ini`

The bundled config reads station coordinates from the SAC headers
(`external_geo_tsv = NONE`). To test external geometry loading, set
`[geometry].external_geo_tsv = data/station_geo.tsv`; the table uses
`station`, `latitude`, and `longitude`, which are accepted aliases for
`station`, `lat`, and `lon`.

The example config uses two period bands:

- `2.5-5 s`, written as `0.2/0.4 Hz`;
- `5-10 s`, written as `0.1/0.2 Hz`.

Layout:

```text
example/
  config.ini
  README.md
  data/
    A7K2/2022/*.sac
    B4Q8/2022/*.sac
    C9M3/2022/*.sac
    D2X6/2022/*.sac
    station_geo.tsv
  workspace/
    plots/
```

`workspace/` is generated output and is ignored by Git.
