# Results and Benchmarks

Large benchmark outputs, retained local workspaces, and private test matrices
are intentionally not included in the public repository.

FastXC result directories can contain machine-local paths, large binary packs,
and data-derived artifacts. Keep those outside Git and publish only sanitized
summaries when needed.

For public validation, use the bundled smoke configuration:

```bash
fastxc doctor configs/test_suite/public_smoke_1hz_kansas.ini
fastxc prepare configs/test_suite/public_smoke_1hz_kansas.ini
fastxc run configs/test_suite/public_smoke_1hz_kansas.ini
```

That smoke configuration reads the bundled example data under `example/data`
and writes the ignored workspace under `example/workspace`.

The public plotting helper lives with the bundled example:
`example/plot_rtz_distance_lines.py`. Project-local benchmark and Hi-net
inspection scripts are intentionally kept out of the public repository.
