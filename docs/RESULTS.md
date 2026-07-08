# Results

Generated FastXC result directories are intentionally not included in the public
repository.

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
`example/plot_rtz_distance_lines.py`. The packaged CLI also provides
`fastxc plot-rtz-grid` for unpacked single-component or 3x3 result SAC files
and `fastxc extract-stepack --plot` for StepPack spectrum inspection.
Project-local inspection scripts, generated plots, and machine-specific result
folders should stay outside the public repository.
