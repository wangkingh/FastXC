# CUDA Kernel Layout

Low-level CUDA kernels live here. Algorithm-level entry points stay in `core/`.

- `misc.*`: generic 2-D array operations such as sanitizing, amplitude,
  division, clamping, smoothing, spectral scaling, and simple reductions.
- `taper.*`: time-domain and spectrum-domain taper kernels.
- `rdcrtr.*`: kernels specific to preprocessing reductions, demean,
  and detrend.

Keep kernels here small and mechanical. Compose them into named processing
steps in `core/` so the main workflow and ABI share the same behavior.
