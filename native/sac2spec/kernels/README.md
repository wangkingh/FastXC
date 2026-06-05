# CUDA Kernel Layout

Low-level CUDA kernels live here. Algorithm-level entry points stay in `core/`.

- `real_matrix.*`: real-valued 2-D matrix kernels for time-domain samples,
  weights, smoothing primitives, division, clamping, and simple accumulation.
- `complex_matrix.*`: complex-valued 2-D matrix kernels for spectrum cleanup,
  amplitude extraction, spectral scaling, complex division, and normalization.
- `taper.*`: time-domain and spectrum-domain taper kernels.
- `rdcrtr.*`: kernels specific to preprocessing reductions, demean,
  and detrend.

Keep kernels here small and mechanical. Compose them into named processing
steps in `core/` so the main workflow and ABI share the same behavior.

## Matrix Contract Conventions

All 2-D device matrices are row-major logical matrices addressed as
`matrix[row * pitch + col]`. `pitch` is measured in elements, not bytes.
`width` and `height` describe the active logical rectangle. For R2C spectra,
the physical pitch is often the real-domain FFT length while the active
frequency width is `nfft / 2 + 1`.

Kernel parameters should be read with this naming convention:

- `d_*`: CUDA device memory.
- `pitch`, `dpitch`, `spitch`, `full_pitch`, `shared_pitch`: matrix row stride.
- `d_data`, `d_segspec`, `d_spectrum`: in-place input/output unless stated.
- `d_tmp`, `d_tmp_weight`, `d_weight`: caller-owned workspace in `core/`.

## Kernel Matrix Contracts

| Kernel | Inputs | Outputs | Workspace / notes |
| --- | --- | --- | --- |
| `abs2DKernel` | `d_data[height][pitch]` | `d_data[:, :width] = abs(d_data)` | In-place, no extra workspace. |
| `clampmin2DKernel` | `d_data`, `minval` | Non-finite or `< minval` entries become `minval` | In-place, no extra workspace. |
| `isnan2DKernel` | `d_data` | Non-finite entries become `0` | In-place, no extra workspace. |
| `cisnan2DKernel` | complex `d_data` | Complex entries with non-finite real or imag become `0+0i` | In-place, no extra workspace. |
| `div2DKernel` | `d_data`, `d_divisor` | `d_data /= d_divisor`, invalid divisors produce `0` | `d_divisor` is an input matrix, commonly a generated weight workspace. |
| `cdiv2DKernel` | complex `d_data`, float `d_divisor` | Complex `d_data /= d_divisor`, invalid divisors produce `0+0i` | `d_divisor` is an input matrix, commonly a generated weight workspace. |
| `spectralOnebit2DKernel` | complex `d_data`, `minval` | Complex phase-only spectrum, amplitude normalized to 1 or zeroed | In-place, no extra workspace. |
| `sum2DKernel` | `d_data_in`, current `d_data_out` | `d_data_out += d_data_in` | `d_data_out` is an accumulator and must be initialized by caller. |
| `expandSharedWeight2DKernel` | `d_weight_shared[height/num_ch][shared_pitch]` | `d_weight_full[height][full_pitch]` | Expands per-frame shared weights to per-channel rows. |
| `cutmax2DKernel` | `d_data`, `maxval` | Clamps values to `[-maxval, maxval]` | In-place, no extra workspace. |
| `amp2DKernel` | complex `d_data[height][spitch]` | float `d_amp[height][dpitch]` amplitudes | Also writes `d_amp[row][width]` from DC imaginary part when `col == 0`; callers must provide `dpitch > width` or remove this legacy side write. |
| `filterKernel` | complex `d_spectrum`, `d_response[width]` | `d_spectrum *= d_response[col]` | In-place spectrum scaling, no extra workspace. |
| `onebit2DKernel` | `d_data` | Sign-only time series `-1/0/1` | In-place, no extra workspace. |
| `FwdNormalize2DKernel` | complex `d_segspec`, `dt` | `d_segspec *= dt` | In-place, no extra workspace. |
| `InvNormalize2DKernel` | `d_segdata`, `dt` | `d_segdata *= 1 / (width * dt)` | In-place, no extra workspace. |
| `smoothRowsRollingKernel` | `d_tmp` input matrix | `d_out` smoothed matrix | One thread per row. `d_out` and `d_tmp` must not alias. O(width) rolling window launched by `core/smoothing.cu`. |
| `sumSingleBlock2DKernel` | `d_data[height][spitch]` | `d_sum[height][dpitch]`, one sum per row | Uses dynamic shared memory `blockDim.x * blockDim.y * sizeof(double)`; launch with one grid block in X. |
| `isumSingleBlock2DKernel` | `d_data[height][spitch]` | `d_isum[height][dpitch]`, sum of `i * x[i]` per row | Uses dynamic shared memory; launch with one grid block in X. |
| `rdc2DKernel` | `d_sum[row]` | Demeaned `d_data` | In-place; current implementation reads `d_sum[row]`, so the effective sum pitch is 1. |
| `rtr2DKernel` | `d_sum[row]`, `d_isum[row]` | Detrended `d_data` | In-place; current implementation reads sum workspaces as pitch-1 vectors. |
| `specTaper2DKernel` | complex `d_segspec`, taper indices | Tapered / zeroed spectrum | In-place, no extra workspace. |
| `timetaper2DKernel` | `d_data`, `taper_size` | Tapered time series | In-place, no extra workspace. |

## Core Workspace Ownership

`core/preprocess.cu` owns reduction workspaces:

- `d_sum[proc_cnt]`: row sums for demean/detrend.
- `d_isum[proc_cnt]`: row index-weighted sums for detrend.

`core/normalization.cu` and `core/spectrum.cu` share whitening/runabs
workspaces:

- `d_tmp_weight[frame_count][pitch]`: per-channel temporary amplitude/absolute
  value and smoothed weight buffer.
- `d_tmp[frame_count][pitch]`: accumulated shared weight across `num_ch`
  channels.
- `d_weight[frame_count * num_ch][pitch]`: expanded per-channel divisor used
  by `div2DKernel` or `cdiv2DKernel`.

`runabs_mf` additionally uses:

- `d_padded_sacdata`: padded time-domain filter workspace.
- `d_padded_spectrum`: per-filter complex spectrum workspace.
- `d_base_padded_spectrum`: reusable broad input spectrum.
- `d_filtered_sacdata`: one filtered band before runabs.
- `d_total_sacdata`: accumulated multi-frequency normalized output.
