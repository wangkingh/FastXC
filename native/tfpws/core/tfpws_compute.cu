#include "tfpws_compute.hpp"
#include "tfpws_workspace.hpp"

#include <cstddef>
#include <cstdlib>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "cuda.stransform.cuh"
#include "cuda.util.cuh"
#include "logger.h"
#include "tfpws_sac.hpp"
#include "tfpws_schedule.hpp"

#define CUDA_KERNEL_CHECK() CUDACHECK(cudaGetLastError())

int compute_tfpws_from_prestack(const char *label,
                                SACHEAD ncf_hd,
                                float *prestack_data,
                                float *linear_stack,
                                float *group_trace_weights,
                                unsigned num_segments,
                                unsigned ngroups,
                                unsigned nsamples,
                                const ARGUTYPE *argument,
                                std::size_t worker_index,
                                int device_id,
                                TfpwsDeviceWorkspace *workspace,
                                SACHEAD *out_header,
                                float **out_data)
{
    CUDACHECK(cudaSetDevice(device_id));

    const std::size_t nfreq = nsamples / 2 + 1;
    const std::size_t prestack_count = (std::size_t)ngroups * nsamples;
    const long double host_input_bytes =
        estimate_tfpws_host_input_bytes(ngroups, nsamples);

    if (!workspace || !workspace->initialized)
    {
        LOG_ERROR("missing_tfpws_device_workspace",
                  "worker_index=%zu gpu_id=%d label=\"%s\"",
                  worker_index,
                  device_id,
                  label ? label : "");
        free(linear_stack);
        free(prestack_data);
        free(group_trace_weights);
        return 1;
    }
    if (nsamples != workspace->plan.nsamples ||
        ngroups > workspace->plan.max_ngroups ||
        nfreq != workspace->plan.nfreq ||
        argument->band_limited != workspace->plan.band_limited)
    {
        LOG_ERROR("tfpws_workspace_shape_mismatch",
                  "worker_index=%zu gpu_id=%d label=\"%s\" groups=%u max_groups=%u samples=%u workspace_samples=%u band_limited=%d workspace_band_limited=%d",
                  worker_index,
                  device_id,
                  label ? label : "",
                  ngroups,
                  workspace->plan.max_ngroups,
                  nsamples,
                  workspace->plan.nsamples,
                  argument->band_limited,
                  workspace->plan.band_limited);
        free(linear_stack);
        free(prestack_data);
        free(group_trace_weights);
        return 1;
    }

    const std::size_t freq_chunk_size = workspace->plan.freq_chunk_size;
    const std::size_t num_freq_chunks = workspace->plan.num_freq_chunks;

    LOG_INFO("tfpws_compute_start",
             "worker_index=%zu gpu_id=%d label=\"%s\"",
             worker_index,
             device_id,
             label ? label : "");
    LOG_INFO("tfpws_chunk_plan",
             "worker_index=%zu gpu_id=%d segments=%u groups=%u max_groups=%u samples=%u freq_bins=%zu fixed_freq_chunk=%zu chunks=%zu host_input_gib=%.3f resident_data_gib=%.3f cufft_workspace_gib=%.3f runtime_reserve_gib=%.3f planned_peak_gib=%.3f",
             worker_index,
             device_id,
             num_segments,
             ngroups,
             workspace->plan.max_ngroups,
             nsamples,
             nfreq,
             freq_chunk_size,
             num_freq_chunks,
             bytes_to_gib(host_input_bytes),
             bytes_to_gib(workspace->plan.resident_data_bytes),
             bytes_to_gib((long double)workspace->plan.cufft_workspace_bytes),
             bytes_to_gib((long double)workspace->plan.runtime_reserve_bytes),
             bytes_to_gib(workspace->plan.planned_peak_bytes));

    if (!prestack_data || !linear_stack || !group_trace_weights)
    {
        LOG_ERROR("missing_tfpws_host_input",
                  "label=\"%s\"",
                  label ? label : "");
        free(linear_stack);
        free(prestack_data);
        free(group_trace_weights);
        return 1;
    }

    configure_tfpws_output_header(&ncf_hd, nsamples, num_segments);

    const double delta = ncf_hd.delta;
    const double df_hz = (delta > 0.0) ? (1.0 / ((double)nsamples * delta)) : 0.0;
    const double nyquist_hz = (delta > 0.0) ? (0.5 / delta) : 0.0;
    int band_limited = argument->band_limited;
    double band_fmin = argument->band_fmin;
    double band_fmax = argument->band_fmax;
    double band_taper_hz = argument->band_taper_hz;

    if (band_limited)
    {
        if (delta <= 0.0)
        {
            LOG_ERROR("invalid_band_delta",
                      "label=\"%s\" delta=%.9g",
                      label ? label : "",
                      delta);
            free(linear_stack);
            free(prestack_data);
            free(group_trace_weights);
            return 1;
        }
        if (band_fmin >= nyquist_hz)
        {
            LOG_ERROR("invalid_band_lower_bound",
                      "fmin_hz=%.6g nyquist_hz=%.6g",
                      band_fmin,
                      nyquist_hz);
            free(linear_stack);
            free(prestack_data);
            free(group_trace_weights);
            return 1;
        }
        if (band_fmax > nyquist_hz)
        {
            LOG_WARN("band_upper_clamped",
                     "fmax_hz=%.6g nyquist_hz=%.6g",
                     band_fmax,
                     nyquist_hz);
            band_fmax = nyquist_hz;
        }
        if (band_taper_hz < 0.0)
            band_taper_hz = 0.05 * (band_fmax - band_fmin);

        LOG_INFO("tfpws_band_limit",
                 "gpu_id=%d fmin_hz=%.6g fmax_hz=%.6g taper_hz=%.6g df_hz=%.6g",
                 device_id,
                 band_fmin,
                 band_fmax,
                 band_taper_hz,
                 df_hz);
    }

    dim3 dimGrid_1D, dimBlock_1D;
    dim3 dimGrid_2D, dimBlock_2D;
    dim3 dimGrid_3D, dimBlock_3D;

    float *d_linear_stack = workspace->d_linear_stack;
    CUDACHECK(cudaMemcpy(d_linear_stack, linear_stack,
                         nsamples * sizeof(float), cudaMemcpyHostToDevice));
    free(linear_stack);

    float *d_prestack_data = workspace->d_prestack_data;
    CUDACHECK(cudaMemcpy(d_prestack_data, prestack_data,
                         prestack_count * sizeof(float), cudaMemcpyHostToDevice));
    free(prestack_data);

    float *d_group_trace_weights = workspace->d_group_trace_weights;
    CUDACHECK(cudaMemcpy(d_group_trace_weights, group_trace_weights,
                         (std::size_t)ngroups * sizeof(float), cudaMemcpyHostToDevice));
    free(group_trace_weights);

    if (ensure_tfpws_fixed_cufft_plans(workspace) != 0 ||
        ensure_tfpws_group_cufft_plans(workspace, ngroups) != 0)
        return 1;

    cuComplex *d_trace_spectrum = workspace->d_trace_spectrum;
    CUFFTCHECK(cufftExecR2C(workspace->plan_fwd_traces,
                            (cufftReal *)d_prestack_data,
                            (cufftComplex *)d_trace_spectrum));
    DimCompute2D(&dimGrid_2D, &dimBlock_2D, nsamples, ngroups);
    hilbertTransformKernel<<<dimGrid_2D, dimBlock_2D>>>(
        d_trace_spectrum, nsamples, ngroups);
    CUDA_KERNEL_CHECK();

    cufftComplex *d_linear_spectrum = workspace->d_linear_spectrum;
    cufftComplex *d_out_spectrum = workspace->d_out_spectrum;
    cufftComplex *d_tfpw_stack_complex = workspace->d_tfpw_stack_complex;
    float *d_tfpw_stack = workspace->d_tfpw_stack;
    cufftComplex *d_stack_tf_chunk = workspace->d_stack_tf_chunk;
    cufftComplex *d_chunk_spectrum = workspace->d_chunk_spectrum;
    cuComplex *d_weight_chunk = workspace->d_weight_chunk;
    cufftComplex *d_trace_tf_chunk = workspace->d_trace_tf_chunk;

    CUFFTCHECK(cufftExecR2C(workspace->plan_fwd_single_trace,
                            (cufftReal *)d_linear_stack,
                            (cufftComplex *)d_linear_spectrum));

    DimCompute2D(&dimGrid_2D, &dimBlock_2D, nsamples, 1);
    hilbertTransformKernel<<<dimGrid_2D, dimBlock_2D>>>(
        d_linear_spectrum, nsamples, 1);
    CUDA_KERNEL_CHECK();

    CUDACHECK(cudaMemset(d_out_spectrum, 0, nsamples * sizeof(cufftComplex)));
    if (band_limited)
    {
        CUDACHECK(cudaMemcpy(d_out_spectrum,
                             d_linear_spectrum,
                             nsamples * sizeof(cufftComplex),
                             cudaMemcpyDeviceToDevice));
    }

    const float weight_order = 1.0f;
    const float gaussian_scale = 0.1f;

    for (std::size_t ichunk = 0; ichunk < num_freq_chunks; ++ichunk)
    {
        const std::size_t f_start = ichunk * freq_chunk_size;
        const std::size_t remaining = nfreq - f_start;
        const int sub_nfreq =
            (int)((remaining < freq_chunk_size) ? remaining : freq_chunk_size);

        if (band_limited &&
            !frequency_chunk_overlaps_band(f_start,
                                           sub_nfreq,
                                           df_hz,
                                           band_fmin,
                                           band_fmax))
        {
            continue;
        }

        DimCompute3D(&dimGrid_3D, &dimBlock_3D, nsamples, sub_nfreq, 1);
        gaussianModulateSub<<<dimGrid_3D, dimBlock_3D>>>(
            d_linear_spectrum,
            d_stack_tf_chunk,
            1,
            nsamples,
            (int)f_start,
            sub_nfreq,
            gaussian_scale);
        CUDA_KERNEL_CHECK();

        if ((std::size_t)sub_nfreq < freq_chunk_size)
        {
            const std::size_t used = (std::size_t)sub_nfreq * nsamples;
            const std::size_t tail = (freq_chunk_size - (std::size_t)sub_nfreq) * nsamples;
            CUDACHECK(cudaMemset(d_stack_tf_chunk + used,
                                 0,
                                 tail * sizeof(cufftComplex)));
        }

        CUFFTCHECK(cufftExecC2C(workspace->plan_inv_stack_chunk,
                                d_stack_tf_chunk,
                                d_stack_tf_chunk,
                                CUFFT_INVERSE));

        DimCompute3D(&dimGrid_3D, &dimBlock_3D,
                     nsamples, sub_nfreq, ngroups);
        gaussianModulateSub<<<dimGrid_3D, dimBlock_3D>>>(
            d_trace_spectrum,
            d_trace_tf_chunk,
            ngroups,
            nsamples,
            (int)f_start,
            sub_nfreq,
            gaussian_scale);
        CUDA_KERNEL_CHECK();

        if ((std::size_t)sub_nfreq < freq_chunk_size)
        {
            const std::size_t used = (std::size_t)ngroups * (std::size_t)sub_nfreq * nsamples;
            const std::size_t tail = (std::size_t)ngroups *
                                     (freq_chunk_size - (std::size_t)sub_nfreq) *
                                     nsamples;
            CUDACHECK(cudaMemset(d_trace_tf_chunk + used,
                                 0,
                                 tail * sizeof(cufftComplex)));
        }

        CUFFTCHECK(cufftExecC2C(workspace->plan_inv_trace_chunk,
                                d_trace_tf_chunk,
                                d_trace_tf_chunk,
                                CUFFT_INVERSE));

        DimCompute2D(&dimGrid_2D, &dimBlock_2D, nsamples, sub_nfreq);
        calculateWeightSub<<<dimGrid_2D, dimBlock_2D>>>(
            d_trace_tf_chunk,
            d_group_trace_weights,
            d_weight_chunk,
            nsamples,
            ngroups,
            0,
            sub_nfreq);
        CUDA_KERNEL_CHECK();

        applyWeight<<<dimGrid_2D, dimBlock_2D>>>(
            d_stack_tf_chunk,
            d_weight_chunk,
            sub_nfreq,
            nsamples,
            weight_order);
        CUDA_KERNEL_CHECK();

        DimCompute1D(&dimGrid_1D, &dimBlock_1D, sub_nfreq);
        sumOverTimeAxisKernel<<<dimGrid_1D, dimBlock_1D>>>(
            d_stack_tf_chunk,
            band_limited ? d_chunk_spectrum : (d_out_spectrum + f_start),
            sub_nfreq,
            nsamples);
        CUDA_KERNEL_CHECK();

        if (band_limited)
        {
            blendBandLimitedSpectrum<<<dimGrid_1D, dimBlock_1D>>>(
                d_out_spectrum,
                d_linear_spectrum,
                d_chunk_spectrum,
                (int)f_start,
                sub_nfreq,
                (float)df_hz,
                (float)band_fmin,
                (float)band_fmax,
                (float)band_taper_hz);
            CUDA_KERNEL_CHECK();
        }
    }

    CUFFTCHECK(cufftExecC2C(workspace->plan_inv_final,
                            d_out_spectrum,
                            d_tfpw_stack_complex,
                            CUFFT_INVERSE));

    DimCompute1D(&dimGrid_1D, &dimBlock_1D, nsamples);
    extractReal<<<dimGrid_1D, dimBlock_1D>>>(
        d_tfpw_stack,
        d_tfpw_stack_complex,
        nsamples);
    CUDA_KERNEL_CHECK();

    float *tfpw_stack = (float *)malloc(nsamples * sizeof(float));
    if (!tfpw_stack)
    {
        LOG_ERROR("host_allocation_failed",
                  "buffer=tfpw_stack bytes=%zu",
                  (std::size_t)nsamples * sizeof(float));
        return 1;
    }
    CUDACHECK(cudaMemcpy(tfpw_stack, d_tfpw_stack,
                         nsamples * sizeof(float), cudaMemcpyDeviceToHost));

    *out_header = ncf_hd;
    *out_data = tfpw_stack;
    return 0;
}
