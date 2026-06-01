#include "gpu.hpp"

#include <climits>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include <cuComplex.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include "cuda.pws.cuh"
#include "cuda_resources.hpp"
#include "cuda.util.cuh"
#include "host_stage.hpp"
#include "logger.h"
#include "sac_output.hpp"

static bool validate_pws_host_batch(const PwsHostBatch *batch)
{
    if (!batch || batch->items.empty())
    {
        LOG_ERROR("pws_batch_empty", "message=\"no host items\"");
        return false;
    }

    const unsigned nsamples = batch->items[0].nsamples;
    std::size_t total_groups = 0;
    for (std::size_t i = 0; i < batch->items.size(); ++i)
    {
        if (batch->items[i].nsamples != nsamples)
        {
            LOG_ERROR("pws_batch_sample_count_mismatch",
                      "item=%zu samples=%u expected_samples=%u input=\"%s\"",
                      i,
                      batch->items[i].nsamples,
                      nsamples,
                      batch->items[i].input_path.c_str());
            return false;
        }
        total_groups += batch->items[i].ngroups;
    }

    if (batch->nsamples != 0 && batch->nsamples != nsamples)
    {
        LOG_ERROR("pws_batch_declared_sample_count_mismatch",
                  "declared_samples=%u actual_samples=%u",
                  batch->nsamples,
                  nsamples);
        return false;
    }

    if (batch->total_groups != 0 && batch->total_groups != total_groups)
    {
        LOG_ERROR("pws_batch_declared_group_count_mismatch",
                  "declared_groups=%zu actual_groups=%zu",
                  batch->total_groups,
                  total_groups);
        return false;
    }

    if (total_groups > (std::size_t)INT_MAX || nsamples > (unsigned)INT_MAX)
    {
        LOG_ERROR("pws_batch_cufft_size_overflow",
                  "groups=%zu samples=%u",
                  total_groups,
                  nsamples);
        return false;
    }

    return true;
}

static void release_pws_host_batch_staging(PwsHostBatch *batch,
                                           HostGroupBudget *host_budget)
{
    if (!batch || !host_budget)
        return;
    for (std::size_t i = 0; i < batch->items.size(); ++i)
    {
        PwsHostItem *item = &batch->items[i];
        if (item->prestack_data || item->linear_stack || item->group_weights)
            release_host_staging(item, host_budget);
    }
}

int compute_pws_host_batch(PwsHostBatch *batch,
                           int gpu_id,
                           HostGroupBudget *host_budget,
                           std::vector<SACHEAD> *out_headers,
                           float **out_data)
{
    if (out_data)
        *out_data = NULL;
    if (out_headers)
        out_headers->clear();
    if (!out_headers || !out_data)
    {
        LOG_ERROR("pws_batch_output_argument_invalid",
                  "headers_ptr=%p data_ptr=%p",
                  (void *)out_headers,
                  (void *)out_data);
        release_pws_host_batch_staging(batch, host_budget);
        return 1;
    }

    if (!validate_pws_host_batch(batch))
    {
        release_pws_host_batch_staging(batch, host_budget);
        return 1;
    }

    const std::size_t num_pairs = batch->items.size();
    const unsigned nsamples = batch->items[0].nsamples;
    std::size_t total_groups = 0;
    for (std::size_t i = 0; i < num_pairs; ++i)
        total_groups += batch->items[i].ngroups;

    LOG_INFO("pws_gpu_batch_start",
             "gpu=%d pairs=%zu total_groups=%zu samples=%u",
             gpu_id,
             num_pairs,
             total_groups,
             nsamples);

    CUDACHECK(cudaSetDevice(gpu_id));

    dim3 dimGrid_1D, dimBlock_1D;
    dim3 dimGrid_2D, dimBlock_2D;

    CudaDeviceBuffer<float> d_linear_stack;
    CudaDeviceBuffer<float> d_ncf_buffer_all;
    CudaDeviceBuffer<float> d_group_weights;
    CudaDeviceBuffer<size_t> d_pair_group_offsets;
    CudaDeviceBuffer<size_t> d_pair_group_counts;
    CudaDeviceBuffer<cufftComplex> d_spectrum;

    d_linear_stack.allocate(num_pairs * nsamples);
    d_ncf_buffer_all.allocate(total_groups * nsamples);
    d_group_weights.allocate(total_groups);
    d_pair_group_offsets.allocate(num_pairs);
    d_pair_group_counts.allocate(num_pairs);

    std::vector<size_t> pair_group_offsets(num_pairs);
    std::vector<size_t> pair_group_counts(num_pairs);
    out_headers->assign(num_pairs, SACHEAD());

    std::size_t group_offset = 0;
    for (std::size_t pair = 0; pair < num_pairs; ++pair)
    {
        PwsHostItem *item = &batch->items[pair];
        const std::size_t group_count = item->ngroups;
        const std::size_t prestack_count = group_count * nsamples;

        pair_group_offsets[pair] = group_offset;
        pair_group_counts[pair] = group_count;
        (*out_headers)[pair] = item->header;

        CUDACHECK(cudaMemcpy(d_linear_stack.get() + pair * nsamples,
                             item->linear_stack,
                             (std::size_t)nsamples * sizeof(float),
                             cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_ncf_buffer_all.get() + group_offset * nsamples,
                             item->prestack_data,
                             prestack_count * sizeof(float),
                             cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_group_weights.get() + group_offset,
                             item->group_weights,
                             group_count * sizeof(float),
                             cudaMemcpyHostToDevice));

        group_offset += group_count;
        release_host_staging(item, host_budget);
    }

    CUDACHECK(cudaMemcpy(d_pair_group_offsets.get(),
                         pair_group_offsets.data(),
                         num_pairs * sizeof(size_t),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_pair_group_counts.get(),
                         pair_group_counts.data(),
                         num_pairs * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    d_spectrum.allocate(total_groups * nsamples);

    int rank_hilb = 1;
    int n_hilb[1] = {(int)nsamples};
    int inembed[1] = {(int)nsamples};
    int onembed[1] = {(int)nsamples};
    int istride = 1;
    int ostride = 1;
    int idist = (int)nsamples;
    int odist = (int)nsamples;

    CufftPlanGuard plan_fwd;
    plan_fwd.create_many(rank_hilb, n_hilb,
                         inembed, istride, idist,
                         onembed, ostride, odist,
                         CUFFT_R2C, (int)total_groups);

    CUFFTCHECK(cufftExecR2C(plan_fwd.get(),
                            (cufftReal *)d_ncf_buffer_all.get(),
                            d_spectrum.get()));
    DimCompute2D(&dimGrid_2D, &dimBlock_2D, nsamples, total_groups);
    hilbertTransformKernel<<<dimGrid_2D, dimBlock_2D>>>(d_spectrum.get(), nsamples, total_groups);
    CudaCheckLastKernel("hilbertTransformKernel");

    plan_fwd.destroy_now();
    d_ncf_buffer_all.free_now();

    CudaDeviceBuffer<cuComplex> hilbert_complex;
    CudaDeviceBuffer<cufftComplex> analyze_mean;
    CudaDeviceBuffer<float> d_pw_stack;

    hilbert_complex.allocate(total_groups * nsamples);
    analyze_mean.allocate(num_pairs * nsamples);
    d_pw_stack.allocate(num_pairs * nsamples);

    CufftPlanGuard plan_inv_pws;
    plan_inv_pws.create_many(rank_hilb, n_hilb,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2C, (int)total_groups);

    CUFFTCHECK(cufftExecC2C(plan_inv_pws.get(), d_spectrum.get(),
                            hilbert_complex.get(), CUFFT_INVERSE));

    DimCompute1D(&dimGrid_1D, &dimBlock_1D, total_groups * nsamples);
    cudaNormalizeComplex<<<dimGrid_1D, dimBlock_1D>>>(
        hilbert_complex.get(), total_groups * nsamples, nsamples);
    CudaCheckLastKernel("cudaNormalizeComplex");

    DimCompute2D(&dimGrid_2D, &dimBlock_2D, nsamples, num_pairs);
    cudaWeightedMeanBatch<<<dimGrid_2D, dimBlock_2D>>>(
        hilbert_complex.get(),
        d_group_weights.get(),
        d_pair_group_offsets.get(),
        d_pair_group_counts.get(),
        analyze_mean.get(),
        num_pairs,
        nsamples);
    CudaCheckLastKernel("cudaWeightedMeanBatch");

    cudaMultiplyBatch<<<dimGrid_2D, dimBlock_2D>>>(
        d_linear_stack.get(),
        analyze_mean.get(),
        d_pw_stack.get(),
        num_pairs,
        nsamples);
    CudaCheckLastKernel("cudaMultiplyBatch");

    float *pw_stack = (float *)std::malloc(num_pairs * (std::size_t)nsamples * sizeof(float));
    if (!pw_stack)
    {
        LOG_ERROR("pws_batch_output_alloc_failed",
                  "gpu=%d pairs=%zu samples=%u bytes=%zu",
                  gpu_id,
                  num_pairs,
                  nsamples,
                  num_pairs * (std::size_t)nsamples * sizeof(float));
        return 1;
    }

    CUDACHECK(cudaMemcpy(pw_stack, d_pw_stack.get(),
                         num_pairs * (std::size_t)nsamples * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (std::size_t pair = 0; pair < num_pairs; ++pair)
        update_output_stats(&(*out_headers)[pair],
                            pw_stack + pair * (std::size_t)nsamples,
                            nsamples);

    *out_data = pw_stack;
    LOG_INFO("pws_gpu_batch_done",
             "gpu=%d pairs=%zu total_groups=%zu samples=%u",
             gpu_id,
             num_pairs,
             total_groups,
             nsamples);
    return 0;
}
