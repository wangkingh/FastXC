#ifndef TFPWS_SCHEDULE_HPP
#define TFPWS_SCHEDULE_HPP

#include <cstddef>
#include <climits>
#include <vector>

#include <cuda_runtime.h>

#include "cuda.util.cuh"
#include "logger.h"
#include "tfpws_input.hpp"
#include "tfpws_types.hpp"

extern "C"
{
#include "arguproc.h"
}

static const double TFPWS_GPU_MEMORY_FRACTION = 0.90;
static const double TFPWS_RUNTIME_RESERVE_FRACTION = 0.05;
static const double TFPWS_RUNTIME_RESERVE_MAX_FRACTION = 0.25;
static const std::size_t TFPWS_RUNTIME_RESERVE_FLOOR_BYTES =
    (std::size_t)64 * 1024 * 1024;

struct TfpwsChunkPlan
{
    std::size_t freq_chunk_size;
    std::size_t num_freq_chunks;
    long double resident_data_bytes;
    std::size_t cufft_workspace_bytes;
    std::size_t runtime_reserve_bytes;
    long double planned_peak_bytes;
    bool plan_valid;
    bool fits_budget;
};

static inline double bytes_to_gib(long double bytes)
{
    return (double)(bytes / (1024.0L * 1024.0L * 1024.0L));
}

static inline std::size_t estimate_tfpws_runtime_reserve(std::size_t memory_budget_bytes)
{
    if (memory_budget_bytes == 0)
        return 0;

    long double reserve =
        (long double)memory_budget_bytes * TFPWS_RUNTIME_RESERVE_FRACTION;
    const long double floor =
        (long double)TFPWS_RUNTIME_RESERVE_FLOOR_BYTES;
    const long double cap =
        (long double)memory_budget_bytes * TFPWS_RUNTIME_RESERVE_MAX_FRACTION;

    if (reserve < floor)
        reserve = floor;
    if (reserve > cap)
        reserve = cap;
    return (std::size_t)reserve;
}

static inline long double estimate_tfpws_host_input_bytes(std::size_t ngroups,
                                                          std::size_t nsamples)
{
    const long double N = (long double)nsamples;
    const long double G = (long double)ngroups;
    return 4.0L * G * N + 4.0L * N + 4.0L * G;
}

static inline long double estimate_tfpws_resident_data_bytes(std::size_t ngroups,
                                                             std::size_t nsamples,
                                                             std::size_t freq_chunk_size)
{
    const long double N = (long double)nsamples;
    const long double G = (long double)ngroups;
    const long double K = (long double)freq_chunk_size;

    return 12.0L * G * N + 32.0L * N + 4.0L * G +
           K * (8.0L * G * N + 16.0L * N + 8.0L);
}

static inline bool query_tfpws_cufft_workspace(std::size_t ngroups,
                                               std::size_t nsamples,
                                               std::size_t freq_chunk_size,
                                               std::size_t *workspace_bytes)
{
    if (nsamples > (std::size_t)INT_MAX ||
        ngroups > (std::size_t)INT_MAX ||
        freq_chunk_size > (std::size_t)INT_MAX)
        return false;

    const std::size_t trace_chunk_batch = ngroups * freq_chunk_size;
    if (freq_chunk_size != 0 &&
        trace_chunk_batch / freq_chunk_size != ngroups)
        return false;
    if (trace_chunk_batch > (std::size_t)INT_MAX)
        return false;

    int n_hilb[1] = {(int)nsamples};
    int inembed[1] = {(int)nsamples};
    int onembed[1] = {(int)nsamples};
    const int istride = 1;
    const int ostride = 1;
    const int idist = (int)nsamples;
    const int odist = (int)nsamples;

    std::size_t max_work = 0;
    std::size_t work = 0;

    if (CufftPlanQueryWorkSize(1, n_hilb, inembed, istride, idist,
                               onembed, ostride, odist,
                               CUFFT_R2C, (int)ngroups, &work) != 0)
        return false;
    if (work > max_work)
        max_work = work;

    if (CufftPlanQueryWorkSize(1, n_hilb, inembed, istride, idist,
                               onembed, ostride, odist,
                               CUFFT_R2C, 1, &work) != 0)
        return false;
    if (work > max_work)
        max_work = work;

    if (CufftPlanQueryWorkSize(1, n_hilb, inembed, istride, idist,
                               onembed, ostride, odist,
                               CUFFT_C2C, (int)freq_chunk_size, &work) != 0)
        return false;
    if (work > max_work)
        max_work = work;

    if (CufftPlanQueryWorkSize(1, n_hilb, inembed, istride, idist,
                               onembed, ostride, odist,
                               CUFFT_C2C, (int)trace_chunk_batch, &work) != 0)
        return false;
    if (work > max_work)
        max_work = work;

    if (CufftPlanQueryWorkSize(1, n_hilb, inembed, istride, idist,
                               onembed, ostride, odist,
                               CUFFT_C2C, 1, &work) != 0)
        return false;
    if (work > max_work)
        max_work = work;

    *workspace_bytes = max_work;
    return true;
}

static inline TfpwsChunkPlan select_tfpws_resident_chunk_plan(std::size_t memory_budget_bytes,
                                                              std::size_t ngroups,
                                                              std::size_t nsamples,
                                                              std::size_t nfreq)
{
    const std::size_t runtime_reserve =
        estimate_tfpws_runtime_reserve(memory_budget_bytes);
    const long double planned_budget = (long double)memory_budget_bytes;

    TfpwsChunkPlan best = {0, 0, 0.0L, 0, runtime_reserve, 0.0L, false, false};
    std::size_t low = 1;
    std::size_t high = (nfreq < 1) ? 1 : nfreq;

    while (low <= high)
    {
        const std::size_t mid = low + (high - low) / 2;
        std::size_t workspace = 0;
        if (!query_tfpws_cufft_workspace(ngroups, nsamples, mid, &workspace))
        {
            if (mid == 0)
                break;
            high = mid - 1;
            continue;
        }

        const long double data_bytes =
            estimate_tfpws_resident_data_bytes(ngroups, nsamples, mid);
        const long double planned_peak =
            data_bytes + (long double)workspace + (long double)runtime_reserve;

        if (planned_peak <= planned_budget)
        {
            best.freq_chunk_size = mid;
            best.num_freq_chunks = (nfreq + mid - 1) / mid;
            best.resident_data_bytes = data_bytes;
            best.cufft_workspace_bytes = workspace;
            best.runtime_reserve_bytes = runtime_reserve;
            best.planned_peak_bytes = planned_peak;
            best.plan_valid = true;
            best.fits_budget = true;
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }

    if (best.freq_chunk_size == 0)
    {
        std::size_t workspace = 0;
        const bool valid =
            query_tfpws_cufft_workspace(ngroups, nsamples, 1, &workspace);
        const long double data_bytes =
            estimate_tfpws_resident_data_bytes(ngroups, nsamples, 1);
        const long double planned_peak =
            data_bytes + (long double)workspace + (long double)runtime_reserve;
        best.freq_chunk_size = 1;
        best.num_freq_chunks = nfreq;
        best.resident_data_bytes = data_bytes;
        best.cufft_workspace_bytes = workspace;
        best.runtime_reserve_bytes = runtime_reserve;
        best.planned_peak_bytes = planned_peak;
        best.plan_valid = valid;
        best.fits_budget = valid && planned_peak <= planned_budget;
    }

    return best;
}

static inline bool frequency_chunk_overlaps_band(std::size_t f_start,
                                                 int sub_nfreq,
                                                 double df,
                                                 double fmin,
                                                 double fmax)
{
    const double chunk_fmin = (double)f_start * df;
    const double chunk_fmax = (double)(f_start + (std::size_t)sub_nfreq - 1) * df;
    return chunk_fmax >= fmin && chunk_fmin <= fmax;
}

static inline std::size_t count_gpu_id(const std::vector<int> &gpu_ids,
                                       int device_id)
{
    std::size_t count = 0;
    for (std::size_t i = 0; i < gpu_ids.size(); ++i)
    {
        if (gpu_ids[i] == device_id)
            ++count;
    }
    return count;
}

static inline std::size_t gpu_id_occurrence_index(const std::vector<int> &gpu_ids,
                                                  int device_id,
                                                  std::size_t position)
{
    std::size_t count = 0;
    for (std::size_t i = 0; i <= position && i < gpu_ids.size(); ++i)
    {
        if (gpu_ids[i] == device_id)
            ++count;
    }
    return count;
}

static inline std::vector<GpuWorkerConfig> make_tfpws_worker_configs(const ARGUTYPE *argument)
{
    std::vector<int> gpu_ids = parse_gpu_list(argument->gpu_list);
    std::vector<double> gpu_ram_limits =
        parse_gpu_ram_limit_mib_list(argument->gpu_ram_limit_mib_list,
                                     gpu_ids.size());
    std::vector<GpuWorkerConfig> configs;
    configs.reserve(gpu_ids.size());

    for (std::size_t i = 0; i < gpu_ids.size(); ++i)
    {
        int device_id = gpu_ids[i];
        double worker_limit_mib = gpu_ram_limits[i];
        std::size_t physical_worker_count = count_gpu_id(gpu_ids, device_id);
        std::size_t worker_ordinal =
            gpu_id_occurrence_index(gpu_ids, device_id, i);
        CUDACHECK(cudaSetDevice(device_id));

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDACHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        long double automatic_budget =
            ((long double)free_bytes * TFPWS_GPU_MEMORY_FRACTION) /
            (long double)physical_worker_count;
        long double worker_budget = automatic_budget;
        long double manual_budget = 0.0L;
        if (worker_limit_mib > 0.0)
        {
            manual_budget = (long double)worker_limit_mib * 1024.0L * 1024.0L;
            if (manual_budget < worker_budget)
                worker_budget = manual_budget;
        }

        GpuWorkerConfig config;
        config.worker_index = i;
        config.device_id = device_id;
        config.memory_budget_bytes = (std::size_t)worker_budget;
        configs.push_back(config);

        LOG_INFO("worker_memory_budget",
                 "worker_index=%zu physical_gpu_id=%d physical_worker_ordinal=%zu physical_worker_count=%zu total_gpu_memory_gib=%.3f free_gpu_memory_gib=%.3f auto_budget_gib=%.3f manual_budget_from_M_gib=%.3f final_worker_budget_gib=%.3f worker_ram_limit_mib=%.2f",
                 i,
                 device_id,
                 worker_ordinal,
                 physical_worker_count,
                 bytes_to_gib((long double)total_bytes),
                 bytes_to_gib((long double)free_bytes),
                 bytes_to_gib(automatic_budget),
                 bytes_to_gib(manual_budget),
                 bytes_to_gib(worker_budget),
                 worker_limit_mib);
    }

    return configs;
}

#endif
