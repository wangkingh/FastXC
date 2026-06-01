#include "tfpws_workspace.hpp"

#include <cstddef>
#include <cstring>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "cuda.util.cuh"
#include "logger.h"
#include "tfpws_schedule.hpp"

void init_tfpws_device_workspace_struct(TfpwsDeviceWorkspace *workspace)
{
    if (!workspace)
        return;
    std::memset(workspace, 0, sizeof(*workspace));
}

static void refresh_tfpws_actual_cufft_workspace(TfpwsDeviceWorkspace *workspace)
{
    workspace->actual_cufft_workspace_bytes =
        (workspace->fixed_cufft_workspace_bytes > workspace->group_cufft_workspace_bytes)
            ? workspace->fixed_cufft_workspace_bytes
            : workspace->group_cufft_workspace_bytes;
}

static void destroy_tfpws_group_cufft_plans(TfpwsDeviceWorkspace *workspace)
{
    if (!workspace)
        return;
    if (workspace->plan_fwd_traces)
    {
        cufftDestroy(workspace->plan_fwd_traces);
        workspace->plan_fwd_traces = 0;
    }
    if (workspace->plan_inv_trace_chunk)
    {
        cufftDestroy(workspace->plan_inv_trace_chunk);
        workspace->plan_inv_trace_chunk = 0;
    }
    workspace->group_cufft_plans_ready = false;
    workspace->cached_ngroups = 0;
    workspace->group_cufft_workspace_bytes = 0;
    refresh_tfpws_actual_cufft_workspace(workspace);
}

static void destroy_tfpws_fixed_cufft_plans(TfpwsDeviceWorkspace *workspace)
{
    if (!workspace)
        return;
    if (workspace->plan_fwd_single_trace)
    {
        cufftDestroy(workspace->plan_fwd_single_trace);
        workspace->plan_fwd_single_trace = 0;
    }
    if (workspace->plan_inv_stack_chunk)
    {
        cufftDestroy(workspace->plan_inv_stack_chunk);
        workspace->plan_inv_stack_chunk = 0;
    }
    if (workspace->plan_inv_final)
    {
        cufftDestroy(workspace->plan_inv_final);
        workspace->plan_inv_final = 0;
    }
    workspace->fixed_cufft_plans_ready = false;
    workspace->fixed_cufft_workspace_bytes = 0;
    refresh_tfpws_actual_cufft_workspace(workspace);
}

static void destroy_tfpws_cufft_plans(TfpwsDeviceWorkspace *workspace)
{
    destroy_tfpws_group_cufft_plans(workspace);
    destroy_tfpws_fixed_cufft_plans(workspace);
}

int allocate_tfpws_device_workspace(TfpwsDeviceWorkspace *workspace,
                                    const TfpwsDeviceWorkspacePlan *plan,
                                    std::size_t worker_index,
                                    int device_id)
{
    if (!workspace || !plan || plan->nsamples == 0 ||
        plan->max_ngroups == 0 || plan->freq_chunk_size == 0)
    {
        LOG_ERROR("invalid_tfpws_workspace_plan",
                  "worker_index=%zu gpu_id=%d",
                  worker_index,
                  device_id);
        return 1;
    }

    init_tfpws_device_workspace_struct(workspace);
    workspace->plan = *plan;
    workspace->worker_index = worker_index;
    workspace->device_id = device_id;

    CUDACHECK(cudaSetDevice(device_id));

    const std::size_t N = plan->nsamples;
    const std::size_t G = plan->max_ngroups;
    const std::size_t K = plan->freq_chunk_size;
    const std::size_t prestack_count = G * N;

    if (plan->cufft_workspace_bytes > 0)
        GpuMalloc(&workspace->d_cufft_workspace, plan->cufft_workspace_bytes);

    GpuMalloc((void **)&workspace->d_linear_stack, N * sizeof(float));
    GpuMalloc((void **)&workspace->d_prestack_data,
              prestack_count * sizeof(float));
    GpuMalloc((void **)&workspace->d_group_trace_weights,
              G * sizeof(float));
    GpuMalloc((void **)&workspace->d_trace_spectrum,
              prestack_count * sizeof(cufftComplex));
    GpuMalloc((void **)&workspace->d_linear_spectrum,
              N * sizeof(cufftComplex));
    GpuMalloc((void **)&workspace->d_out_spectrum,
              N * sizeof(cufftComplex));
    GpuMalloc((void **)&workspace->d_tfpw_stack_complex,
              N * sizeof(cufftComplex));
    GpuMalloc((void **)&workspace->d_tfpw_stack,
              N * sizeof(float));
    GpuMalloc((void **)&workspace->d_stack_tf_chunk,
              K * N * sizeof(cufftComplex));
    if (plan->band_limited)
    {
        GpuMalloc((void **)&workspace->d_chunk_spectrum,
                  K * sizeof(cufftComplex));
    }
    GpuMalloc((void **)&workspace->d_weight_chunk,
              K * N * sizeof(cuComplex));
    GpuMalloc((void **)&workspace->d_trace_tf_chunk,
              G * K * N * sizeof(cufftComplex));

    workspace->initialized = true;
    LOG_INFO("tfpws_device_workspace_allocated",
             "worker_index=%zu gpu_id=%d max_groups=%u samples=%u freq_chunk_size=%zu chunks=%zu resident_data_gib=%.3f cufft_workspace_gib=%.3f planned_peak_gib=%.3f",
             worker_index,
             device_id,
             plan->max_ngroups,
             plan->nsamples,
             plan->freq_chunk_size,
             plan->num_freq_chunks,
             bytes_to_gib(plan->resident_data_bytes),
             bytes_to_gib((long double)plan->cufft_workspace_bytes),
             bytes_to_gib(plan->planned_peak_bytes));
    return 0;
}

void free_tfpws_device_workspace(TfpwsDeviceWorkspace *workspace)
{
    if (!workspace)
        return;

    destroy_tfpws_cufft_plans(workspace);
    GpuFree((void **)&workspace->d_trace_tf_chunk);
    GpuFree((void **)&workspace->d_weight_chunk);
    GpuFree((void **)&workspace->d_chunk_spectrum);
    GpuFree((void **)&workspace->d_stack_tf_chunk);
    GpuFree((void **)&workspace->d_tfpw_stack);
    GpuFree((void **)&workspace->d_tfpw_stack_complex);
    GpuFree((void **)&workspace->d_out_spectrum);
    GpuFree((void **)&workspace->d_linear_spectrum);
    GpuFree((void **)&workspace->d_trace_spectrum);
    GpuFree((void **)&workspace->d_group_trace_weights);
    GpuFree((void **)&workspace->d_prestack_data);
    GpuFree((void **)&workspace->d_linear_stack);
    GpuFree((void **)&workspace->d_cufft_workspace);
    init_tfpws_device_workspace_struct(workspace);
}

static void update_workspace_plan_bytes(std::size_t *current_max,
                                        std::size_t plan_workspace)
{
    if (plan_workspace > *current_max)
        *current_max = plan_workspace;
}

int ensure_tfpws_fixed_cufft_plans(TfpwsDeviceWorkspace *workspace)
{
    if (workspace->fixed_cufft_plans_ready)
    {
        return 0;
    }

    destroy_tfpws_fixed_cufft_plans(workspace);

    const std::size_t nsamples = workspace->plan.nsamples;
    const std::size_t freq_chunk_size = workspace->plan.freq_chunk_size;
    const int rank_hilb = 1;
    int n_hilb[1] = {(int)nsamples};
    int inembed[1] = {(int)nsamples};
    int onembed[1] = {(int)nsamples};
    const int istride = 1;
    const int ostride = 1;
    const int idist = (int)nsamples;
    const int odist = (int)nsamples;
    size_t plan_workspace = 0;

    CufftPlanAllocManual(&workspace->plan_fwd_single_trace, rank_hilb, n_hilb,
                         inembed, istride, idist,
                         onembed, ostride, odist,
                         CUFFT_R2C, 1,
                         workspace->d_cufft_workspace,
                         workspace->plan.cufft_workspace_bytes,
                         &plan_workspace);
    update_workspace_plan_bytes(&workspace->fixed_cufft_workspace_bytes,
                                plan_workspace);

    CufftPlanAllocManual(&workspace->plan_inv_stack_chunk, rank_hilb, n_hilb,
                         inembed, istride, idist,
                         onembed, ostride, odist,
                         CUFFT_C2C, (int)freq_chunk_size,
                         workspace->d_cufft_workspace,
                         workspace->plan.cufft_workspace_bytes,
                         &plan_workspace);
    update_workspace_plan_bytes(&workspace->fixed_cufft_workspace_bytes,
                                plan_workspace);

    CufftPlanAllocManual(&workspace->plan_inv_final, rank_hilb, n_hilb,
                         inembed, istride, idist,
                         onembed, ostride, odist,
                         CUFFT_C2C, 1,
                         workspace->d_cufft_workspace,
                         workspace->plan.cufft_workspace_bytes,
                         &plan_workspace);
    update_workspace_plan_bytes(&workspace->fixed_cufft_workspace_bytes,
                                plan_workspace);

    workspace->fixed_cufft_plans_ready = true;
    refresh_tfpws_actual_cufft_workspace(workspace);
    LOG_INFO("tfpws_fixed_cufft_plans_ready",
             "worker_index=%zu gpu_id=%d freq_chunk_size=%zu cufft_workspace_bound_gib=%.3f fixed_cufft_workspace_actual_gib=%.3f",
             workspace->worker_index,
             workspace->device_id,
             freq_chunk_size,
             bytes_to_gib((long double)workspace->plan.cufft_workspace_bytes),
             bytes_to_gib((long double)workspace->fixed_cufft_workspace_bytes));
    return 0;
}

int ensure_tfpws_group_cufft_plans(TfpwsDeviceWorkspace *workspace,
                                   unsigned ngroups)
{
    if (workspace->group_cufft_plans_ready &&
        workspace->cached_ngroups == ngroups)
    {
        return 0;
    }

    destroy_tfpws_group_cufft_plans(workspace);

    const std::size_t nsamples = workspace->plan.nsamples;
    const std::size_t freq_chunk_size = workspace->plan.freq_chunk_size;
    const int rank_hilb = 1;
    int n_hilb[1] = {(int)nsamples};
    int inembed[1] = {(int)nsamples};
    int onembed[1] = {(int)nsamples};
    const int istride = 1;
    const int ostride = 1;
    const int idist = (int)nsamples;
    const int odist = (int)nsamples;
    size_t plan_workspace = 0;

    CufftPlanAllocManual(&workspace->plan_fwd_traces, rank_hilb, n_hilb,
                         inembed, istride, idist,
                         onembed, ostride, odist,
                         CUFFT_R2C, (int)ngroups,
                         workspace->d_cufft_workspace,
                         workspace->plan.cufft_workspace_bytes,
                         &plan_workspace);
    update_workspace_plan_bytes(&workspace->group_cufft_workspace_bytes,
                                plan_workspace);

    CufftPlanAllocManual(&workspace->plan_inv_trace_chunk, rank_hilb, n_hilb,
                         inembed, istride, idist,
                         onembed, ostride, odist,
                         CUFFT_C2C, (int)((std::size_t)ngroups * freq_chunk_size),
                         workspace->d_cufft_workspace,
                         workspace->plan.cufft_workspace_bytes,
                         &plan_workspace);
    update_workspace_plan_bytes(&workspace->group_cufft_workspace_bytes,
                                plan_workspace);

    workspace->cached_ngroups = ngroups;
    workspace->group_cufft_plans_ready = true;
    refresh_tfpws_actual_cufft_workspace(workspace);
    LOG_INFO("tfpws_group_cufft_plan_cache_refresh",
             "worker_index=%zu gpu_id=%d groups=%u freq_chunk_size=%zu cufft_workspace_bound_gib=%.3f group_cufft_workspace_actual_gib=%.3f total_cufft_workspace_actual_gib=%.3f",
             workspace->worker_index,
             workspace->device_id,
             ngroups,
             freq_chunk_size,
             bytes_to_gib((long double)workspace->plan.cufft_workspace_bytes),
             bytes_to_gib((long double)workspace->group_cufft_workspace_bytes),
             bytes_to_gib((long double)workspace->actual_cufft_workspace_bytes));
    return 0;
}
