#include "device_memory_internal.cuh"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static const double GPU_RAM_AUTO_FRACTION = 0.70;
static const double GPU_RAM_AUTO_RESERVE_MIB = 4096.0;

static size_t ApplyGpuRamLimit(size_t device_id, size_t freeram,
                               double limit_mib)
{
    size_t limit_bytes = (size_t)(limit_mib * 1024.0 * 1024.0);
    size_t budget = (limit_bytes < freeram) ? limit_bytes : freeram;

    LOG_INFO("gpu_memory_manual_budget",
             "device=%zu raw_free_mib=%.3f limit_mib=%.3f budget_mib=%.3f",
             device_id, DeviceMemoryBytesToMiB(freeram), limit_mib,
             DeviceMemoryBytesToMiB(budget));
    return budget;
}

static size_t ApplyAutoGpuRamBudget(size_t device_id, size_t freeram)
{
    size_t fraction_budget = (size_t)((double)freeram * GPU_RAM_AUTO_FRACTION);
    size_t reserve_bytes = (size_t)(GPU_RAM_AUTO_RESERVE_MIB * 1024.0 * 1024.0);
    size_t reserve_budget = (freeram > reserve_bytes) ? freeram - reserve_bytes
                                                       : fraction_budget;
    size_t budget = (fraction_budget < reserve_budget) ? fraction_budget
                                                       : reserve_budget;

    LOG_INFO("gpu_memory_auto_budget",
             "device=%zu raw_free_mib=%.3f fraction=%.3f reserve_mib=%.3f budget_mib=%.3f",
             device_id, DeviceMemoryBytesToMiB(freeram), GPU_RAM_AUTO_FRACTION,
             GPU_RAM_AUTO_RESERVE_MIB, DeviceMemoryBytesToMiB(budget));
    return budget;
}

static size_t QueryAvailGpuRam(size_t device_id, double gpu_ram_limit_mib)
{
    size_t freeram, totalram;
    CUDACHECK(cudaSetDevice(device_id));
    CUDACHECK(cudaMemGetInfo(&freeram, &totalram));
    size_t raw_freeram = freeram;
    const char *mode = (gpu_ram_limit_mib > 0.0) ? "manual" : "auto";

    if (gpu_ram_limit_mib > 0.0)
    {
        freeram = ApplyGpuRamLimit(device_id, raw_freeram, gpu_ram_limit_mib);
    }
    else
    {
        freeram = ApplyAutoGpuRamBudget(device_id, raw_freeram);
    }

    LOG_INFO("gpu_memory_available",
             "device=%zu mode=%s raw_free_mib=%.3f total_mib=%.3f available_mib=%.3f",
             device_id, mode, DeviceMemoryBytesToMiB(raw_freeram),
             DeviceMemoryBytesToMiB(totalram), DeviceMemoryBytesToMiB(freeram));
    return freeram;
}

typedef struct DeviceMemoryBreakdown
{
    size_t fixed_filter_size;
    size_t sac_seg_size;
    size_t spec_seg_size;
    size_t sac_work_size;
    size_t spec_work_size;
    size_t pre_process_size;
    size_t weight_size;
    size_t tmp_weight_size;
    size_t tmp_size;
    size_t filtered_sacdata_size;
    size_t total_sacdata_size;
    size_t base_spectrum_2x_size;
    size_t whiten_norm_size;
    size_t per_batch_total_size;
    size_t work_nfft_size;
} DeviceMemoryBreakdown;

static void ComputeDeviceMemoryBreakdown(int nseg, int filter_nfft, int output_nfft,
                                         int num_ch, int filter_count,
                                         size_t wh_flag, size_t runabs_flag,
                                         size_t runabs_mf_flag,
                                         DeviceMemoryBreakdown *m)
{
    size_t nseg_size = (size_t)nseg;
    size_t filter_nfft_size = (size_t)filter_nfft;
    size_t output_nfft_size = (size_t)output_nfft;
    size_t channel_count = (size_t)num_ch;

    memset(m, 0, sizeof(*m));
    m->work_nfft_size = (filter_nfft > output_nfft) ? filter_nfft_size : output_nfft_size;

    m->sac_seg_size = channel_count * nseg_size * sizeof(float);
    m->spec_seg_size = channel_count * nseg_size * sizeof(cuComplex);
    m->sac_work_size = channel_count * m->work_nfft_size * sizeof(float);
    m->spec_work_size = channel_count * m->work_nfft_size * sizeof(cuComplex);
    m->pre_process_size = channel_count * (sizeof(double) + sizeof(double));
    m->fixed_filter_size = (size_t)filter_count * filter_nfft_size * sizeof(float);

    if (runabs_flag || runabs_mf_flag)
    {
        m->weight_size = channel_count * nseg_size * sizeof(float);
        m->tmp_weight_size = nseg_size * sizeof(float);
        m->tmp_size = nseg_size * sizeof(float);
        if (runabs_mf_flag)
        {
            m->filtered_sacdata_size = channel_count * nseg_size * sizeof(float);
            m->total_sacdata_size = channel_count * nseg_size * sizeof(float);
            m->base_spectrum_2x_size = channel_count * filter_nfft_size * sizeof(cuComplex);
        }
        m->whiten_norm_size = m->weight_size + m->tmp_weight_size + m->tmp_size
                              + m->filtered_sacdata_size + m->total_sacdata_size
                              + m->base_spectrum_2x_size;
    }
    else if (wh_flag)
    {
        m->weight_size = channel_count * nseg_size * sizeof(float);
        m->tmp_weight_size = nseg_size * sizeof(float);
        m->tmp_size = nseg_size * sizeof(float);
        m->whiten_norm_size = m->weight_size + m->tmp_weight_size + m->tmp_size;
    }

    m->per_batch_total_size = m->sac_seg_size
                              + m->spec_seg_size
                              + m->pre_process_size
                              + m->sac_work_size
                              + m->spec_work_size
                              + m->whiten_norm_size;
}

static cufftResult_t QueryCufftWorkspaceBreakdown(int nseg, int filter_nfft,
                                                  int output_nfft, int num_ch,
                                                  size_t batch_groups,
                                                  CufftWorkspaceBreakdown *ws);

static int GpuBatchFits(int nseg, int filter_nfft, int output_nfft, int num_ch,
                        const DeviceMemoryBreakdown *mem, size_t batch,
                        size_t availram, CufftWorkspaceBreakdown *ws,
                        size_t *required_ram)
{
    cufftResult_t err = QueryCufftWorkspaceBreakdown(nseg, filter_nfft, output_nfft,
                                                     num_ch, batch, ws);
    if (err != CUFFT_SUCCESS)
    {
        LOG_DEBUG("gpu_memory_cufft_plan_rejected",
                  "frame_batch=%zu code=%d name=%s",
                  batch, (int)err, cufftResultName(err));
        return 0;
    }

    *required_ram = mem->fixed_filter_size
                    + batch * mem->per_batch_total_size
                    + ws->shared_max;
    return *required_ram <= availram;
}

static cufftResult_t QueryCufftWorkspaceBreakdown(int nseg, int filter_nfft,
                                                  int output_nfft, int num_ch,
                                                  size_t batch_groups,
                                                  CufftWorkspaceBreakdown *ws)
{
    cufftHandle plan = 0;
    cufftResult_t err;
    size_t plan_batch_size = (size_t)num_ch * batch_groups;

    memset(ws, 0, sizeof(*ws));
    if (plan_batch_size == 0 || plan_batch_size > (size_t)INT_MAX)
    {
        return CUFFT_INVALID_VALUE;
    }

    int plan_batch = (int)plan_batch_size;

    err = CreateCufftPlanNoAuto(&plan, nseg, CUFFT_R2C, plan_batch, &ws->fwd_1x);
    if (err != CUFFT_SUCCESS)
    {
        return err;
    }
    cufftDestroy(plan);

    err = CreateCufftPlanNoAuto(&plan, nseg, CUFFT_C2R, plan_batch, &ws->inv_1x);
    if (err != CUFFT_SUCCESS)
    {
        return err;
    }
    cufftDestroy(plan);

    err = CreateCufftPlanNoAuto(&plan, filter_nfft, CUFFT_R2C, plan_batch, &ws->fwd_filter);
    if (err != CUFFT_SUCCESS)
    {
        return err;
    }
    cufftDestroy(plan);

    err = CreateCufftPlanNoAuto(&plan, filter_nfft, CUFFT_C2R, plan_batch, &ws->inv_filter);
    if (err != CUFFT_SUCCESS)
    {
        return err;
    }
    cufftDestroy(plan);

    err = CreateCufftPlanNoAuto(&plan, output_nfft, CUFFT_R2C, plan_batch, &ws->fwd_output);
    if (err != CUFFT_SUCCESS)
    {
        return err;
    }
    cufftDestroy(plan);

    ws->shared_max = ws->fwd_1x;
    ws->shared_max = DeviceMemoryMaxSize(ws->shared_max, ws->inv_1x);
    ws->shared_max = DeviceMemoryMaxSize(ws->shared_max, ws->fwd_filter);
    ws->shared_max = DeviceMemoryMaxSize(ws->shared_max, ws->inv_filter);
    ws->shared_max = DeviceMemoryMaxSize(ws->shared_max, ws->fwd_output);
    ws->sum_if_auto = ws->fwd_1x + ws->inv_1x + ws->fwd_filter + ws->inv_filter + ws->fwd_output;
    return CUFFT_SUCCESS;
}

size_t EstimateGpuFrameBatch(size_t gpu_id, int nseg, int filter_nfft,
                             int output_nfft, int num_ch,
                             int filter_count, size_t wh_flag,
                             size_t runabs_flag, size_t runabs_mf_flag,
                             double gpu_ram_limit_mib)
{
    if (nseg <= 0 || filter_nfft < nseg || output_nfft < nseg ||
        num_ch <= 0 || filter_count <= 0)
    {
        LOG_ERROR("gpu_memory_estimate_invalid_input",
                  "nseg=%d filter_nfft=%d output_nfft=%d num_ch=%d filter_count=%d",
                  nseg, filter_nfft, output_nfft, num_ch, filter_count);
        exit(EXIT_FAILURE);
    }

    DeviceMemoryBreakdown mem;
    ComputeDeviceMemoryBreakdown(nseg, filter_nfft, output_nfft, num_ch,
                                 filter_count, wh_flag, runabs_flag,
                                 runabs_mf_flag, &mem);
    size_t availram = QueryAvailGpuRam(gpu_id, gpu_ram_limit_mib);
    CUDACHECK(cudaSetDevice(gpu_id));
    LOG_INFO("gpu_memory_estimate_start", "device=%zu available_gb=%.3f",
             gpu_id, availram * 1.0 / (1L << 30));
    LOG_INFO("gpu_memory_estimate_layout",
             "device=%zu nseg=%d filter_nfft=%d output_nfft=%d work_nfft=%zu num_ch=%d filter_count=%d fixed_filter_mib=%.3f per_frame_data_mib=%.3f per_frame_norm_mib=%.3f per_frame_total_mib=%.3f wh_flag=%zu runabs_flag=%zu runabs_mf_flag=%zu",
             gpu_id, nseg, filter_nfft, output_nfft, mem.work_nfft_size, num_ch, filter_count,
             DeviceMemoryBytesToMiB(mem.fixed_filter_size),
             DeviceMemoryBytesToMiB(mem.per_batch_total_size - mem.whiten_norm_size),
             DeviceMemoryBytesToMiB(mem.whiten_norm_size),
             DeviceMemoryBytesToMiB(mem.per_batch_total_size),
             wh_flag, runabs_flag, runabs_mf_flag);
    LOG_INFO("gpu_memory_estimate_arrays",
             "device=%zu d_sacdata_mib=%.3f d_spectrum_mib=%.3f d_sacdata_2x_mib=%.3f d_spectrum_2x_mib=%.3f d_sum_isum_mib=%.3f d_weight_mib=%.3f d_tmp_weight_mib=%.3f d_tmp_mib=%.3f d_filtered_sacdata_mib=%.3f d_total_sacdata_mib=%.3f d_base_spectrum_2x_mib=%.3f",
             gpu_id,
             DeviceMemoryBytesToMiB(mem.sac_seg_size),
             DeviceMemoryBytesToMiB(mem.spec_seg_size),
             DeviceMemoryBytesToMiB(mem.sac_work_size),
             DeviceMemoryBytesToMiB(mem.spec_work_size),
             DeviceMemoryBytesToMiB(mem.pre_process_size),
             DeviceMemoryBytesToMiB(mem.weight_size),
             DeviceMemoryBytesToMiB(mem.tmp_weight_size),
             DeviceMemoryBytesToMiB(mem.tmp_size),
             DeviceMemoryBytesToMiB(mem.filtered_sacdata_size),
             DeviceMemoryBytesToMiB(mem.total_sacdata_size),
             DeviceMemoryBytesToMiB(mem.base_spectrum_2x_size));

    if (availram <= mem.fixed_filter_size || mem.per_batch_total_size == 0)
    {
        LOG_ERROR("gpu_memory_estimate_no_fit",
                  "device=%zu fixed_filter_mib=%.3f per_batch_total_mib=%.3f available_mib=%.3f",
                  gpu_id, DeviceMemoryBytesToMiB(mem.fixed_filter_size),
                  DeviceMemoryBytesToMiB(mem.per_batch_total_size),
                  DeviceMemoryBytesToMiB(availram));
        return 0;
    }

    size_t high = (availram - mem.fixed_filter_size) / mem.per_batch_total_size;
    high = high > _RISTRICT_MAX_GPU_BATCH ? _RISTRICT_MAX_GPU_BATCH : high;

    size_t low = 0;
    CufftWorkspaceBreakdown ws = {0};
    size_t reqram = 0;
    while (low < high)
    {
        size_t mid = low + (high - low + 1) / 2;
        if (GpuBatchFits(nseg, filter_nfft, output_nfft, num_ch,
                         &mem, mid, availram, &ws, &reqram))
        {
            low = mid;
        }
        else
        {
            high = mid - 1;
        }
    }

    size_t batch = low;
    if (batch > 0)
    {
        CUFFTCHECK(QueryCufftWorkspaceBreakdown(nseg, filter_nfft, output_nfft,
                                                num_ch, batch, &ws));
        size_t selected_reqram = mem.fixed_filter_size
                                 + batch * mem.per_batch_total_size
                                 + ws.shared_max;
        LOG_INFO("gpu_memory_estimate_selected",
                 "device=%zu frame_batch=%zu fixed_filter_mib=%.3f frame_arrays_mib=%.3f cufft_workspace_shared_mib=%.3f cufft_workspace_auto_sum_mib=%.3f cufft_workspace_saved_mib=%.3f total_mib=%.3f available_mib=%.3f cufft_fwd_1x_mib=%.3f cufft_inv_1x_mib=%.3f cufft_fwd_filter_mib=%.3f cufft_inv_filter_mib=%.3f cufft_fwd_output_mib=%.3f",
                 gpu_id, batch,
                 DeviceMemoryBytesToMiB(mem.fixed_filter_size),
                 DeviceMemoryBytesToMiB(batch * mem.per_batch_total_size),
                 DeviceMemoryBytesToMiB(ws.shared_max),
                 DeviceMemoryBytesToMiB(ws.sum_if_auto),
                 DeviceMemoryBytesToMiB(ws.sum_if_auto - ws.shared_max),
                 DeviceMemoryBytesToMiB(selected_reqram),
                 DeviceMemoryBytesToMiB(availram),
                 DeviceMemoryBytesToMiB(ws.fwd_1x),
                 DeviceMemoryBytesToMiB(ws.inv_1x),
                 DeviceMemoryBytesToMiB(ws.fwd_filter),
                 DeviceMemoryBytesToMiB(ws.inv_filter),
                 DeviceMemoryBytesToMiB(ws.fwd_output));
    }
    else
    {
        LOG_ERROR("gpu_memory_estimate_no_fit",
                  "device=%zu fixed_filter_mib=%.3f per_batch_total_mib=%.3f available_mib=%.3f",
                  gpu_id, DeviceMemoryBytesToMiB(mem.fixed_filter_size),
                  DeviceMemoryBytesToMiB(mem.per_batch_total_size),
                  DeviceMemoryBytesToMiB(availram));
    }
    return batch;
}
