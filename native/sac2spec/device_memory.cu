#include "device_memory_internal.cuh"

void AllocateGpuMemory(int batch, int nseg, int filter_nfft, int output_nfft,
                       int num_ch, int do_runabs, int do_runabs_mf,
                       int wh_flag,
                       float **d_sacdata, cuComplex **d_spectrum,
                       float **d_padded_sacdata, cuComplex **d_padded_spectrum,
                       cuComplex **d_base_padded_spectrum,
                       float **d_filtered_sacdata, float **d_total_sacdata,
                       float **d_filter_responses, float **d_tmp,
                       float **d_weight, float **d_tmp_weight,
                       int filter_count, double **d_sum, double **d_isum,
                       void **d_cufft_work,
                       cufftHandle *planfwd, cufftHandle *planinv,
                       cufftHandle *planfwd_filter,
                       cufftHandle *planinv_filter,
                       cufftHandle *planfwd_output)
{
    int work_nfft = (filter_nfft > output_nfft) ? filter_nfft : output_nfft;
    CufftWorkspaceBreakdown ws = {0};

    LOG_DEBUG("cufft_plan_channels", "num_ch=%d frame_batch=%d", num_ch, batch);
    CUFFTCHECK(CreateCufftPlanNoAuto(planfwd, nseg, CUFFT_R2C,
                                     num_ch * batch, &ws.fwd_1x));
    CUFFTCHECK(CreateCufftPlanNoAuto(planinv, nseg, CUFFT_C2R,
                                     num_ch * batch, &ws.inv_1x));
    CUFFTCHECK(CreateCufftPlanNoAuto(planfwd_filter, filter_nfft, CUFFT_R2C,
                                     num_ch * batch, &ws.fwd_filter));
    CUFFTCHECK(CreateCufftPlanNoAuto(planinv_filter, filter_nfft, CUFFT_C2R,
                                     num_ch * batch, &ws.inv_filter));
    CUFFTCHECK(CreateCufftPlanNoAuto(planfwd_output, output_nfft, CUFFT_R2C,
                                     num_ch * batch, &ws.fwd_output));

    ws.shared_max = ws.fwd_1x;
    ws.shared_max = DeviceMemoryMaxSize(ws.shared_max, ws.inv_1x);
    ws.shared_max = DeviceMemoryMaxSize(ws.shared_max, ws.fwd_filter);
    ws.shared_max = DeviceMemoryMaxSize(ws.shared_max, ws.inv_filter);
    ws.shared_max = DeviceMemoryMaxSize(ws.shared_max, ws.fwd_output);
    ws.sum_if_auto = ws.fwd_1x + ws.inv_1x + ws.fwd_filter + ws.inv_filter + ws.fwd_output;

    if (ws.shared_max > 0)
    {
        CUDACHECK(cudaMalloc(d_cufft_work, ws.shared_max));
        CUFFTCHECK(cufftSetWorkArea(*planfwd, *d_cufft_work));
        CUFFTCHECK(cufftSetWorkArea(*planinv, *d_cufft_work));
        CUFFTCHECK(cufftSetWorkArea(*planfwd_filter, *d_cufft_work));
        CUFFTCHECK(cufftSetWorkArea(*planinv_filter, *d_cufft_work));
        CUFFTCHECK(cufftSetWorkArea(*planfwd_output, *d_cufft_work));
    }
    LOG_INFO("cufft_workspace_allocated",
             "frame_batch=%d shared_mib=%.3f auto_sum_mib=%.3f saved_mib=%.3f fwd_1x_mib=%.3f inv_1x_mib=%.3f fwd_filter_mib=%.3f inv_filter_mib=%.3f fwd_output_mib=%.3f",
             batch,
             DeviceMemoryBytesToMiB(ws.shared_max),
             DeviceMemoryBytesToMiB(ws.sum_if_auto),
             DeviceMemoryBytesToMiB(ws.sum_if_auto - ws.shared_max),
             DeviceMemoryBytesToMiB(ws.fwd_1x),
             DeviceMemoryBytesToMiB(ws.inv_1x),
             DeviceMemoryBytesToMiB(ws.fwd_filter),
             DeviceMemoryBytesToMiB(ws.inv_filter),
             DeviceMemoryBytesToMiB(ws.fwd_output));

    CUDACHECK(cudaMalloc((void **)d_sacdata, num_ch * batch * nseg * sizeof(float)));
    CUDACHECK(cudaMalloc((void **)d_spectrum, num_ch * batch * nseg * sizeof(cuComplex)));

    CUDACHECK(cudaMalloc((void **)d_padded_sacdata, num_ch * batch * work_nfft * sizeof(float)));
    CUDACHECK(cudaMalloc((void **)d_padded_spectrum, num_ch * batch * work_nfft * sizeof(cuComplex)));

    if (do_runabs_mf)
    {
        CUDACHECK(cudaMalloc((void **)d_base_padded_spectrum,
                             num_ch * batch * filter_nfft * sizeof(cuComplex)));
    }

    CUDACHECK(cudaMalloc((void **)d_filter_responses,
                         filter_count * filter_nfft * sizeof(float)));
    CUDACHECK(cudaMalloc((void **)d_sum, num_ch * batch * sizeof(double)));
    CUDACHECK(cudaMalloc((void **)d_isum, num_ch * batch * sizeof(double)));

    if (!do_runabs && !do_runabs_mf && wh_flag)
    {
        CUDACHECK(cudaMalloc((void **)d_weight, num_ch * batch * nseg * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)d_tmp_weight, batch * nseg * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)d_tmp, batch * nseg * sizeof(float)));
    }
    else if (do_runabs || do_runabs_mf)
    {
        CUDACHECK(cudaMalloc((void **)d_weight, num_ch * batch * nseg * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)d_tmp_weight, batch * nseg * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)d_tmp, batch * nseg * sizeof(float)));
        if (do_runabs_mf)
        {
            CUDACHECK(cudaMalloc((void **)d_filtered_sacdata,
                                 num_ch * batch * nseg * sizeof(float)));
            CUDACHECK(cudaMalloc((void **)d_total_sacdata,
                                 num_ch * batch * nseg * sizeof(float)));
        }
    }
}
