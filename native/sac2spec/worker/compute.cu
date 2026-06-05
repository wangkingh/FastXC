#include "worker_runtime.hpp"
#include "cuda.util.cuh"
#include "sac2spec_plan.hpp"
#include "core/normalization.cuh"
#include "core/preprocess.cuh"
#include "core/spectrum.cuh"

int ComputeWorkerBatch(WorkerBatch *batch)
{
    GpuWorker *worker = batch->worker;
    WorkerHostSlot *slot = batch->slot;
    const Sac2SpecPlan *plan = worker->plan;
    size_t proc_file_cnt = batch->file_rows;
    size_t frame_batch = batch->frame_batch;
    size_t proc_cnt = batch->frame_rows;
    size_t plan_cnt = batch->plan_rows;

    CUDACHECK(cudaSetDevice(worker->gpu_id));
    if (slot == NULL)
    {
        LOG_ERROR("worker_slot_missing", "stage=compute gpu=%d", worker->gpu_id);
        return -1;
    }

    LOG_INFO("worker_batch_compute_start",
             "gpu=%d slot=%d start_group=%zu group_count=%zu frame_batch=%zu proc_cnt=%zu segment_pts=%d filter_nfft=%d output_nfft=%d",
             worker->gpu_id, batch->slot_index, batch->start_group, batch->group_count,
             frame_batch, proc_cnt, plan->segment_pts, plan->filter_nfft,
             plan->output_nfft);

    CUDACHECK(cudaMemset(worker->d_sacdata, 0, plan_cnt * plan->segment_pts * sizeof(float)));
    CUDACHECK(cudaMemset(worker->d_spectrum, 0, plan_cnt * plan->segment_pts * sizeof(cuComplex)));

    for (int out_step = 0; out_step < plan->nstep_valid; out_step++)
    {
        int stepidx = plan->valid_steps[out_step];
        size_t src_offset = (size_t)stepidx * (size_t)plan->shift_pts;
        float *dst = worker->d_sacdata
                     + (size_t)out_step * proc_file_cnt * (size_t)plan->segment_pts;
        const float *src = slot->h_sacdata + src_offset;
        CUDACHECK(cudaMemcpy2D(dst, plan->segment_pts * sizeof(float),
                               src, plan->npts * sizeof(float),
                               plan->segment_pts * sizeof(float), proc_file_cnt,
                               cudaMemcpyHostToDevice));
    }

    LOG_INFO("worker_h2d_windowed_copy_done",
             "gpu=%d slot=%d start_group=%zu valid_steps=%d file_rows=%zu",
             worker->gpu_id, batch->slot_index, batch->start_group,
             plan->nstep_valid, proc_file_cnt);

    if (preprocess(worker->d_sacdata, worker->d_sum, worker->d_isum,
                                   plan->segment_pts, proc_cnt,
                                   plan->freq_low, plan->delta) != 0)
    {
        return -1;
    }

    CUDACHECK(cudaMemset(worker->d_padded_sacdata, 0, plan_cnt * plan->filter_nfft * sizeof(float)));
    CUDACHECK(cudaMemset(worker->d_padded_spectrum, 0, plan_cnt * plan->filter_nfft * sizeof(cuComplex)));
    CUDACHECK(cudaMemcpy2D(worker->d_padded_sacdata, plan->filter_nfft * sizeof(float),
                           worker->d_sacdata, plan->segment_pts * sizeof(float),
                           plan->segment_pts * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice));
    CUFFTCHECK(cufftExecR2C(worker->planfwd_filter, (cufftReal *)worker->d_padded_sacdata,
                            (cufftComplex *)worker->d_padded_spectrum));
    size_t fwidth_filter = plan->filter_nfft / 2 + 1;
    if (fft_forward_normalize(worker->d_padded_spectrum, plan->filter_nfft,
                                          fwidth_filter, proc_cnt, plan->delta) != 0 ||
        complex_sanitize(worker->d_padded_spectrum, plan->filter_nfft,
                                         fwidth_filter, proc_cnt) != 0 ||
        apply_filter_response(worker->d_padded_spectrum, worker->d_responses,
                                              plan->filter_nfft, fwidth_filter, proc_cnt) != 0)
    {
        return -1;
    }
    CUFFTCHECK(cufftExecC2R(worker->planinv_filter, (cufftComplex *)worker->d_padded_spectrum,
                            (cufftReal *)worker->d_padded_sacdata));
    if (fft_inverse_normalize(worker->d_padded_sacdata, plan->filter_nfft,
                                          plan->filter_nfft, proc_cnt, plan->delta) != 0)
    {
        return -1;
    }

    CUDACHECK(cudaMemcpy2D(worker->d_sacdata, plan->segment_pts * sizeof(float),
                           worker->d_padded_sacdata, plan->filter_nfft * sizeof(float),
                           plan->segment_pts * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice));

    LOG_INFO("worker_prefilter_done",
             "gpu=%d slot=%d start_group=%zu proc_cnt=%zu filter_nfft=%d",
             worker->gpu_id, batch->slot_index, batch->start_group,
             proc_cnt, plan->filter_nfft);

    if (plan->wh_before)
    {
        CUFFTCHECK(cufftExecR2C(worker->planfwd, (cufftReal *)worker->d_sacdata,
                                (cufftComplex *)worker->d_spectrum));
        if (fft_forward_normalize(worker->d_spectrum, plan->segment_pts,
                                              plan->segment_pts, proc_cnt, plan->delta) != 0)
        {
            return -1;
        }
        if (freq_whiten(worker->d_spectrum, worker->d_weight,
                                        worker->d_tmp_weight, worker->d_tmp,
                                        plan->num_ch, plan->segment_pts,
                                        (int)frame_batch, plan->delta,
                                        plan->f_idx1, plan->f_idx2,
                                        plan->f_idx3, plan->f_idx4) != 0)
        {
            return -1;
        }
        if (complex_sanitize(worker->d_spectrum, plan->segment_pts,
                                             plan->segment_pts, proc_cnt) != 0)
        {
            return -1;
        }
        CUFFTCHECK(cufftExecC2R(worker->planinv, (cufftComplex *)worker->d_spectrum,
                                (cufftReal *)worker->d_sacdata));
        if (fft_inverse_normalize(worker->d_sacdata, plan->segment_pts,
                                              plan->segment_pts, proc_cnt, plan->delta) != 0)
        {
            return -1;
        }
    }

    if (plan->wh_before)
    {
        LOG_INFO("worker_whiten_before_done",
                 "gpu=%d slot=%d start_group=%zu frame_batch=%zu",
                 worker->gpu_id, batch->slot_index, batch->start_group, frame_batch);
    }

    if (plan->do_runabs_mf)
    {
        if (runabs_mf(worker->d_sacdata, worker->d_filtered_sacdata,
                                      worker->d_total_sacdata,
                                      worker->d_padded_sacdata, worker->d_padded_spectrum,
                                      worker->d_base_padded_spectrum,
                                      worker->d_responses, worker->d_tmp,
                                      worker->d_weight, worker->d_tmp_weight,
                                      plan->freq_lows, plan->filter_count,
                                      plan->delta, (int)frame_batch,
                                      plan->num_ch, MAXVAL, plan->segment_pts,
                                      plan->filter_nfft,
                                      (int)worker->frame_capacity,
                                      &worker->planinv_filter,
                                      &worker->planfwd_filter) != 0)
        {
            return -1;
        }
        CUDACHECK(cudaMemcpy2D(worker->d_sacdata, plan->segment_pts * sizeof(float),
                               worker->d_total_sacdata, plan->segment_pts * sizeof(float),
                               plan->segment_pts * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice));
        LOG_INFO("worker_runabs_mf_done",
                 "gpu=%d slot=%d start_group=%zu frame_batch=%zu filter_count=%d",
                 worker->gpu_id, batch->slot_index, batch->start_group,
                 frame_batch, plan->filter_count);
    }

    if (plan->do_onebit)
    {
        if (time_onebit(worker->d_sacdata, plan->segment_pts,
                                        plan->segment_pts, proc_cnt) != 0)
        {
            return -1;
        }
        LOG_INFO("worker_onebit_done",
                 "gpu=%d slot=%d start_group=%zu proc_cnt=%zu",
                 worker->gpu_id, batch->slot_index, batch->start_group, proc_cnt);
    }

    if (plan->do_runabs)
    {
        float freq_lows_limit = plan->freq_low * 0.667f;
        if (runabs(worker->d_sacdata, worker->d_tmp,
                                   worker->d_weight, worker->d_tmp_weight,
                                   freq_lows_limit, plan->delta,
                                   (int)frame_batch, plan->num_ch,
                                   plan->segment_pts, MAXVAL) != 0)
        {
            return -1;
        }
        LOG_INFO("worker_runabs_done",
                 "gpu=%d slot=%d start_group=%zu frame_batch=%zu",
                 worker->gpu_id, batch->slot_index, batch->start_group, frame_batch);
    }

    if (plan->wh_after)
    {
        CUFFTCHECK(cufftExecR2C(worker->planfwd, (cufftReal *)worker->d_sacdata,
                                (cufftComplex *)worker->d_spectrum));
        if (fft_forward_normalize(worker->d_spectrum, plan->segment_pts,
                                              plan->segment_pts, proc_cnt, plan->delta) != 0)
        {
            return -1;
        }
        if (freq_whiten(worker->d_spectrum, worker->d_weight,
                                        worker->d_tmp_weight, worker->d_tmp,
                                        plan->num_ch, plan->segment_pts,
                                        (int)frame_batch, plan->delta,
                                        plan->f_idx1, plan->f_idx2,
                                        plan->f_idx3, plan->f_idx4) != 0)
        {
            return -1;
        }
        if (complex_sanitize(worker->d_spectrum, plan->segment_pts,
                                             plan->segment_pts, proc_cnt) != 0)
        {
            return -1;
        }
        CUFFTCHECK(cufftExecC2R(worker->planinv, (cufftComplex *)worker->d_spectrum,
                                (cufftReal *)worker->d_sacdata));
        if (fft_inverse_normalize(worker->d_sacdata, plan->segment_pts,
                                              plan->segment_pts, proc_cnt, plan->delta) != 0)
        {
            return -1;
        }
    }

    if (plan->wh_after)
    {
        LOG_INFO("worker_whiten_after_done",
                 "gpu=%d slot=%d start_group=%zu frame_batch=%zu",
                 worker->gpu_id, batch->slot_index, batch->start_group, frame_batch);
    }

    CUDACHECK(cudaMemset(worker->d_padded_sacdata, 0, plan_cnt * plan->output_nfft * sizeof(float)));
    CUDACHECK(cudaMemset(worker->d_padded_spectrum, 0, plan_cnt * plan->output_nfft * sizeof(cuComplex)));
    CUDACHECK(cudaMemcpy2D(worker->d_padded_sacdata, plan->output_nfft * sizeof(float),
                           worker->d_sacdata, plan->segment_pts * sizeof(float),
                           plan->segment_pts * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice));
    CUFFTCHECK(cufftExecR2C(worker->planfwd_output, (cufftReal *)worker->d_padded_sacdata,
                            (cufftComplex *)worker->d_padded_spectrum));
    if (fft_forward_normalize(worker->d_padded_spectrum, plan->output_nfft,
                                          plan->nspec_output, proc_cnt, plan->delta) != 0 ||
        complex_sanitize(worker->d_padded_spectrum, plan->output_nfft,
                                         plan->nspec_output, proc_cnt) != 0)
    {
        return -1;
    }
    if (plan->output_phase_only)
    {
        if (phase_only(worker->d_padded_spectrum, plan->output_nfft,
                                       plan->nspec_output, proc_cnt, MINVAL) != 0)
        {
            return -1;
        }
        if (complex_sanitize(worker->d_padded_spectrum, plan->output_nfft,
                                             plan->nspec_output, proc_cnt) != 0)
        {
            return -1;
        }
    }

    LOG_INFO("worker_output_fft_done",
             "gpu=%d slot=%d start_group=%zu nspec=%d phase_only=%d",
             worker->gpu_id, batch->slot_index, batch->start_group, plan->nspec_output,
             plan->output_phase_only);

    CUDACHECK(cudaMemcpy2D(slot->h_spectrum,
                           plan->nspec_output * sizeof(complex),
                           worker->d_padded_spectrum,
                           plan->output_nfft * sizeof(cuComplex),
                           plan->nspec_output * sizeof(complex), proc_cnt,
                           cudaMemcpyDeviceToHost));

    LOG_INFO("worker_d2h_spectrum_done",
             "gpu=%d slot=%d start_group=%zu group_count=%zu file_rows=%zu nstep_valid=%d nspec=%d",
             worker->gpu_id, batch->slot_index, batch->start_group, batch->group_count,
             batch->file_rows, plan->nstep_valid, plan->nspec_output);

    return 0;
}
