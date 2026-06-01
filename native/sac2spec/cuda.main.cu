/* last updated by wangjx@20250421 */

#include "include/device_memory.cuh"
#include "include/output_layout.hpp"
#include "include/progress.hpp"
#include "include/sac2spec_plan.hpp"
#include "include/task_queue.hpp"
#include "include/timestamp_tracker.hpp"
#include "include/worker_runtime.hpp"

#include <limits.h>
#include <math.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C"
{
#include "arguproc.h"
#include "fft_length.h"
#include "core/filter_response.h"
#include "io/sac_index.h"
#include "include/sac.h"
#include "include/util.h"
}

static int WriteSuccessMarker(const char *path)
{
    FILE *fp = fopen(path, "wb");
    if (fp == NULL)
    {
        LOG_ERROR("write_success_marker_failed", "path=\"%s\"", path);
        return -1;
    }
    fprintf(fp, "DONE\n");
    if (fclose(fp) != 0)
    {
        LOG_ERROR("close_success_marker_failed", "path=\"%s\"", path);
        return -1;
    }
    return 0;
}

static int SpackRootHasPreviousOutputs(const char *path)
{
    DIR *dir = opendir(path);
    struct dirent *entry;
    if (dir == NULL)
    {
        LOG_ERROR("open_spack_dir_failed", "path=\"%s\"", path);
        return -1;
    }

    while ((entry = readdir(dir)) != NULL)
    {
        const char *name = entry->d_name;
        if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
        {
            continue;
        }
        LOG_ERROR("spack_dir_not_empty",
                  "path=\"%s\" entry=\"%s\" action=clean_or_choose_new_output",
                  path, name);
        closedir(dir);
        return 1;
    }

    closedir(dir);
    return 0;
}

int main(int argc, char **argv)
{
    ARGUTYPE argument;
    ArgumentProcess(argc, argv, &argument);

    OutputLayout output;
    if (InitOutputLayout(argument.output_root, &output) != 0)
    {
        return EXIT_FAILURE;
    }
    int spack_state = SpackRootHasPreviousOutputs(output.spack_root.c_str());
    if (spack_state != 0)
    {
        return EXIT_FAILURE;
    }

    LOG_INFO("sac2spec_run_start",
             "input=\"%s\" output_root=\"%s\" spack_by_timestamp_root=\"%s\" progress_file=\"%s\" sac_len_sec=%.8g num_ch=%d gpu_worker_count=%d threads=%d lazy_async=%d gpu_ram_limit_count=%d",
             argument.input_list, output.root.c_str(),
             output.spack_root.c_str(), output.progress_file.c_str(),
             argument.sac_len_sec,
             argument.num_ch, argument.gpu_worker_count, argument.thread_num,
             argument.lazy_async, argument.gpu_ram_limit_count);

    SacIndexPaths sacIndexPaths = readSacIndexPaths(argument.input_list,
                                                    argument.num_ch);
    FilePathArray InPaths = sacIndexPaths.in_paths;
    SacIndexMetaArray MetaRows = sacIndexPaths.meta;
    LOG_INFO("sac_index_paths_ready",
             "input_count=%d meta_count=%d first_input=\"%s\" first_timestamp=\"%s\" first_nsl_id=%04d",
             InPaths.count, MetaRows.count,
             InPaths.count > 0 ? InPaths.paths[0] : "",
             MetaRows.count > 0 ? MetaRows.values[0].timestamp : "",
             MetaRows.count > 0 ? MetaRows.values[0].gnsl_id : -1);

    int num_ch = argument.num_ch;
    int wh_before = 0, wh_after = 0, do_runabs_mf = 0, do_runabs = 0, do_onebit = 0;
    switch (argument.whitenType)
    {
    case WHITEN_NONE:
        wh_before = 0, wh_after = 0;
        break;
    case WHITEN_BEFORE_NORMALIZE:
        wh_before = 1;
        break;
    case WHITEN_AFTER_NORMALIZE:
        wh_after = 1;
        break;
    case WHITEN_BEFORE_AND_AFTER:
        wh_before = 1, wh_after = 1;
        break;
    default:
        LOG_ERROR("invalid_whiten_type", "value=%d", argument.whitenType);
        exit(EXIT_FAILURE);
    }
    if (argument.outputPhaseOnly != OUTPUT_KEEP_AMPLITUDE &&
        argument.outputPhaseOnly != OUTPUT_PHASE_ONLY)
    {
        LOG_ERROR("invalid_output_phase_only", "value=%d", argument.outputPhaseOnly);
        exit(EXIT_FAILURE);
    }
    LOG_INFO("normalization_plan",
             "whiten_type=%d output_phase_only=%d normalize_type=%d",
             argument.whitenType, argument.outputPhaseOnly, argument.normalizeType);

    switch (argument.normalizeType)
    {
    case NORMALIZE_NONE:
        do_runabs_mf = 0, do_onebit = 0, do_runabs = 0;
        break;
    case NORMALIZE_RUNABS_MF:
        do_runabs_mf = 1;
        break;
    case NORMALIZE_ONEBIT:
        do_onebit = 1;
        break;
    case NORMALIZE_RUNABS:
        do_runabs = 1;
        break;
    default:
        LOG_ERROR("invalid_normalize_type", "value=%d", argument.normalizeType);
        exit(EXIT_FAILURE);
    }

    if (InPaths.count != MetaRows.count)
    {
        LOG_ERROR("sac_index_meta_count_mismatch", "input_count=%d meta_count=%d",
                  InPaths.count, MetaRows.count);
        exit(EXIT_FAILURE);
    }
    if (InPaths.count < 1)
    {
        LOG_ERROR("sac_index_no_input", "input_count=%d", InPaths.count);
        exit(EXIT_FAILURE);
    }
    if (InPaths.count % num_ch != 0)
    {
        LOG_ERROR("sac_index_component_count_invalid",
                  "component_count=%d num_ch=%d", InPaths.count, num_ch);
        exit(EXIT_FAILURE);
    }

    int device_count = 0;
    CUDACHECK(cudaGetDeviceCount(&device_count));
    for (int i = 0; i < argument.gpu_worker_count; i++)
    {
        if (argument.gpu_ids[i] >= device_count)
        {
            LOG_ERROR("gpu_id_out_of_range", "gpu=%d device_count=%d",
                      argument.gpu_ids[i], device_count);
            exit(EXIT_FAILURE);
        }
    }

    size_t nValid_sacnum = InPaths.count;
    size_t nValid_batch = nValid_sacnum / (size_t)num_ch;

    SACHEAD sachd;
    if (read_sachead(InPaths.paths[0], &sachd) != 0)
    {
        LOG_ERROR("read_first_sachead_failed", "path=\"%s\"", InPaths.paths[0]);
        exit(EXIT_FAILURE);
    }
    float delta = sachd.delta;
    if (delta <= 0.0f)
    {
        LOG_ERROR("invalid_first_sac_delta",
                  "path=\"%s\" delta=%.9g", InPaths.paths[0], delta);
        exit(EXIT_FAILURE);
    }
    double target_samples = (double)argument.sac_len_sec / (double)delta;
    if (!isfinite(target_samples) || target_samples < 1.0 ||
        target_samples > (double)INT_MAX)
    {
        LOG_ERROR("invalid_sac_len",
                  "sac_len_sec=%.9g delta=%.9g samples=%.9g",
                  argument.sac_len_sec, delta, target_samples);
        exit(EXIT_FAILURE);
    }
    int npts = (int)llround(target_samples);
    if (npts < 1)
    {
        LOG_ERROR("invalid_target_npts",
                  "sac_len_sec=%.9g delta=%.9g npts=%d",
                  argument.sac_len_sec, delta, npts);
        exit(EXIT_FAILURE);
    }
    LOG_INFO("sac_read_contract",
             "sac_len_sec=%.8g delta=%.8g first_header_npts=%d target_npts=%d",
             argument.sac_len_sec, delta, sachd.npts, npts);

    int segment_pts_1x = segment_length_from_seconds(argument.seglen, npts, delta);

    int shift_length_pts = (int)llround((double)argument.segshift / (double)delta);
    if (shift_length_pts < 1)
        shift_length_pts = segment_pts_1x;
    int nstep = 0;
    if (segment_pts_1x <= npts)
    {
        nstep = (npts - segment_pts_1x) / shift_length_pts + 1;
        if (nstep < 1)
            nstep = 1;
    }
    else
    {
        nstep = 1;
    }
    LOG_INFO("segment_plan", "npts=%d delta=%.8g segment_pts=%d shift_pts=%d nstep=%d",
             npts, delta, segment_pts_1x, shift_length_pts, nstep);

    int max_output_nfft = segment_pts_1x * 2;
    int xcorr_keep_pts = (int)llround((double)argument.xcorr_lag_sec / (double)delta);
    if (xcorr_keep_pts > segment_pts_1x)
    {
        LOG_WARN("xcorr_keep_clamped",
                 "requested_sec=%.8g requested_pts=%d segment_pts=%d effective_pts=%d",
                 argument.xcorr_lag_sec, xcorr_keep_pts, segment_pts_1x, segment_pts_1x);
        xcorr_keep_pts = segment_pts_1x;
    }
    int output_nfft = next_even_smooth_2357_length(segment_pts_1x + xcorr_keep_pts);
    if (output_nfft > max_output_nfft)
    {
        output_nfft = max_output_nfft;
    }
    int nspec_output = output_nfft / 2 + 1;

    float df_1x = 1.0f / (segment_pts_1x * delta);
    float df_output = 1.0f / (output_nfft * delta);
    LOG_INFO("output_fft_plan",
             "segment_pts=%d xcorr_keep_sec=%.8g xcorr_keep_pts=%d output_nfft=%d nspec=%d df=%.8g",
             segment_pts_1x, argument.xcorr_lag_sec, xcorr_keep_pts,
             output_nfft, nspec_output, df_output);

    int filter_count = 0;
    ButterworthFilter *filter = readButterworthFilters(argument.filter_file, &filter_count);
    if (filter == NULL || filter_count <= 0)
    {
        LOG_ERROR("filter_file_empty", "path=\"%s\" filter_count=%d", argument.filter_file, filter_count);
        exit(EXIT_FAILURE);
    }
    float freq_low = filter[0].freq_low;
    float freq_high = filter[0].freq_high;
    if (freq_low <= 0.0f || freq_high <= freq_low)
    {
        LOG_ERROR("filter_broad_band_invalid",
                  "path=\"%s\" freq_low=%.8g freq_high=%.8g",
                  argument.filter_file, freq_low, freq_high);
        exit(EXIT_FAILURE);
    }
    int f_idx1 = int(freq_low * 0.667f / df_1x);
    int f_idx2 = int(freq_low / df_1x);
    int f_idx3 = int(freq_high / df_1x);
    int f_idx4 = int(freq_high * 1.333f / df_1x);
    int filter_padding_pts = estimateButterworthFilterPadding(filter, filter_count, segment_pts_1x);
    int filter_nfft = next_even_smooth_2357_length(segment_pts_1x + filter_padding_pts);
    if (filter_nfft > max_output_nfft)
    {
        filter_nfft = max_output_nfft;
    }
    float df_filter = 1.0f / (filter_nfft * delta);
    LOG_INFO("filter_padding_plan",
             "segment_pts=%d filter_padding_pts=%d filter_nfft=%d output_nfft=%d max_output_nfft=%d",
             segment_pts_1x, filter_padding_pts, filter_nfft, output_nfft, max_output_nfft);
    FilterResp *myResp = processButterworthFilters(filter, filter_count, df_filter, filter_nfft);
    if (myResp == NULL)
    {
        LOG_ERROR("filter_response_build_failed", "path=\"%s\" filter_count=%d",
                  argument.filter_file, filter_count);
        exit(EXIT_FAILURE);
    }

    int *skip_flags = (int *)calloc((size_t)nstep, sizeof(int));
    if (skip_flags == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "bytes=%zu", (size_t)nstep * sizeof(int));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < argument.skip_step_count; ++i)
    {
        int skip_step = argument.skip_steps[i];
        if (skip_step >= 0 && skip_step < nstep)
        {
            skip_flags[skip_step] = 1;
            LOG_INFO("segment_step_skipped", "step=%d", skip_step);
        }
        else
        {
            LOG_WARN("segment_step_skip_out_of_range",
                     "step=%d nstep=%d ignored=1", skip_step, nstep);
        }
    }

    int nstep_valid = 0;
    int *valid_steps = (int *)malloc((size_t)nstep * sizeof(int));
    if (valid_steps == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "bytes=%zu", (size_t)nstep * sizeof(int));
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nstep; ++i)
    {
        if (!skip_flags[i])
        {
            valid_steps[nstep_valid++] = i;
        }
    }
    if (nstep_valid < 1)
    {
        LOG_ERROR("no_valid_segment_steps", "nstep=%d skip_count=%d",
                  nstep, argument.skip_step_count);
        exit(EXIT_FAILURE);
    }
    LOG_INFO("valid_segment_steps", "nstep_total=%d nstep_valid=%d skip_count=%d",
             nstep, nstep_valid, argument.skip_step_count);

    float *freq_lows = (float *)malloc((size_t)filter_count * sizeof(float));
    if (freq_lows == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "bytes=%zu", (size_t)filter_count * sizeof(float));
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < filter_count; i++)
    {
        freq_lows[i] = myResp[i].freq_low;
    }

    size_t unit_sacdata_size = (size_t)npts * sizeof(float);
    size_t unit_spectrum_size = (size_t)nstep_valid * (size_t)nspec_output * sizeof(complex);
    size_t unit_InOutNode_size = sizeof(InOutNode);

    LOG_INFO("cpu_memory_unit",
             "sacdata_mb=%zu spectrum_mb=%zu inoutnode_mb=%zu",
             unit_sacdata_size / 1024 / 1024,
             unit_spectrum_size / 1024 / 1024,
             unit_InOutNode_size / 1024 / 1024);

    int host_slot_count = argument.lazy_async ? 3 : 1;
    size_t unitCpuRamPerSlot = (size_t)num_ch * (unit_sacdata_size +
                                                unit_spectrum_size +
                                                unit_InOutNode_size);
    size_t unitCpuRam = unitCpuRamPerSlot * (size_t)host_slot_count;
    LOG_INFO("cpu_memory_slot_plan",
             "lazy_async=%d host_slots=%d unit_per_slot_mb=%zu unit_effective_mb=%zu",
             argument.lazy_async, host_slot_count,
             unitCpuRamPerSlot / 1024 / 1024,
             unitCpuRam / 1024 / 1024);

    size_t h_batch_total = EstimateCpuBatch(unitCpuRam, 1);
    h_batch_total = h_batch_total < nValid_batch ? h_batch_total : nValid_batch;
    LOG_INFO("cpu_batch_estimated", "h_batch_total=%zu valid_batch=%zu",
             h_batch_total, nValid_batch);
    if (h_batch_total < 1)
    {
        LOG_ERROR("cpu_batch_unavailable", "h_batch_total=%zu valid_batch=%zu",
                  h_batch_total, nValid_batch);
        exit(EXIT_FAILURE);
    }

    Sac2SpecPlan plan;
    memset(&plan, 0, sizeof(plan));
    plan.in_paths = InPaths;
    plan.meta = MetaRows;
    plan.num_ch = num_ch;
    plan.spack_root = output.spack_root.c_str();
    plan.npts = npts;
    plan.delta = delta;
    plan.segment_pts = segment_pts_1x;
    plan.shift_pts = shift_length_pts;
    plan.nstep_valid = nstep_valid;
    plan.output_nfft = output_nfft;
    plan.nspec_output = nspec_output;
    plan.filter_nfft = filter_nfft;
    plan.filter_count = filter_count;
    plan.freq_low = freq_low;
    plan.f_idx1 = f_idx1;
    plan.f_idx2 = f_idx2;
    plan.f_idx3 = f_idx3;
    plan.f_idx4 = f_idx4;
    plan.df_output = df_output;
    plan.wh_before = wh_before;
    plan.wh_after = wh_after;
    plan.output_phase_only = argument.outputPhaseOnly;
    plan.do_runabs_mf = do_runabs_mf;
    plan.do_runabs = do_runabs;
    plan.do_onebit = do_onebit;
    plan.lazy_async = argument.lazy_async;
    plan.host_slot_count = host_slot_count;
    plan.wh_flag = (size_t)(wh_before || wh_after);
    plan.valid_steps = valid_steps;
    plan.filter_responses = myResp;
    plan.freq_lows = freq_lows;
    plan.unit_sacdata_size = unit_sacdata_size;
    plan.unit_spectrum_size = unit_spectrum_size;
    plan.unit_InOutNode_size = unit_InOutNode_size;

    ProgressState progress;
    ProgressInit(&progress, output.progress_file.c_str(), nValid_sacnum);
    plan.progress = &progress;

    TimestampTracker timestamp_tracker;
    memset(&timestamp_tracker, 0, sizeof(timestamp_tracker));
    if (TimestampTrackerInit(&timestamp_tracker, &MetaRows, num_ch,
                             nValid_batch, output.spack_root.c_str()) != 0)
    {
        LOG_ERROR("timestamp_tracker_init_failed", "groups=%zu num_ch=%d",
                  nValid_batch, num_ch);
        exit(EXIT_FAILURE);
    }
    plan.timestamp_tracker = &timestamp_tracker;

    WorkerCapacityPlan worker_plans[MAX_GPU_DEVICES];
    memset(worker_plans, 0, sizeof(worker_plans));
    for (int i = 0; i < argument.gpu_worker_count; i++)
    {
        double worker_gpu_ram_limit_mib = argument.gpu_ram_limits_mib[i];
        LOG_INFO("virtual_gpu_worker_plan",
                 "worker=%d gpu=%d gpu_ram_limit_mib=%.3f",
                 i, argument.gpu_ids[i], worker_gpu_ram_limit_mib);
        worker_plans[i].gpu_id = argument.gpu_ids[i];
        worker_plans[i].gpu_frame_cap = EstimateGpuFrameBatch((size_t)argument.gpu_ids[i],
                                                              segment_pts_1x,
                                                              filter_nfft, output_nfft,
                                                              num_ch, filter_count,
                                                              plan.wh_flag, (size_t)do_runabs,
                                                              (size_t)do_runabs_mf,
                                                              worker_gpu_ram_limit_mib);
        if (worker_plans[i].gpu_frame_cap < (size_t)nstep_valid)
        {
            LOG_WARN("gpu_frame_capacity_too_small",
                     "gpu=%d gpu_frame_cap=%zu nstep_valid=%d disabled=1",
                     argument.gpu_ids[i], worker_plans[i].gpu_frame_cap, nstep_valid);
            worker_plans[i].gpu_cap = 0;
        }
        else
        {
            worker_plans[i].gpu_cap = worker_plans[i].gpu_frame_cap / (size_t)nstep_valid;
            worker_plans[i].gpu_cap = worker_plans[i].gpu_cap < nValid_batch
                                           ? worker_plans[i].gpu_cap
                                           : nValid_batch;
        }
    }

    int active_workers = PlanWorkerCapacities(worker_plans, argument.gpu_worker_count,
                                              nValid_batch, h_batch_total,
                                              argument.thread_num);
    if (active_workers < 1)
    {
        LOG_ERROR("gpu_worker_unavailable", "active_workers=%d valid_batch=%zu host_batch=%zu",
                  active_workers, nValid_batch, h_batch_total);
        exit(EXIT_FAILURE);
    }

    TaskQueue queue;
    TaskQueueInit(&queue, nValid_batch);

    int failed = 0;
    GpuWorkerRuntime worker_runtime;

    // Async worker lifecycle: queue -> start virtual GPU workers -> wait -> cleanup.
    if (InitGpuWorkerRuntime(&worker_runtime, &plan, &queue,
                             worker_plans, active_workers) != 0)
    {
        failed = 1;
    }
    else if (StartGpuWorkerThreads(&worker_runtime) != 0)
    {
        failed = 1;
        if (JoinGpuWorkerThreads(&worker_runtime) != 0)
        {
            failed = 1;
        }
    }
    else if (JoinGpuWorkerThreads(&worker_runtime) != 0)
    {
        failed = 1;
    }

    DestroyGpuWorkerRuntime(&worker_runtime);
    TaskQueueDestroy(&queue);

    if (!failed && WriteSuccessMarker(output.spack_success_file.c_str()) != 0)
    {
        failed = 1;
    }

    if (failed)
    {
        ProgressFinish(&progress, "FAILED", 0);
        ProgressDestroy(&progress);
        TimestampTrackerDestroy(&timestamp_tracker);
        LOG_ERROR("gpu_worker_failed", "active_workers=%d", active_workers);
        exit(EXIT_FAILURE);
    }

    ProgressFinish(&progress, "DONE", 1);
    ProgressDestroy(&progress);
    TimestampTrackerDestroy(&timestamp_tracker);

    for (int i = 0; i < filter_count; i++)
    {
        free(myResp[i].response);
    }
    free(myResp);
    free(filter);
    free(freq_lows);
    free(valid_steps);
    free(skip_flags);

    return 0;
}
