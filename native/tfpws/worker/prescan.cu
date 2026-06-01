#include "prescan.hpp"

#include <climits>

#include "logger.h"
#include "tfpws_schedule.hpp"

int estimate_tfpws_group_shape(const std::vector<SourcePackRecord> &records,
                               int sub_stack_size,
                               unsigned *ngroups,
                               unsigned *nsamples)
{
    if (records.empty())
        return 1;

    long long max_data_bytes = 0;
    for (std::size_t i = 0; i < records.size(); ++i)
    {
        const long long data_bytes =
            records[i].record_bytes - (long long)sizeof(SACHEAD);
        if (data_bytes <= 0 || data_bytes % (long long)sizeof(float) != 0)
        {
            LOG_ERROR("invalid_sourcepack_record_bytes",
                      "record_path=\"%s\" record_offset=%lld bytes=%lld",
                      records[i].record_path.c_str(),
                      records[i].record_offset,
                      records[i].record_bytes);
            return 1;
        }
        if (data_bytes > max_data_bytes)
            max_data_bytes = data_bytes;
    }

    const unsigned sample_count =
        (unsigned)(max_data_bytes / (long long)sizeof(float));
    if (sample_count == 0 ||
        (long long)sample_count * (long long)sizeof(float) != max_data_bytes)
    {
        LOG_ERROR("sourcepack_record_too_large",
                  "max_data_bytes=%lld",
                  max_data_bytes);
        return 1;
    }

    const std::size_t group_sz =
        (sub_stack_size < 2) ? 1 : (std::size_t)sub_stack_size;
    const std::size_t group_count =
        (records.size() + group_sz - 1) / group_sz;
    if (group_count == 0 || group_count > (std::size_t)UINT_MAX)
    {
        LOG_ERROR("sourcepack_group_count_too_large",
                  "records=%zu sub_stack_size=%d group_count=%zu",
                  records.size(),
                  sub_stack_size,
                  group_count);
        return 1;
    }

    *ngroups = (unsigned)group_count;
    *nsamples = sample_count;
    return 0;
}

std::size_t estimate_tfpws_host_workspace_bytes(unsigned ngroups,
                                                unsigned nsamples)
{
    const long double host_input =
        estimate_tfpws_host_input_bytes(ngroups, nsamples);
    const long double read_buffer = 4.0L * (long double)nsamples;
    return (std::size_t)(host_input + read_buffer);
}

static int build_tfpws_worker_workspace_plan(const GpuWorkerConfig &gpu_config,
                                             const ARGUTYPE *argument,
                                             TfpwsWorkerWorkspacePlan *worker_plan)
{
    if (!worker_plan->has_work)
        return 0;

    if (worker_plan->max_host_workspace_bytes > gpu_config.memory_budget_bytes)
    {
        LOG_ERROR("host_workspace_exceeds_worker_budget",
                  "worker_index=%zu gpu_id=%d max_host_workspace_gib=%.3f worker_budget_gib=%.3f hint=\"increase -M or -B\"",
                  gpu_config.worker_index,
                  gpu_config.device_id,
                  bytes_to_gib((long double)worker_plan->max_host_workspace_bytes),
                  bytes_to_gib((long double)gpu_config.memory_budget_bytes));
        return 1;
    }

    const unsigned nsamples = worker_plan->device.nsamples;
    const unsigned max_ngroups = worker_plan->device.max_ngroups;
    const std::size_t nfreq = (std::size_t)nsamples / 2 + 1;
    const TfpwsChunkPlan chunk_plan =
        select_tfpws_resident_chunk_plan(gpu_config.memory_budget_bytes,
                                         max_ngroups,
                                         nsamples,
                                         nfreq);
    if (!chunk_plan.plan_valid || !chunk_plan.fits_budget)
    {
        LOG_ERROR("tfpws_resident_workspace_plan_failed",
                  "worker_index=%zu gpu_id=%d max_groups=%u samples=%u planned_peak_gib=%.3f budget_gib=%.3f plan_valid=%s budget_fit=%s hint=\"increase -M or -B\"",
                  gpu_config.worker_index,
                  gpu_config.device_id,
                  max_ngroups,
                  nsamples,
                  bytes_to_gib(chunk_plan.planned_peak_bytes),
                  bytes_to_gib((long double)gpu_config.memory_budget_bytes),
                  chunk_plan.plan_valid ? "yes" : "no",
                  chunk_plan.fits_budget ? "yes" : "no");
        return 1;
    }

    worker_plan->device.nfreq = nfreq;
    worker_plan->device.freq_chunk_size = chunk_plan.freq_chunk_size;
    worker_plan->device.num_freq_chunks = chunk_plan.num_freq_chunks;
    worker_plan->device.resident_data_bytes = chunk_plan.resident_data_bytes;
    worker_plan->device.cufft_workspace_bytes = chunk_plan.cufft_workspace_bytes;
    worker_plan->device.runtime_reserve_bytes = chunk_plan.runtime_reserve_bytes;
    worker_plan->device.planned_peak_bytes = chunk_plan.planned_peak_bytes;
    worker_plan->device.band_limited = argument->band_limited;

    LOG_INFO("tfpws_worker_workspace_plan",
             "worker_index=%zu gpu_id=%d groups=%zu max_groups=%u samples=%u fixed_freq_chunk=%zu chunks=%zu max_host_workspace_gib=%.3f resident_data_gib=%.3f cufft_workspace_gib=%.3f runtime_reserve_gib=%.3f planned_peak_gib=%.3f worker_budget_gib=%.3f",
             gpu_config.worker_index,
             gpu_config.device_id,
             worker_plan->group_count,
             max_ngroups,
             nsamples,
             worker_plan->device.freq_chunk_size,
             worker_plan->device.num_freq_chunks,
             bytes_to_gib((long double)worker_plan->max_host_workspace_bytes),
             bytes_to_gib(worker_plan->device.resident_data_bytes),
             bytes_to_gib((long double)worker_plan->device.cufft_workspace_bytes),
             bytes_to_gib((long double)worker_plan->device.runtime_reserve_bytes),
             bytes_to_gib(worker_plan->device.planned_peak_bytes),
             bytes_to_gib((long double)gpu_config.memory_budget_bytes));
    return 0;
}

int prescan_tfpws_inputs(const std::vector<std::string> &indexes,
                         const ARGUTYPE *argument,
                         const std::vector<GpuWorkerConfig> &gpu_configs,
                         TfpwsPrescan *prescan)
{
    prescan->total_groups = 0;
    prescan->nsamples = 0;
    prescan->workers.assign(gpu_configs.size(), TfpwsWorkerWorkspacePlan());

    SourcePackGroupReader reader(indexes);
    if (reader.open() != 0)
        return 1;

    std::vector<SourcePackRecord> group;
    std::size_t cursor = 0;
    while (reader.next_group(&group))
    {
        unsigned ngroups = 0;
        unsigned nsamples = 0;
        if (estimate_tfpws_group_shape(group,
                                       argument->sub_stack_size,
                                       &ngroups,
                                       &nsamples) != 0)
        {
            return 1;
        }
        if (prescan->nsamples == 0)
            prescan->nsamples = nsamples;
        else if (prescan->nsamples != nsamples)
        {
            LOG_ERROR("sourcepack_sample_count_not_constant",
                      "expected=%u actual=%u path_id=\"%s\" component_slot=%d",
                      prescan->nsamples,
                      nsamples,
                      group.empty() ? "" : group[0].path_id.c_str(),
                      group.empty() ? -1 : group[0].component_slot);
            return 1;
        }

        TfpwsWorkerWorkspacePlan &worker_plan =
            prescan->workers[cursor % gpu_configs.size()];
        worker_plan.has_work = true;
        worker_plan.group_count += 1;
        worker_plan.device.nsamples = nsamples;
        if (ngroups > worker_plan.device.max_ngroups)
            worker_plan.device.max_ngroups = ngroups;

        const std::size_t host_workspace_bytes =
            estimate_tfpws_host_workspace_bytes(ngroups, nsamples);
        if (host_workspace_bytes > worker_plan.max_host_workspace_bytes)
            worker_plan.max_host_workspace_bytes = host_workspace_bytes;

        ++prescan->total_groups;
        ++cursor;
    }

    if (prescan->total_groups == 0)
    {
        LOG_ERROR("sourcepack_group_empty", "index_count=%zu", indexes.size());
        return 1;
    }

    for (std::size_t i = 0; i < gpu_configs.size(); ++i)
    {
        if (build_tfpws_worker_workspace_plan(gpu_configs[i],
                                              argument,
                                              &prescan->workers[i]) != 0)
        {
            return 1;
        }
    }

    LOG_INFO("tfpws_prescan_done",
             "groups=%zu samples=%u workers=%zu",
             prescan->total_groups,
             prescan->nsamples,
             gpu_configs.size());
    return 0;
}
