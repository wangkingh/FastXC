#include "pipeline.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "host_workspace_budget.hpp"
#include "logger.h"
#include "progress_sidecar.hpp"
#include "sourcepack_io.hpp"
#include "prescan.hpp"
#include "tfpws_compute.hpp"
#include "tfpws_schedule.hpp"
#include "work_queue.hpp"

struct TfpwsJob
{
    std::vector<SourcePackRecord> records;
};

struct TfpwsWorkerContext
{
    std::size_t worker_index;
    int gpu_id;
    std::size_t memory_budget_bytes;
    const TfpwsWorkerWorkspacePlan *workspace_plan;
    int sub_stack_size;
    std::string output_dir;
    const ARGUTYPE *argument;
    WorkQueue<TfpwsJob> *queue;
    HostWorkspaceBudget *host_budget;
    std::atomic<int> *failed;
    std::atomic<std::size_t> *completed;
    FastxcProgressSidecar *progress;
};

static void tfpws_worker_main(TfpwsWorkerContext *ctx)
{
    CUDACHECK(cudaSetDevice(ctx->gpu_id));

    TfpwsDeviceWorkspace device_workspace;
    init_tfpws_device_workspace_struct(&device_workspace);
    if (ctx->workspace_plan && ctx->workspace_plan->has_work)
    {
        if (allocate_tfpws_device_workspace(&device_workspace,
                                            &ctx->workspace_plan->device,
                                            ctx->worker_index,
                                            ctx->gpu_id) != 0)
        {
            ++(*ctx->failed);
            return;
        }
    }

    TfpwsSourcePackShardWriter writer;
    if (writer.open(ctx->output_dir, ctx->worker_index) != 0)
    {
        free_tfpws_device_workspace(&device_workspace);
        ++(*ctx->failed);
        return;
    }

    PackFileCache cache;
    TfpwsJob job;
    while (ctx->queue->pop(job))
    {
        if (job.records.empty())
            continue;

        unsigned planned_ngroups = 0;
        unsigned planned_nsamples = 0;
        if (estimate_tfpws_group_shape(job.records,
                                       ctx->sub_stack_size,
                                       &planned_ngroups,
                                       &planned_nsamples) != 0)
        {
            ++(*ctx->failed);
            continue;
        }

        if (!ctx->workspace_plan || !ctx->workspace_plan->has_work ||
            planned_nsamples != ctx->workspace_plan->device.nsamples ||
            planned_ngroups > ctx->workspace_plan->device.max_ngroups)
        {
            LOG_ERROR("tfpws_worker_workspace_shape_mismatch",
                      "worker_index=%zu gpu_id=%d groups=%u max_groups=%u samples=%u workspace_samples=%u",
                      ctx->worker_index,
                      ctx->gpu_id,
                      planned_ngroups,
                      ctx->workspace_plan ? ctx->workspace_plan->device.max_ngroups : 0,
                      planned_nsamples,
                      ctx->workspace_plan ? ctx->workspace_plan->device.nsamples : 0);
            ++(*ctx->failed);
            continue;
        }

        const std::size_t host_workspace_bytes =
            estimate_tfpws_host_workspace_bytes(planned_ngroups,
                                                planned_nsamples);
        if (host_workspace_bytes > ctx->memory_budget_bytes)
        {
            LOG_ERROR("host_workspace_exceeds_worker_budget",
                      "worker_index=%zu gpu_id=%d host_workspace_gib=%.3f worker_budget_gib=%.3f path_id=\"%s\" component_slot=%d hint=\"increase -M or -B\"",
                      ctx->worker_index,
                      ctx->gpu_id,
                      bytes_to_gib((long double)host_workspace_bytes),
                      bytes_to_gib((long double)ctx->memory_budget_bytes),
                      job.records[0].path_id.c_str(),
                      job.records[0].component_slot);
            ++(*ctx->failed);
            continue;
        }

        HostWorkspaceLease host_lease;
        if (!host_lease.acquire(ctx->host_budget, host_workspace_bytes))
        {
            LOG_ERROR("host_workspace_exceeds_global_budget",
                      "worker_index=%zu gpu_id=%d host_workspace_gib=%.3f",
                      ctx->worker_index,
                      ctx->gpu_id,
                      bytes_to_gib((long double)host_workspace_bytes));
            ++(*ctx->failed);
            continue;
        }

        TfpwsSourcePackItem item;
        if (prepare_tfpws_sourcepack_item(job.records,
                                          ctx->sub_stack_size,
                                          &cache,
                                          &item) != 0)
        {
            ++(*ctx->failed);
            continue;
        }

        SACHEAD out_header;
        float *out_data = NULL;
        int ret = compute_tfpws_from_prestack(item.label.c_str(),
                                              item.header,
                                              item.prestack_data,
                                              item.linear_stack,
                                              item.group_trace_weights,
                                              item.num_segments,
                                              item.ngroups,
                                              item.nsamples,
                                              ctx->argument,
                                              ctx->worker_index,
                                              ctx->gpu_id,
                                              &device_workspace,
                                              &out_header,
                                              &out_data);
        item.prestack_data = NULL;
        item.linear_stack = NULL;
        item.group_trace_weights = NULL;
        host_lease.release();
        if (ret != 0 || !out_data)
        {
            cleanup_tfpws_sourcepack_item(&item);
            ++(*ctx->failed);
            continue;
        }

        ret = writer.append(item.record, out_header, out_data);
        std::free(out_data);
        if (ret != 0)
        {
            ++(*ctx->failed);
            continue;
        }

        ++(*ctx->completed);
        if (ctx->progress)
        {
            ctx->progress->add("overall",
                               1,
                               std::string("worker=") + std::to_string(ctx->worker_index) +
                                   " gpu=" + std::to_string(ctx->gpu_id) +
                                   " " + item.record.path_id + ":" +
                                   item.record.src_component + "-" +
                                   item.record.rec_component);
        }
    }

    cache.close_all();
    if (writer.close() != 0)
        ++(*ctx->failed);
    free_tfpws_device_workspace(&device_workspace);
}

int run_tfpws_pipeline(const ARGUTYPE *argument,
                       const std::vector<GpuWorkerConfig> &gpu_configs)
{
    std::vector<std::string> indexes;
    if (read_sourcepack_list(argument->sourcepack_list, &indexes) != 0)
        return 1;
    if (gpu_configs.empty())
    {
        LOG_ERROR("no_gpu_worker_configured", "worker_count=0");
        return 1;
    }

    TfpwsPrescan prescan;
    if (prescan_tfpws_inputs(indexes, argument, gpu_configs, &prescan) != 0)
        return 1;

    std::string output_dir = tfpws_absolute_path(argument->output_sourcepack);
    std::string index_path = tfpws_join_path(output_dir, "sourcepack_index.tsv");
    std::string success_path = tfpws_join_path(output_dir, "_SUCCESS");

    FastxcProgressSidecar progress;
    progress.init(argument->progress_file);
    progress.set_rows({
        {"overall", "RUNNING", 0, prescan.total_groups, "groups", ""},
    });

    SourcePackGroupReader reader(indexes);
    if (reader.open() != 0)
    {
        progress.finish("FAILED", false);
        return 1;
    }

    std::size_t worker_count = gpu_configs.size();
    std::vector<WorkQueue<TfpwsJob> *> queues;
    std::vector<TfpwsWorkerContext> contexts(worker_count);
    std::vector<std::thread> workers;
    queues.reserve(worker_count);
    workers.reserve(worker_count);

    std::atomic<int> failed(0);
    std::atomic<std::size_t> completed(0);
    std::size_t host_budget_bytes = 0;
    for (std::size_t i = 0; i < gpu_configs.size(); ++i)
    {
        if ((std::size_t)-1 - host_budget_bytes < gpu_configs[i].memory_budget_bytes)
            host_budget_bytes = (std::size_t)-1;
        else
            host_budget_bytes += gpu_configs[i].memory_budget_bytes;
    }
    HostWorkspaceBudget host_budget(host_budget_bytes);
    LOG_INFO("tfpws_host_staging_budget",
             "host_staging_budget_gib=%.3f",
             bytes_to_gib((long double)host_budget_bytes));

    for (std::size_t i = 0; i < worker_count; ++i)
    {
        queues.push_back(new WorkQueue<TfpwsJob>());
        contexts[i].worker_index = gpu_configs[i].worker_index;
        contexts[i].gpu_id = gpu_configs[i].device_id;
        contexts[i].memory_budget_bytes = gpu_configs[i].memory_budget_bytes;
        contexts[i].workspace_plan = &prescan.workers[i];
        contexts[i].sub_stack_size = argument->sub_stack_size;
        contexts[i].output_dir = output_dir;
        contexts[i].argument = argument;
        contexts[i].queue = queues[i];
        contexts[i].host_budget = &host_budget;
        contexts[i].failed = &failed;
        contexts[i].completed = &completed;
        contexts[i].progress = &progress;
        workers.push_back(std::thread(tfpws_worker_main, &contexts[i]));
    }

    std::vector<SourcePackRecord> group;
    std::size_t cursor = 0;
    while (reader.next_group(&group))
    {
        TfpwsJob job;
        job.records.swap(group);
        queues[cursor % worker_count]->push(std::move(job));
        ++cursor;
    }

    for (std::size_t i = 0; i < queues.size(); ++i)
        queues[i]->close();
    for (std::size_t i = 0; i < workers.size(); ++i)
        workers[i].join();
    for (std::size_t i = 0; i < queues.size(); ++i)
        delete queues[i];

    if (failed.load() != 0)
    {
        LOG_ERROR("tfpws_failed",
                  "failed_groups=%d completed=%zu",
                  failed.load(),
                  completed.load());
        progress.finish("FAILED", false);
        return 1;
    }

    if (merge_tfpws_sourcepack_shard_indexes(output_dir, worker_count, index_path) != 0)
    {
        progress.finish("FAILED", false);
        return 1;
    }

    std::ofstream success(success_path.c_str());
    if (!success)
    {
        LOG_ERROR("create_success_marker_failed",
                  "path=\"%s\"",
                  success_path.c_str());
        progress.finish("FAILED", false);
        return 1;
    }
    success << "records\t" << completed.load() << "\n"
            << "workers\t" << worker_count << "\n"
            << "index\t" << index_path << "\n";
    for (std::size_t i = 0; i < worker_count; ++i)
    {
        char pack_name[64];
        std::snprintf(pack_name, sizeof(pack_name), "tfpws.w%03zu.pack", i);
        success << "pack\t" << tfpws_join_path(output_dir, pack_name) << "\n";
    }
    success.close();

    LOG_INFO("tfpws_completed",
             "completed=%zu workers=%zu output=\"%s\"",
             completed.load(),
             worker_count,
             output_dir.c_str());
    progress.finish("DONE", true);
    return 0;
}
