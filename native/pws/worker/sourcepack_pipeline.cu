#include "sourcepack_pipeline.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "concurrency.hpp"
#include "cuda.util.cuh"
#include "gpu.hpp"
#include "gpu_budget.hpp"
#include "host_stage.hpp"
#include "logger.h"
#include "memory_estimate.hpp"
#include "progress_sidecar.hpp"
#include "sourcepack_io.hpp"

static const float PWS_SOURCEPACK_GPU_MEMORY_FRACTION = 0.90f;

struct PwsSourcePackJob
{
    std::vector<SourcePackRecord> records;
};

struct PwsSourcePackWorkerContext
{
    std::size_t worker_index;
    int gpu_id;
    std::size_t physical_worker_count;
    std::size_t worker_memory_limit_mib;
    int substack_size;
    std::string output_dir;
    WorkQueue<PwsSourcePackJob> *queue;
    HostGroupBudget *host_budget;
    std::atomic<int> *failed;
    std::atomic<std::size_t> *completed;
    FastxcProgressSidecar *progress;
};

struct PwsPreparedBatch
{
    PwsHostBatch host_batch;
    std::vector<SourcePackRecord> output_records;
};

static std::size_t count_sourcepack_groups(const std::vector<std::string> &indexes)
{
    SourcePackGroupReader reader(indexes);
    if (reader.open() != 0)
        return 0;
    std::vector<SourcePackRecord> group;
    std::size_t count = 0;
    while (reader.next_group(&group))
        ++count;
    return count;
}

static bool prepared_batch_empty(const PwsPreparedBatch &batch)
{
    return batch.host_batch.items.empty();
}

static void cleanup_prepared_batch(PwsPreparedBatch *batch)
{
    for (std::size_t i = 0; i < batch->host_batch.items.size(); ++i)
        cleanup_host_item(&batch->host_batch.items[i]);
    batch->host_batch.items.clear();
    batch->host_batch.nsamples = 0;
    batch->host_batch.total_groups = 0;
    batch->output_records.clear();
}

static void add_prepared_item_to_batch(PwsPreparedBatch *batch,
                                       const SourcePackRecord &output_record,
                                       const PwsHostItem &host)
{
    if (prepared_batch_empty(*batch))
        batch->host_batch.nsamples = host.nsamples;
    batch->host_batch.total_groups += host.ngroups;
    batch->host_batch.items.push_back(host);
    batch->output_records.push_back(output_record);
}

static bool batch_can_accept_item(const PwsPreparedBatch &batch,
                                  const PwsHostItem &host,
                                  std::size_t gpu_budget)
{
    if (prepared_batch_empty(batch))
        return true;
    if (batch.host_batch.nsamples != host.nsamples)
        return false;

    std::size_t next_groups = batch.host_batch.total_groups + host.ngroups;
    std::size_t next_pairs = batch.host_batch.items.size() + 1;
    std::size_t next_estimate = estimate_pws_batch_gpu_bytes(batch.host_batch.nsamples,
                                                             next_groups,
                                                             next_pairs);
    return next_estimate <= gpu_budget;
}

static bool prepare_job_for_batch(const PwsSourcePackWorkerContext *ctx,
                                  const PwsSourcePackJob &job,
                                  PackFileCache *cache,
                                  PwsHostItem *host,
                                  SourcePackRecord *output_record)
{
    if (job.records.empty())
        return false;

    std::size_t group_sz = (ctx->substack_size < 2) ? 1 : (std::size_t)ctx->substack_size;
    std::size_t planned_groups = (job.records.size() + group_sz - 1) / group_sz;
    LOG_DEBUG("pws_job_start",
              "worker=%zu gpu=%d path_id=\"%s\" component_slot=%d records=%zu planned_groups=%zu",
              ctx->worker_index,
              ctx->gpu_id,
              job.records[0].path_id.c_str(),
              job.records[0].component_slot,
              job.records.size(),
              planned_groups);
    ctx->host_budget->acquire(planned_groups);

    if (prepare_sourcepack_host_item(job.records, ctx->substack_size, cache, host) != 0)
    {
        ctx->host_budget->release(planned_groups);
        LOG_ERROR("pws_host_item_prepare_failed",
                  "worker=%zu gpu=%d path_id=\"%s\" component_slot=%d records=%zu",
                  ctx->worker_index,
                  ctx->gpu_id,
                  job.records[0].path_id.c_str(),
                  job.records[0].component_slot,
                  job.records.size());
        return false;
    }

    if (host->ngroups != planned_groups)
    {
        LOG_ERROR("pws_group_count_mismatch",
                  "worker=%zu gpu=%d actual_groups=%u planned_groups=%zu",
                  ctx->worker_index,
                  ctx->gpu_id,
                  host->ngroups,
                  planned_groups);
        cleanup_host_item(host);
        ctx->host_budget->release(planned_groups);
        return false;
    }

    *output_record = job.records[0];
    return true;
}

static void run_prepared_batch(PwsSourcePackWorkerContext *ctx,
                               std::size_t gpu_budget,
                               PwsPreparedBatch *batch,
                               PwsSourcePackShardWriter *writer)
{
    if (prepared_batch_empty(*batch))
        return;

    std::size_t estimate = estimate_pws_batch_gpu_bytes(batch->host_batch.nsamples,
                                                        batch->host_batch.total_groups,
                                                        batch->host_batch.items.size());
    if (estimate > gpu_budget)
    {
        LOG_WARN("pws_batch_exceeds_gpu_budget",
                 "worker=%zu gpu=%d pairs=%zu groups=%zu estimate_mib=%.3f budget_mib=%.3f",
                 ctx->worker_index,
                 ctx->gpu_id,
                 batch->host_batch.items.size(),
                 batch->host_batch.total_groups,
                 estimate / (1024.0 * 1024.0),
                 gpu_budget / (1024.0 * 1024.0));
    }

    LOG_INFO("pws_batch_compute_start",
             "worker=%zu gpu=%d pairs=%zu groups=%zu samples=%u",
             ctx->worker_index,
             ctx->gpu_id,
             batch->host_batch.items.size(),
             batch->host_batch.total_groups,
             batch->host_batch.nsamples);

    std::vector<SACHEAD> out_headers;
    float *out_data = NULL;
    int ret = compute_pws_host_batch(&batch->host_batch,
                                     ctx->gpu_id,
                                     ctx->host_budget,
                                     &out_headers,
                                     &out_data);
    if (ret != 0 || !out_data)
    {
        LOG_ERROR("pws_batch_compute_failed",
                  "worker=%zu gpu=%d pairs=%zu groups=%zu ret=%d has_output=%d",
                  ctx->worker_index,
                  ctx->gpu_id,
                  batch->host_batch.items.size(),
                  batch->host_batch.total_groups,
                  ret,
                  out_data ? 1 : 0);
        if (out_data)
            std::free(out_data);
        std::size_t failed_pairs = batch->host_batch.items.size();
        cleanup_prepared_batch(batch);
        ctx->failed->fetch_add((int)failed_pairs);
        return;
    }

    std::size_t success_count = 0;
    for (std::size_t i = 0; i < batch->output_records.size(); ++i)
    {
        float *pair_data = out_data + i * (std::size_t)batch->host_batch.nsamples;
        ret = writer->append(batch->output_records[i], out_headers[i], pair_data);
        if (ret != 0)
        {
            LOG_ERROR("pws_writer_append_failed",
                      "worker=%zu gpu=%d path_id=\"%s\" component_slot=%d ret=%d",
                      ctx->worker_index,
                      ctx->gpu_id,
                      batch->output_records[i].path_id.c_str(),
                      batch->output_records[i].component_slot,
                      ret);
            ++(*ctx->failed);
            continue;
        }

        ++success_count;
        ++(*ctx->completed);
        if (ctx->progress)
        {
            ctx->progress->add("overall",
                               1,
                               std::string("worker=") + std::to_string(ctx->worker_index) +
                                   " gpu=" + std::to_string(ctx->gpu_id) +
                                   " " + batch->output_records[i].path_id + ":" +
                                   batch->output_records[i].src_component + "-" +
                                   batch->output_records[i].rec_component);
        }
    }

    std::free(out_data);
    LOG_INFO("pws_batch_compute_done",
             "worker=%zu gpu=%d pairs=%zu written=%zu",
             ctx->worker_index,
             ctx->gpu_id,
             batch->host_batch.items.size(),
             success_count);
    cleanup_prepared_batch(batch);
}

static void pws_sourcepack_worker_main(PwsSourcePackWorkerContext *ctx)
{
    CUDACHECK(cudaSetDevice(ctx->gpu_id));
    std::size_t gpu_free = QueryGpuFreeBytes(ctx->gpu_id);
    std::size_t auto_budget = estimate_auto_worker_gpu_budget_bytes(gpu_free,
                                                                     PWS_SOURCEPACK_GPU_MEMORY_FRACTION,
                                                                     ctx->physical_worker_count);
    std::size_t gpu_budget = estimate_worker_gpu_budget_bytes(gpu_free,
                                                              PWS_SOURCEPACK_GPU_MEMORY_FRACTION,
                                                              ctx->physical_worker_count,
                                                              ctx->worker_memory_limit_mib);

    LOG_INFO("pws_worker_memory_budget",
             "worker=%zu gpu=%d free_gpu_memory_gib=%.3f auto_budget_gib=%.3f manual_budget_gib=%.3f final_worker_budget_gib=%.3f",
             ctx->worker_index,
             ctx->gpu_id,
             gpu_free / (1024.0 * 1024.0 * 1024.0),
             auto_budget / (1024.0 * 1024.0 * 1024.0),
             ctx->worker_memory_limit_mib > 0 ? ctx->worker_memory_limit_mib / 1024.0 : 0.0,
             gpu_budget / (1024.0 * 1024.0 * 1024.0));

    PwsSourcePackShardWriter writer;
    if (writer.open(ctx->output_dir, ctx->worker_index) != 0)
    {
        LOG_ERROR("pws_worker_writer_open_failed",
                  "worker=%zu gpu=%d output_dir=\"%s\"",
                  ctx->worker_index,
                  ctx->gpu_id,
                  ctx->output_dir.c_str());
        ++(*ctx->failed);
        return;
    }

    PackFileCache cache;
    PwsSourcePackJob job;
    while (ctx->queue->pop(job))
    {
        PwsPreparedBatch batch;
        bool has_job = true;
        while (has_job)
        {
            PwsHostItem host;
            SourcePackRecord output_record;
            if (prepare_job_for_batch(ctx, job, &cache, &host, &output_record))
            {
                if (!batch_can_accept_item(batch, host, gpu_budget))
                    run_prepared_batch(ctx, gpu_budget, &batch, &writer);
                add_prepared_item_to_batch(&batch, output_record, host);
            }
            else if (!job.records.empty())
            {
                ++(*ctx->failed);
            }

            has_job = ctx->queue->try_pop(job);
        }

        run_prepared_batch(ctx, gpu_budget, &batch, &writer);
    }

    cache.close_all();
    if (writer.close() != 0)
    {
        LOG_ERROR("pws_worker_writer_close_failed",
                  "worker=%zu gpu=%d",
                  ctx->worker_index,
                  ctx->gpu_id);
        ++(*ctx->failed);
    }
    LOG_INFO("pws_worker_done",
             "worker=%zu gpu=%d completed=%zu failed=%d",
             ctx->worker_index,
             ctx->gpu_id,
             ctx->completed->load(),
             ctx->failed->load());
}

int run_pws_sourcepack_pipeline(const PwsSourcePackArgs &args,
                                const PwsGpuWorkerConfig &gpu_config)
{
    std::vector<std::string> indexes;
    if (read_sourcepack_list(args.index_list_path, &indexes) != 0)
        return 1;
    if (gpu_config.gpu_ids.empty())
    {
        LOG_ERROR("pws_no_gpu_workers", "index_list=\"%s\"", args.index_list_path);
        return 1;
    }

    std::string output_dir = pws_absolute_path(args.output_dir);
    std::string index_path = pws_join_path(output_dir, "sourcepack_index.tsv");
    std::string success_path = pws_join_path(output_dir, "_SUCCESS");

    std::size_t total_groups = count_sourcepack_groups(indexes);
    LOG_INFO("pws_sourcepack_inputs_ready",
             "index_count=%zu total_groups=%zu output_dir=\"%s\"",
             indexes.size(),
             total_groups,
             output_dir.c_str());
    FastxcProgressSidecar progress;
    progress.init(args.progress_path);
    progress.set_rows({
        {"overall", "RUNNING", 0, total_groups, "groups", ""},
    });

    SourcePackGroupReader reader(indexes);
    if (reader.open() != 0)
    {
        LOG_ERROR("pws_group_reader_open_failed",
                  "index_count=%zu",
                  indexes.size());
        progress.finish("FAILED", false);
        return 1;
    }

    std::size_t worker_count = gpu_config.gpu_ids.size();
    std::vector<WorkQueue<PwsSourcePackJob> *> queues;
    std::vector<PwsSourcePackWorkerContext> contexts(worker_count);
    std::vector<std::thread> workers;
    queues.reserve(worker_count);
    workers.reserve(worker_count);

    HostGroupBudget host_budget(args.staged_group_limit);
    std::atomic<int> failed(0);
    std::atomic<std::size_t> completed(0);

    for (std::size_t i = 0; i < worker_count; ++i)
    {
        queues.push_back(new WorkQueue<PwsSourcePackJob>());
        contexts[i].worker_index = i;
        contexts[i].gpu_id = gpu_config.gpu_ids[i];
        contexts[i].physical_worker_count = gpu_config.physical_worker_counts[i];
        contexts[i].worker_memory_limit_mib = gpu_config.gpu_memory_limits_mib[i];
        contexts[i].substack_size = args.substack_size;
        contexts[i].output_dir = output_dir;
        contexts[i].queue = queues[i];
        contexts[i].host_budget = &host_budget;
        contexts[i].failed = &failed;
        contexts[i].completed = &completed;
        contexts[i].progress = &progress;
        workers.push_back(std::thread(pws_sourcepack_worker_main, &contexts[i]));
        LOG_INFO("pws_worker_started",
                 "worker=%zu gpu=%d",
                 i,
                 gpu_config.gpu_ids[i]);
    }

    std::vector<SourcePackRecord> group;
    std::size_t cursor = 0;
    while (reader.next_group(&group))
    {
        PwsSourcePackJob job;
        job.records.swap(group);
        queues[cursor % worker_count]->push(std::move(job));
        ++cursor;
    }
    LOG_INFO("pws_jobs_enqueued",
             "jobs=%zu workers=%zu",
             cursor,
             worker_count);

    for (std::size_t i = 0; i < queues.size(); ++i)
        queues[i]->close();
    for (std::size_t i = 0; i < workers.size(); ++i)
        workers[i].join();
    for (std::size_t i = 0; i < queues.size(); ++i)
        delete queues[i];

    if (failed.load() != 0)
    {
        LOG_ERROR("pws_pipeline_failed",
                  "failed=%d completed=%zu",
                  failed.load(),
                  completed.load());
        progress.finish("FAILED", false);
        return 1;
    }

    if (merge_pws_sourcepack_shard_indexes(output_dir, worker_count, index_path) != 0)
    {
        LOG_ERROR("pws_shard_index_merge_failed",
                  "output_dir=\"%s\" worker_count=%zu index=\"%s\"",
                  output_dir.c_str(),
                  worker_count,
                  index_path.c_str());
        progress.finish("FAILED", false);
        return 1;
    }

    std::ofstream success(success_path.c_str());
    if (!success)
    {
        LOG_ERROR("pws_success_marker_open_failed",
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
        std::snprintf(pack_name, sizeof(pack_name), "pws.w%03zu.pack", i);
        success << "pack\t" << pws_join_path(output_dir, pack_name) << "\n";
    }
    success.close();

    LOG_INFO("pws_pipeline_done",
             "completed=%zu workers=%zu output_dir=\"%s\" index=\"%s\"",
             completed.load(), worker_count, output_dir.c_str(), index_path.c_str());
    progress.finish("DONE", true);
    return 0;
}
