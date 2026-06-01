#include "writer.hpp"
#include "logger.h"
#include "pack_writer.hpp"
#include "progress_sidecar.hpp"
#include "sac_record.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <pthread.h>
#include <vector>

extern "C"
{
#include "my_write_sac.h"
}

static void *write_thread_main(void *arg)
{
  WriteContext *ctx = (WriteContext *)arg;
  XcTimeData tinfo;
  XcPackWriter pack_writer;
  std::vector<char> pack_record;
  bool pack_open = false;

  if (!parse_timestamp_text(ctx->timestamp->timestamp, &tinfo))
  {
    LOG_WARN("timestamp_parse_failed",
             "timestamp=\"%s\" worker=%zu action=use_zero_sac_time",
             ctx->timestamp->timestamp.c_str(), ctx->cfg->worker_id);
  }
  if (ctx->cfg->write_mode == MODE_PACK)
  {
    std::string pack_root = xc_pack_root_dir(ctx->cfg->output_dir);
    if (xc_pack_writer_open(&pack_writer,
                            pack_root.c_str(),
                            ctx->timestamp->timestamp.c_str(),
                            ctx->cfg->worker_id,
                            ctx->job,
                            kXcPackDefaultMaxBytes) != 0)
    {
      LOG_ERROR("xcpack_writer_open_failed",
                "worker=%zu block_i=%zu block_j=%zu timestamp=\"%s\" root=\"%s\"",
                ctx->cfg->worker_id,
                ctx->job ? ctx->job->anchor_block : 0,
                ctx->job ? ctx->job->target_begin_block : 0,
                ctx->timestamp->timestamp.c_str(), pack_root.c_str());
      return NULL;
    }
    pack_open = true;
  }

  for (size_t k = ctx->begin; k < ctx->end; ++k)
  {
    const XcTask &task = (*(ctx->tasks))[k];
    const SpecMeta &src = ctx->timestamp->specs[task.src_meta_idx];
    const SpecMeta &rec = ctx->timestamp->specs[task.rec_meta_idx];
    char out_path[PATH_MAX];
    if (build_output_path(out_path, sizeof(out_path), ctx->cfg->output_dir, src, rec,
                          ctx->cfg->write_mode != MODE_PACK) != 0)
    {
      LOG_ERROR("output_path_build_failed", "worker=%zu path_id=%08d",
                ctx->cfg->worker_id, task.path_id);
      continue;
    }

    SACHEAD hd;
    build_sac_header_for_task(&hd, ctx->shape, task, src, rec, &tinfo);

    const float *trace = ctx->cc + k * (size_t)ctx->shape->cc_size;
    if (ctx->cfg->write_mode == MODE_PACK)
    {
      XcPackRecordMeta meta;
      fill_pack_record_meta(&meta, ctx->timestamp, ctx->shape, ctx->cfg, ctx->job,
                            task, src, rec, out_path);
      if (make_sac_record(&pack_record, hd, trace, (size_t)ctx->shape->cc_size) != 0)
      {
        LOG_ERROR("xcpack_record_build_failed",
                  "worker=%zu block_i=%zu block_j=%zu path_id=%08d",
                  ctx->cfg->worker_id,
                  ctx->job ? ctx->job->anchor_block : 0,
                  ctx->job ? ctx->job->target_begin_block : 0,
                  task.path_id);
        continue;
      }
      if (xc_pack_writer_append(&pack_writer, &meta, pack_record.data(),
                                (uint64_t)pack_record.size()) != 0)
      {
        LOG_ERROR("xcpack_record_write_failed",
                  "worker=%zu block_i=%zu block_j=%zu path_id=%08d timestamp=\"%s\"",
                  ctx->cfg->worker_id,
                  ctx->job ? ctx->job->anchor_block : 0,
                  ctx->job ? ctx->job->target_begin_block : 0,
                  task.path_id,
                  ctx->timestamp->timestamp.c_str());
      }
    }
    else if (my_write_sac(out_path, hd, trace, ctx->cfg->write_mode) != 0)
      LOG_ERROR("sac_write_failed", "worker=%zu path=\"%s\"",
                ctx->cfg->worker_id, out_path);
  }

  if (pack_open)
    xc_pack_writer_close(&pack_writer);
  return NULL;
}

static void write_results_parallel(const WorkerConfig *cfg,
                                   const RuntimeShape *shape,
                                   const TimestampWork *timestamp,
                                   const RowBatchJob *job,
                                   const std::vector<XcTask> &tasks,
                                   const float *cc)
{
  if (tasks.empty())
    return;
  size_t nthreads = cfg->write_mode == MODE_PACK
                        ? (size_t)1
                        : std::min(cfg->writer_threads, tasks.size());
  if (nthreads == 0)
    return;
  std::vector<pthread_t> threads(nthreads);
  std::vector<WriteContext> contexts(nthreads);
  size_t per = tasks.size() / nthreads;
  size_t rem = tasks.size() % nthreads;
  size_t begin = 0;
  for (size_t i = 0; i < nthreads; ++i)
  {
    size_t end = begin + per + (i < rem ? 1 : 0);
    contexts[i].cfg = cfg;
    contexts[i].shape = shape;
    contexts[i].timestamp = timestamp;
    contexts[i].job = job;
    contexts[i].tasks = &tasks;
    contexts[i].cc = cc;
    contexts[i].begin = begin;
    contexts[i].end = end;
    pthread_create(&threads[i], NULL, write_thread_main, &contexts[i]);
    begin = end;
  }
  for (size_t i = 0; i < nthreads; ++i)
    pthread_join(threads[i], NULL);

  if (cfg->progress)
    cfg->progress->add("current", tasks.size(), timestamp->timestamp);
}

static void *lazy_writer_main(void *arg)
{
  LazyWriteQueue *queue = (LazyWriteQueue *)arg;
  while (true)
  {
    WriteBatch *batch = NULL;
    pthread_mutex_lock(&queue->mutex);
    while (queue->pending.empty() && !queue->closed)
      pthread_cond_wait(&queue->can_pop, &queue->mutex);
    if (queue->pending.empty() && queue->closed)
    {
      pthread_mutex_unlock(&queue->mutex);
      break;
    }
    batch = queue->pending.front();
    queue->pending.pop_front();
    pthread_mutex_unlock(&queue->mutex);

    write_results_parallel(queue->cfg, queue->shape, queue->timestamp,
                           &batch->job,
                           batch->tasks, batch->cc.data());
    batch->tasks.clear();

    pthread_mutex_lock(&queue->mutex);
    queue->free_batches.push_back(batch);
    if (queue->inflight > 0)
      --queue->inflight;
    pthread_cond_signal(&queue->can_push);
    pthread_mutex_unlock(&queue->mutex);
  }
  return NULL;
}

int lazy_writer_init(LazyWriteQueue *queue,
                     const WorkerConfig *cfg,
                     const RuntimeShape *shape,
                     const TimestampWork *timestamp,
                     size_t pair_capacity)
{
  queue->cfg = NULL;
  queue->shape = NULL;
  queue->timestamp = NULL;
  queue->pair_capacity = 0;
  queue->max_inflight = 0;
  queue->inflight = 0;
  queue->closed = false;
  queue->active = false;
  queue->pool.clear();
  queue->free_batches.clear();
  queue->pending.clear();
  if (cfg->lazy_write_depth == 0)
    return 0;
  if (pair_capacity == 0 || pair_capacity > cfg->pair_capacity)
    pair_capacity = cfg->pair_capacity;

  size_t cc_values = 0;
  size_t cc_bytes = 0;
  size_t task_bytes = 0;
  size_t batch_bytes = 0;
  size_t total_bytes = 0;
  if (!checked_mul_size(pair_capacity, (size_t)shape->cc_size, &cc_values) ||
      !checked_mul_size(cc_values, sizeof(float), &cc_bytes) ||
      !checked_mul_size(pair_capacity, sizeof(XcTask), &task_bytes) ||
      !checked_add_size(cc_bytes, task_bytes, &batch_bytes) ||
      !checked_mul_size(cfg->lazy_write_depth, batch_bytes, &total_bytes))
  {
    LOG_ERROR("lazy_write_buffer_overflow", "worker=%zu", cfg->worker_id);
    return -1;
  }
  try
  {
    queue->pool.resize(cfg->lazy_write_depth);
    for (size_t i = 0; i < queue->pool.size(); ++i)
    {
      queue->pool[i].tasks.reserve(pair_capacity);
      queue->pool[i].cc.resize(cc_values);
      queue->free_batches.push_back(&queue->pool[i]);
    }
  }
  catch (...)
  {
    LOG_ERROR("lazy_write_prealloc_failed",
              "worker=%zu depth=%zu pair_capacity=%zu cc_size=%d",
              cfg->worker_id, cfg->lazy_write_depth, cfg->pair_capacity, shape->cc_size);
    queue->pool.clear();
    queue->free_batches.clear();
    return -1;
  }

  queue->cfg = cfg;
  queue->shape = shape;
  queue->timestamp = timestamp;
  queue->pair_capacity = pair_capacity;
  queue->max_inflight = cfg->lazy_write_depth;
  pthread_mutex_init(&queue->mutex, NULL);
  pthread_cond_init(&queue->can_push, NULL);
  pthread_cond_init(&queue->can_pop, NULL);
  queue->active = true;
  if (pthread_create(&queue->thread, NULL, lazy_writer_main, queue) != 0)
  {
    LOG_ERROR("lazy_writer_thread_create_failed", "worker=%zu", cfg->worker_id);
    queue->active = false;
    pthread_cond_destroy(&queue->can_pop);
    pthread_cond_destroy(&queue->can_push);
    pthread_mutex_destroy(&queue->mutex);
    queue->pool.clear();
    queue->free_batches.clear();
    return -1;
  }
  LOG_INFO("lazy_write_prealloc",
           "worker=%zu depth=%zu pair_capacity=%zu worker_pair_capacity=%zu cc_size=%d bytes_mib=%.3f",
           cfg->worker_id, cfg->lazy_write_depth, pair_capacity, cfg->pair_capacity, shape->cc_size,
           bytes_to_mib(total_bytes));
  return 0;
}

void lazy_writer_close(LazyWriteQueue *queue)
{
  if (!queue->active)
    return;
  pthread_mutex_lock(&queue->mutex);
  queue->closed = true;
  pthread_cond_signal(&queue->can_pop);
  pthread_mutex_unlock(&queue->mutex);
  pthread_join(queue->thread, NULL);
  pthread_cond_destroy(&queue->can_pop);
  pthread_cond_destroy(&queue->can_push);
  pthread_mutex_destroy(&queue->mutex);
  queue->pending.clear();
  queue->free_batches.clear();
  queue->pool.clear();
  queue->active = false;
}

void submit_write_results(LazyWriteQueue *queue,
                          const WorkerConfig *cfg,
                          const RuntimeShape *shape,
                          const TimestampWork *timestamp,
                          const RowBatchJob *job,
                          const std::vector<XcTask> &tasks,
                          const float *cc)
{
  if (!queue || !queue->active)
  {
    write_results_parallel(cfg, shape, timestamp, job, tasks, cc);
    return;
  }

  WriteBatch *batch = NULL;
  pthread_mutex_lock(&queue->mutex);
  while (!queue->closed && queue->free_batches.empty())
    pthread_cond_wait(&queue->can_push, &queue->mutex);
  if (queue->closed)
  {
    pthread_mutex_unlock(&queue->mutex);
    LOG_ERROR("lazy_writer_submit_closed", "worker=%zu", cfg->worker_id);
    exit(1);
  }
  batch = queue->free_batches.front();
  queue->free_batches.pop_front();
  ++queue->inflight;
  pthread_mutex_unlock(&queue->mutex);

  try
  {
    if (tasks.size() > queue->pair_capacity)
    {
      LOG_ERROR("lazy_write_capacity_exceeded",
                "worker=%zu task_count=%zu pair_capacity=%zu",
                cfg->worker_id, tasks.size(), queue->pair_capacity);
      exit(1);
    }
    batch->tasks.resize(tasks.size());
    if (job)
      batch->job = *job;
    else
      batch->job = RowBatchJob();
    if (!tasks.empty())
      memcpy(batch->tasks.data(), tasks.data(), tasks.size() * sizeof(XcTask));
    memcpy(batch->cc.data(), cc, tasks.size() * (size_t)shape->cc_size * sizeof(float));
  }
  catch (...)
  {
    pthread_mutex_lock(&queue->mutex);
    batch->tasks.clear();
    queue->free_batches.push_back(batch);
    if (queue->inflight > 0)
      --queue->inflight;
    pthread_cond_signal(&queue->can_push);
    pthread_mutex_unlock(&queue->mutex);
    LOG_ERROR("lazy_write_batch_alloc_failed", "worker=%zu task_count=%zu",
              cfg->worker_id, tasks.size());
    exit(1);
  }

  pthread_mutex_lock(&queue->mutex);
  queue->pending.push_back(batch);
  pthread_cond_signal(&queue->can_pop);
  pthread_mutex_unlock(&queue->mutex);
}
