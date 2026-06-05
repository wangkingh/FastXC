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

static void write_results_pack(const WorkerConfig *cfg,
                               const RuntimeShape *shape,
                               const TimestampWork *timestamp,
                               const RowBatchJob *job,
                               const std::vector<XcTask> &tasks,
                               const float *cc)
{
  if (tasks.empty())
    return;

  XcTimeData tinfo;
  XcPackWriter pack_writer;
  std::vector<char> pack_record;

  if (!parse_timestamp_text(timestamp->timestamp, &tinfo))
  {
    LOG_WARN("timestamp_parse_failed",
             "timestamp=\"%s\" worker=%zu action=use_zero_sac_time",
             timestamp->timestamp.c_str(), cfg->worker_id);
  }
  std::string pack_root = xc_pack_root_dir(cfg->output_dir);
  if (xc_pack_writer_open(&pack_writer,
                          pack_root.c_str(),
                          timestamp->timestamp.c_str(),
                          cfg->worker_id,
                          job,
                          kXcPackDefaultMaxBytes) != 0)
  {
    LOG_ERROR("xcpack_writer_open_failed",
              "worker=%zu block_i=%zu block_j=%zu timestamp=\"%s\" root=\"%s\"",
              cfg->worker_id,
              job ? job->anchor_block : 0,
              job ? job->target_begin_block : 0,
              timestamp->timestamp.c_str(), pack_root.c_str());
    return;
  }

  for (size_t k = 0; k < tasks.size(); ++k)
  {
    const XcTask &task = tasks[k];
    const SpecMeta &src = timestamp->specs[task.src_meta_idx];
    const SpecMeta &rec = timestamp->specs[task.rec_meta_idx];
    char out_path[PATH_MAX];
    if (build_output_path(out_path, sizeof(out_path), cfg->output_dir, src, rec,
                          false) != 0)
    {
      LOG_ERROR("output_path_build_failed", "worker=%zu path_id=%08d",
                cfg->worker_id, task.path_id);
      continue;
    }

    SACHEAD hd;
    build_sac_header_for_task(&hd, shape, task, src, rec, &tinfo);

    const float *trace = cc + k * (size_t)shape->cc_size;
    XcPackRecordMeta meta;
    fill_pack_record_meta(&meta, timestamp, shape, cfg, job,
                          task, src, rec, out_path);
    if (make_sac_record(&pack_record, hd, trace, (size_t)shape->cc_size) != 0)
    {
      LOG_ERROR("xcpack_record_build_failed",
                "worker=%zu block_i=%zu block_j=%zu path_id=%08d",
                cfg->worker_id,
                job ? job->anchor_block : 0,
                job ? job->target_begin_block : 0,
                task.path_id);
      continue;
    }
    if (xc_pack_writer_append(&pack_writer, &meta, pack_record.data(),
                              (uint64_t)pack_record.size()) != 0)
    {
      LOG_ERROR("xcpack_record_write_failed",
                "worker=%zu block_i=%zu block_j=%zu path_id=%08d timestamp=\"%s\"",
                cfg->worker_id,
                job ? job->anchor_block : 0,
                job ? job->target_begin_block : 0,
                task.path_id,
                timestamp->timestamp.c_str());
    }
  }

  xc_pack_writer_close(&pack_writer);

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

    write_results_pack(queue->cfg, queue->shape, queue->timestamp,
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
    write_results_pack(cfg, shape, timestamp, job, tasks, cc);
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
