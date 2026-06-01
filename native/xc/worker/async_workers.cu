#include "async_workers.hpp"

#include "cuda.kernels.cuh"
#include "cuda.util.cuh"
#include "executor.hpp"
#include "input.hpp"
#include "logger.h"
#include "memory.hpp"
#include "scheduler.hpp"
#include "writer.hpp"

#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <time.h>
#include <unistd.h>

static void execute_row_batch_job(WorkerContext *ctx,
                                  GpuBuffers *buf,
                                  LazyWriteQueue *writer_queue,
                                  const RowBatchJob *job)
{
  std::vector<size_t> loaded_indices;
  std::vector<XcTask> tasks =
      build_tasks_for_job(ctx->timestamp, ctx->paths, job, &loaded_indices);
  if (tasks.empty())
    return;
  if (tasks.size() > ctx->cfg.pair_capacity)
  {
    LOG_ERROR("task_capacity_exceeded",
              "worker=%zu task_count=%zu pair_capacity=%zu timestamp=\"%s\"",
              ctx->cfg.worker_id, tasks.size(), ctx->cfg.pair_capacity,
              ctx->timestamp->timestamp.c_str());
    exit(1);
  }

  for (size_t i = 0; i < tasks.size(); ++i)
  {
    buf->h_src_idx[i] = tasks[i].src_local_idx;
    buf->h_rec_idx[i] = tasks[i].rec_local_idx;
  }
  CUDACHECK(cudaMemcpy(buf->d_src_idx, buf->h_src_idx,
                       tasks.size() * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(buf->d_rec_idx, buf->h_rec_idx,
                       tasks.size() * sizeof(size_t), cudaMemcpyHostToDevice));

  dim3 grid, block;
  DimCompute(&grid, &block, (size_t)ctx->shape->nspec, tasks.size());
  size_t input_bytes = 0;
  if (!checked_mul_size(loaded_indices.size(), ctx->shape->step_bytes, &input_bytes))
  {
    LOG_ERROR("input_bytes_overflow", "worker=%zu loaded_files=%zu step_bytes=%zu",
              ctx->cfg.worker_id, loaded_indices.size(), ctx->shape->step_bytes);
    exit(1);
  }
  size_t stack_values = 0;
  size_t stack_bytes = 0;
  if (!checked_mul_size(tasks.size(), (size_t)ctx->shape->nspec, &stack_values) ||
      !checked_mul_size(stack_values, sizeof(cuComplex), &stack_bytes))
  {
    LOG_ERROR("stack_bytes_overflow", "worker=%zu task_count=%zu nspec=%d",
              ctx->cfg.worker_id, tasks.size(), ctx->shape->nspec);
    exit(1);
  }

  const float scale = ctx->shape->nstep > 0 ? 1.0f / (float)ctx->shape->nstep : 0.0f;
  CUDACHECK(cudaMemset(buf->d_stack, 0, stack_bytes));
  for (size_t step = 0; step < (size_t)ctx->shape->nstep; ++step)
  {
    load_job_step_input(ctx->timestamp, ctx->shape, loaded_indices, step, buf->h_spec);
    CUDACHECK(cudaMemcpy(buf->d_spec,
                         buf->h_spec,
                         input_bytes,
                         cudaMemcpyHostToDevice));
    accumulateStepXc2DKernel<<<grid, block>>>(buf->d_spec,
                                              buf->d_src_idx,
                                              buf->d_rec_idx,
                                              buf->d_stack,
                                              (size_t)ctx->shape->nspec,
                                              scale,
                                              tasks.size());
    CUDACHECK(cudaGetLastError());
  }
  finalize_xc_batch(buf, ctx->shape, tasks.size());
  submit_write_results(writer_queue, &ctx->cfg, ctx->shape, ctx->timestamp,
                       job, tasks, buf->h_cc);
}

static size_t timestamp_writer_pair_capacity(const WorkerConfig *cfg,
                                             const TimestampWork *timestamp)
{
  size_t active_block_size = 1;
  size_t pair_capacity = cfg->pair_capacity;
  if (!cfg || !timestamp || timestamp->specs.empty())
    return pair_capacity;
  active_block_size = cfg->block_file_count < timestamp->specs.size()
                          ? cfg->block_file_count
                          : timestamp->specs.size();
  if (active_block_size == 0)
    active_block_size = 1;
  if (!compute_pair_capacity_for_block(active_block_size, &pair_capacity) ||
      pair_capacity == 0 || pair_capacity > cfg->pair_capacity)
    pair_capacity = cfg->pair_capacity;
  return pair_capacity;
}

void resident_worker_pool_init(ResidentWorkerPool *pool, size_t worker_count)
{
  memset(pool, 0, sizeof(*pool));
  pool->worker_count = worker_count;
  pthread_mutex_init(&pool->mutex, NULL);
  pthread_cond_init(&pool->work_ready, NULL);
  pthread_cond_init(&pool->all_ready, NULL);
  pthread_cond_init(&pool->all_done, NULL);
}

void resident_worker_pool_destroy(ResidentWorkerPool *pool)
{
  pthread_cond_destroy(&pool->all_done);
  pthread_cond_destroy(&pool->all_ready);
  pthread_cond_destroy(&pool->work_ready);
  pthread_mutex_destroy(&pool->mutex);
  memset(pool, 0, sizeof(*pool));
}

size_t resident_worker_pool_wait_ready_or_timeout(ResidentWorkerPool *pool,
                                                  int timeout_seconds)
{
  struct timespec deadline;
  size_t ready = 0;
  clock_gettime(CLOCK_REALTIME, &deadline);
  deadline.tv_sec += timeout_seconds;

  pthread_mutex_lock(&pool->mutex);
  while (pool->ready_count < pool->worker_count)
  {
    int rc = pthread_cond_timedwait(&pool->all_ready, &pool->mutex, &deadline);
    if (rc == ETIMEDOUT)
      break;
  }
  ready = pool->ready_count;
  pthread_mutex_unlock(&pool->mutex);
  return ready;
}

size_t resident_worker_pool_select_ready(ResidentWorkerPool *pool,
                                         ResidentWorkerContext *contexts,
                                         size_t context_count)
{
  size_t ready = 0;
  pthread_mutex_lock(&pool->mutex);
  for (size_t i = 0; i < context_count; ++i)
  {
    if (contexts[i].ready)
      ++ready;
    else
      contexts[i].stop_before_ready = true;
  }
  pool->worker_count = ready;
  pthread_mutex_unlock(&pool->mutex);
  return ready;
}

void resident_worker_pool_submit(ResidentWorkerPool *pool,
                                 const TimestampWork *timestamp,
                                 JobQueue *queue)
{
  pthread_mutex_lock(&pool->mutex);
  pool->timestamp = timestamp;
  pool->queue = queue;
  pool->done_count = 0;
  pool->failed = false;
  ++pool->generation;
  pthread_cond_broadcast(&pool->work_ready);
  pthread_mutex_unlock(&pool->mutex);
}

bool resident_worker_pool_wait_done(ResidentWorkerPool *pool)
{
  bool ok = true;
  pthread_mutex_lock(&pool->mutex);
  while (pool->done_count < pool->worker_count)
    pthread_cond_wait(&pool->all_done, &pool->mutex);
  ok = !pool->failed;
  pthread_mutex_unlock(&pool->mutex);
  return ok;
}

void resident_worker_pool_stop(ResidentWorkerPool *pool)
{
  pthread_mutex_lock(&pool->mutex);
  pool->shutdown = true;
  pthread_cond_broadcast(&pool->work_ready);
  pthread_mutex_unlock(&pool->mutex);
}

static void resident_worker_mark_done(ResidentWorkerPool *pool, bool failed)
{
  pthread_mutex_lock(&pool->mutex);
  if (failed)
    pool->failed = true;
  ++pool->done_count;
  if (pool->done_count >= pool->worker_count)
    pthread_cond_broadcast(&pool->all_done);
  pthread_mutex_unlock(&pool->mutex);
}

static bool resident_worker_shutdown_requested(ResidentWorkerPool *pool)
{
  bool shutdown = false;
  pthread_mutex_lock(&pool->mutex);
  shutdown = pool->shutdown;
  pthread_mutex_unlock(&pool->mutex);
  return shutdown;
}

static bool resident_worker_stop_before_ready_requested(ResidentWorkerContext *resident)
{
  bool stop = false;
  pthread_mutex_lock(&resident->pool->mutex);
  stop = resident->stop_before_ready;
  pthread_mutex_unlock(&resident->pool->mutex);
  return stop;
}

void *resident_gpu_worker_main(void *arg)
{
  ResidentWorkerContext *resident = (ResidentWorkerContext *)arg;
  GpuBuffers buf;
  size_t retry = 0;
  memset(&buf, 0, sizeof(buf));

  LOG_INFO("resident_worker_alloc_start", "worker=%zu gpu=%zu block_files=%zu pair_capacity=%zu",
           resident->cfg.worker_id, resident->cfg.gpu_id,
           resident->cfg.block_file_count, resident->cfg.pair_capacity);
  while (init_gpu_buffers(&buf, &resident->cfg, resident->shape) != 0)
  {
    free_gpu_buffers(&buf);
    memset(&buf, 0, sizeof(buf));
    ++retry;
    LOG_WARN("resident_worker_alloc_retry",
             "worker=%zu gpu=%zu retry=%zu block_files=%zu pair_capacity=%zu",
             resident->cfg.worker_id, resident->cfg.gpu_id, retry,
             resident->cfg.block_file_count, resident->cfg.pair_capacity);
    if (resident_worker_shutdown_requested(resident->pool))
    {
      LOG_INFO("resident_worker_exit_before_ready", "worker=%zu gpu=%zu retry=%zu",
               resident->cfg.worker_id, resident->cfg.gpu_id, retry);
      return NULL;
    }
    if (resident_worker_stop_before_ready_requested(resident))
    {
      LOG_INFO("resident_worker_exit_before_ready", "worker=%zu gpu=%zu retry=%zu",
               resident->cfg.worker_id, resident->cfg.gpu_id, retry);
      return NULL;
    }
    sleep(kResidentRetrySeconds);
  }
  if (resident_worker_stop_before_ready_requested(resident))
  {
    free_gpu_buffers(&buf);
    LOG_INFO("resident_worker_exit_before_ready", "worker=%zu gpu=%zu retry=%zu",
             resident->cfg.worker_id, resident->cfg.gpu_id, retry);
    return NULL;
  }

  pthread_mutex_lock(&resident->pool->mutex);
  resident->ready = true;
  ++resident->pool->ready_count;
  pthread_cond_broadcast(&resident->pool->all_ready);
  pthread_mutex_unlock(&resident->pool->mutex);
  LOG_INFO("resident_worker_ready", "worker=%zu gpu=%zu retry=%zu",
           resident->cfg.worker_id, resident->cfg.gpu_id, retry);

  size_t seen_generation = 0;
  while (true)
  {
    const TimestampWork *timestamp = NULL;
    JobQueue *queue = NULL;
    pthread_mutex_lock(&resident->pool->mutex);
    while (!resident->pool->shutdown &&
           resident->pool->generation == seen_generation)
    {
      pthread_cond_wait(&resident->pool->work_ready, &resident->pool->mutex);
    }
    if (resident->pool->shutdown)
    {
      pthread_mutex_unlock(&resident->pool->mutex);
      break;
    }
    seen_generation = resident->pool->generation;
    timestamp = resident->pool->timestamp;
    queue = resident->pool->queue;
    pthread_mutex_unlock(&resident->pool->mutex);

    WorkerContext ctx;
    LazyWriteQueue writer_queue;
    bool failed = false;
    ctx.cfg = resident->cfg;
    ctx.shape = resident->shape;
    ctx.paths = resident->paths;
    ctx.timestamp = timestamp;
    ctx.queue = queue;

    LOG_INFO("resident_worker_timestamp_start",
             "worker=%zu gpu=%zu timestamp=\"%s\"",
             ctx.cfg.worker_id, ctx.cfg.gpu_id, ctx.timestamp->timestamp.c_str());
    const size_t writer_pair_capacity =
        timestamp_writer_pair_capacity(&ctx.cfg, ctx.timestamp);
    if (lazy_writer_init(&writer_queue, &ctx.cfg, ctx.shape, ctx.timestamp,
                         writer_pair_capacity) != 0)
    {
      LOG_ERROR("lazy_writer_start_failed", "worker=%zu gpu=%zu timestamp=\"%s\"",
                ctx.cfg.worker_id, ctx.cfg.gpu_id, ctx.timestamp->timestamp.c_str());
      failed = true;
    }
    else
    {
      RowBatchJob job;
      while (queue_pop(ctx.queue, &job))
      {
        execute_row_batch_job(&ctx, &buf, &writer_queue, &job);
      }
      lazy_writer_close(&writer_queue);
    }

    LOG_INFO("resident_worker_timestamp_done",
             "worker=%zu gpu=%zu timestamp=\"%s\" failed=%s",
             ctx.cfg.worker_id, ctx.cfg.gpu_id, ctx.timestamp->timestamp.c_str(),
             failed ? "yes" : "no");
    resident_worker_mark_done(resident->pool, failed);
  }

  free_gpu_buffers(&buf);
  LOG_INFO("resident_worker_exit", "worker=%zu gpu=%zu",
           resident->cfg.worker_id, resident->cfg.gpu_id);
  return NULL;
}
