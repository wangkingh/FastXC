#include "io_pool.h"

#include "logger.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct IoWorker
{
  IoPool *pool;
  size_t worker_id;
} IoWorker;

struct IoPool
{
  pthread_t *threads;
  IoWorker *workers;
  size_t num_threads;
  const char *kind_name;

  pthread_mutex_t mutex;
  pthread_cond_t work_ready;
  pthread_cond_t work_done;

  int stopping;
  int job_active;
  int job_status;
  size_t generation;
  size_t active_threads;
  size_t remaining_threads;
  size_t next_index;
  size_t item_count;
  IoPoolItemFn item_fn;
  void *item_context;
};

static const char *poolKindName(const IoPool *pool)
{
  return (pool != NULL && pool->kind_name != NULL) ? pool->kind_name : "io";
}

static void finishWorkerForJob(IoPool *pool)
{
  if (pool->remaining_threads > 0)
  {
    pool->remaining_threads--;
  }
  if (pool->remaining_threads == 0)
  {
    pool->job_active = 0;
    pthread_cond_signal(&pool->work_done);
  }
}

static void *ioWorkerMain(void *arg)
{
  IoWorker *worker = (IoWorker *)arg;
  IoPool *pool = worker->pool;
  size_t seen_generation = 0;

  for (;;)
  {
    pthread_mutex_lock(&pool->mutex);
    while (!pool->stopping &&
           (!pool->job_active ||
            seen_generation == pool->generation ||
            worker->worker_id >= pool->active_threads))
    {
      pthread_cond_wait(&pool->work_ready, &pool->mutex);
    }

    if (pool->stopping)
    {
      pthread_mutex_unlock(&pool->mutex);
      break;
    }

    seen_generation = pool->generation;

    for (;;)
    {
      if (pool->job_status != 0 || pool->next_index >= pool->item_count)
      {
        finishWorkerForJob(pool);
        pthread_mutex_unlock(&pool->mutex);
        break;
      }

      size_t index = pool->next_index++;
      IoPoolItemFn item_fn = pool->item_fn;
      void *item_context = pool->item_context;
      pthread_mutex_unlock(&pool->mutex);

      if (item_fn == NULL || item_fn(item_context, index) != 0)
      {
        pthread_mutex_lock(&pool->mutex);
        pool->job_status = -1;
      }
      else
      {
        pthread_mutex_lock(&pool->mutex);
      }
    }
  }

  return NULL;
}

IoPool *IoPoolCreate(size_t num_threads, const char *kind_name)
{
  IoPool *pool = (IoPool *)malloc(sizeof(IoPool));
  if (pool == NULL)
  {
    LOG_ERROR("alloc_failed", "target=io_thread_pool");
    return NULL;
  }

  memset(pool, 0, sizeof(*pool));
  if (num_threads < 1)
  {
    LOG_ERROR("io_thread_group_invalid", "threads=%zu", num_threads);
    free(pool);
    return NULL;
  }

  pool->num_threads = num_threads;
  pool->kind_name = kind_name;
  pthread_mutex_init(&pool->mutex, NULL);
  pthread_cond_init(&pool->work_ready, NULL);
  pthread_cond_init(&pool->work_done, NULL);

  pool->threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  pool->workers = (IoWorker *)malloc(num_threads * sizeof(IoWorker));
  if (pool->threads == NULL || pool->workers == NULL)
  {
    LOG_ERROR("alloc_failed", "target=io_thread_pool_arrays threads=%zu", num_threads);
    free(pool->threads);
    free(pool->workers);
    pthread_cond_destroy(&pool->work_done);
    pthread_cond_destroy(&pool->work_ready);
    pthread_mutex_destroy(&pool->mutex);
    free(pool);
    return NULL;
  }

  for (size_t i = 0; i < num_threads; i++)
  {
    pool->workers[i].pool = pool;
    pool->workers[i].worker_id = i;
    int ret = pthread_create(&pool->threads[i], NULL, ioWorkerMain, &pool->workers[i]);
    if (ret != 0)
    {
      LOG_ERROR("thread_create_failed", "phase=io_pool_start index=%zu", i);
      pthread_mutex_lock(&pool->mutex);
      pool->stopping = 1;
      pthread_cond_broadcast(&pool->work_ready);
      pthread_mutex_unlock(&pool->mutex);
      for (size_t j = 0; j < i; j++)
      {
        pthread_join(pool->threads[j], NULL);
      }
      free(pool->threads);
      free(pool->workers);
      pthread_cond_destroy(&pool->work_done);
      pthread_cond_destroy(&pool->work_ready);
      pthread_mutex_destroy(&pool->mutex);
      free(pool);
      return NULL;
    }
  }

  LOG_INFO("io_thread_pool_started", "kind=%s threads=%zu",
           poolKindName(pool), num_threads);
  return pool;
}

int IoPoolRun(IoPool *pool, size_t item_count, int requested_threads,
              const char *phase, const char *start_event,
              const char *done_event, const char *failed_event,
              IoPoolItemFn item_fn, void *item_context)
{
  if (pool == NULL || requested_threads <= 0 || item_fn == NULL)
  {
    LOG_ERROR("io_thread_group_invalid",
              "phase=%s threads=%d count=%zu item_fn_valid=%d",
              phase != NULL ? phase : "unknown", requested_threads,
              item_count, item_fn != NULL);
    return -1;
  }
  if ((size_t)requested_threads > pool->num_threads)
  {
    LOG_ERROR("io_thread_group_overflow",
              "phase=%s requested=%d allocated=%zu",
              phase != NULL ? phase : "unknown",
              requested_threads, pool->num_threads);
    return -1;
  }
  if (item_count == 0)
  {
    return 0;
  }

  size_t thread_count = (size_t)requested_threads;
  if (thread_count > item_count)
  {
    thread_count = item_count;
  }

  LOG_INFO(start_event != NULL ? start_event : "io_pool_job_start",
           "phase=%s count=%zu threads=%zu persistent=1",
           phase != NULL ? phase : "unknown", item_count, thread_count);

  pthread_mutex_lock(&pool->mutex);
  if (pool->job_active)
  {
    LOG_ERROR("io_thread_group_busy", "phase=%s",
              phase != NULL ? phase : "unknown");
    pthread_mutex_unlock(&pool->mutex);
    return -1;
  }

  pool->item_count = item_count;
  pool->next_index = 0;
  pool->active_threads = thread_count;
  pool->remaining_threads = thread_count;
  pool->job_status = 0;
  pool->item_fn = item_fn;
  pool->item_context = item_context;
  pool->job_active = 1;
  pool->generation++;

  pthread_cond_broadcast(&pool->work_ready);
  while (pool->job_active)
  {
    pthread_cond_wait(&pool->work_done, &pool->mutex);
  }

  int status = pool->job_status;
  pool->item_count = 0;
  pool->next_index = 0;
  pool->active_threads = 0;
  pool->remaining_threads = 0;
  pool->job_status = 0;
  pool->item_fn = NULL;
  pool->item_context = NULL;
  pthread_mutex_unlock(&pool->mutex);

  if (status != 0)
  {
    LOG_ERROR(failed_event != NULL ? failed_event : "io_pool_job_failed",
              "phase=%s persistent=1", phase != NULL ? phase : "unknown");
    return -1;
  }

  LOG_INFO(done_event != NULL ? done_event : "io_pool_job_done",
           "phase=%s count=%zu persistent=1",
           phase != NULL ? phase : "unknown", item_count);
  return 0;
}

void IoPoolDestroy(IoPool *pool)
{
  if (pool == NULL)
  {
    return;
  }

  pthread_mutex_lock(&pool->mutex);
  pool->stopping = 1;
  pthread_cond_broadcast(&pool->work_ready);
  pthread_mutex_unlock(&pool->mutex);

  for (size_t i = 0; i < pool->num_threads; i++)
  {
    pthread_join(pool->threads[i], NULL);
  }

  LOG_INFO("io_thread_pool_stopped", "kind=%s threads=%zu",
           poolKindName(pool), pool->num_threads);

  free(pool->threads);
  free(pool->workers);
  pthread_cond_destroy(&pool->work_done);
  pthread_cond_destroy(&pool->work_ready);
  pthread_mutex_destroy(&pool->mutex);
  free(pool);
}
