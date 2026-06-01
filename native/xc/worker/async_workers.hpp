#ifndef XC_ASYNC_WORKERS_HPP
#define XC_ASYNC_WORKERS_HPP

#include "runtime.hpp"

static const int kResidentStartupWaitSeconds = 60;
static const int kResidentRetrySeconds = 2;

struct ResidentWorkerPool
{
  const TimestampWork *timestamp = NULL;
  JobQueue *queue = NULL;
  size_t worker_count = 0;
  size_t ready_count = 0;
  size_t done_count = 0;
  size_t generation = 0;
  bool shutdown = false;
  bool failed = false;
  pthread_mutex_t mutex;
  pthread_cond_t work_ready;
  pthread_cond_t all_ready;
  pthread_cond_t all_done;
};

struct ResidentWorkerContext
{
  WorkerConfig cfg;
  const RuntimeShape *shape = NULL;
  const AllowedPathTable *paths = NULL;
  ResidentWorkerPool *pool = NULL;
  bool ready = false;
  bool stop_before_ready = false;
};

void *resident_gpu_worker_main(void *arg);

void resident_worker_pool_init(ResidentWorkerPool *pool, size_t worker_count);
void resident_worker_pool_destroy(ResidentWorkerPool *pool);
size_t resident_worker_pool_wait_ready_or_timeout(ResidentWorkerPool *pool,
                                                  int timeout_seconds);
size_t resident_worker_pool_select_ready(ResidentWorkerPool *pool,
                                         ResidentWorkerContext *contexts,
                                         size_t context_count);
void resident_worker_pool_submit(ResidentWorkerPool *pool,
                                 const TimestampWork *timestamp,
                                 JobQueue *queue);
bool resident_worker_pool_wait_done(ResidentWorkerPool *pool);
void resident_worker_pool_stop(ResidentWorkerPool *pool);

#endif
