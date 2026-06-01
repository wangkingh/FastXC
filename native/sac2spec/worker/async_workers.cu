#include "worker_runtime.hpp"
#include "logger.h"

#include <stdlib.h>
#include <string.h>

int InitGpuWorkerRuntime(GpuWorkerRuntime *runtime,
                         const Sac2SpecPlan *plan,
                         TaskQueue *queue,
                         const WorkerCapacityPlan *plans,
                         int worker_count)
{
    memset(runtime, 0, sizeof(*runtime));
    runtime->plan = plan;
    runtime->queue = queue;
    runtime->worker_count = worker_count;

    if (worker_count < 1)
    {
        LOG_ERROR("gpu_worker_runtime_empty", "worker_count=%d", worker_count);
        return -1;
    }

    runtime->workers = (GpuWorker *)calloc((size_t)worker_count, sizeof(GpuWorker));
    runtime->threads = (pthread_t *)calloc((size_t)worker_count, sizeof(pthread_t));
    if (runtime->workers == NULL || runtime->threads == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=gpu_worker_runtime bytes=%zu",
                  (size_t)worker_count * (sizeof(GpuWorker) + sizeof(pthread_t)));
        DestroyGpuWorkerRuntime(runtime);
        return -1;
    }

    for (int i = 0; i < worker_count; i++)
    {
        runtime->workers[i].plan = plan;
        runtime->workers[i].queue = queue;
        runtime->workers[i].gpu_id = plans[i].gpu_id;
        runtime->workers[i].worker_index = i;
        runtime->workers[i].capacity = plans[i].capacity;
        runtime->workers[i].io_threads = plans[i].io_threads;
    }

    return 0;
}

int StartGpuWorkerThreads(GpuWorkerRuntime *runtime)
{
    for (int i = 0; i < runtime->worker_count; i++)
    {
        GpuWorker *worker = &runtime->workers[i];
        int err = pthread_create(&runtime->threads[i], NULL,
                                 RunGpuWorkerThreadMain, worker);
        if (err != 0)
        {
            LOG_ERROR("thread_create_failed",
                      "phase=gpu_worker index=%d gpu=%d",
                      i, worker->gpu_id);
            return -1;
        }
        runtime->started_count++;

        LOG_INFO("gpu_worker_thread_started",
                 "index=%d gpu=%d capacity=%zu io_threads=%d",
                 worker->worker_index, worker->gpu_id, worker->capacity, worker->io_threads);
    }

    return 0;
}

int JoinGpuWorkerThreads(GpuWorkerRuntime *runtime)
{
    int failed = 0;
    for (int i = 0; i < runtime->started_count; i++)
    {
        GpuWorker *worker = &runtime->workers[i];
        if (pthread_join(runtime->threads[i], NULL) != 0)
        {
            LOG_ERROR("thread_join_failed",
                      "phase=gpu_worker index=%d gpu=%d",
                      i, worker->gpu_id);
            failed = 1;
        }
        if (worker->failed)
        {
            failed = 1;
        }
    }

    runtime->started_count = 0;
    return failed ? -1 : 0;
}

void DestroyGpuWorkerRuntime(GpuWorkerRuntime *runtime)
{
    if (runtime == NULL)
    {
        return;
    }

    free(runtime->threads);
    free(runtime->workers);
    runtime->threads = NULL;
    runtime->workers = NULL;
    runtime->worker_count = 0;
    runtime->started_count = 0;
    runtime->plan = NULL;
    runtime->queue = NULL;
}
