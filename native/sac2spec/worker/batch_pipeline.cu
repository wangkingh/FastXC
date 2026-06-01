#include "worker_runtime.hpp"
#include "cuda.util.cuh"
#include "logger.h"
#include "sac2spec_plan.hpp"
#include "task_queue.hpp"

#include <stdlib.h>
#include <string.h>

typedef void *(*BatchStageEntry)(void *);

static void InitWorkerBatch(WorkerBatch *batch, GpuWorker *worker,
                            WorkerHostSlot *slot,
                            size_t start_group, size_t group_count)
{
    const Sac2SpecPlan *plan = worker->plan;
    memset(batch, 0, sizeof(*batch));
    batch->worker = worker;
    batch->slot = slot;
    batch->slot_index = slot != NULL ? slot->slot_id : -1;
    batch->start_group = start_group;
    batch->group_count = group_count;
    batch->file_rows = group_count * (size_t)plan->num_ch;
    batch->frame_batch = group_count * (size_t)plan->nstep_valid;
    batch->frame_rows = batch->frame_batch * (size_t)plan->num_ch;
    batch->plan_rows = worker->frame_capacity * (size_t)plan->num_ch;
}

static void *LoadSacBatchStageMain(void *arg)
{
    WorkerBatch *batch = (WorkerBatch *)arg;
    batch->status = LoadWorkerBatchSac(batch);
    return NULL;
}

static void *ComputeBatchStageMain(void *arg)
{
    WorkerBatch *batch = (WorkerBatch *)arg;
    batch->status = ComputeWorkerBatch(batch);
    return NULL;
}

static void *WriteOutputBatchStageMain(void *arg)
{
    WorkerBatch *batch = (WorkerBatch *)arg;
    batch->status = WriteWorkerBatchOutput(batch);
    return NULL;
}

static int StartBatchStage(WorkerBatch *batch, const char *stage_name,
                           BatchStageEntry entry)
{
    batch->status = 0;
    batch->stage_name = stage_name;
    LOG_DEBUG("batch_stage_start",
              "stage=%s gpu=%d slot=%d start_group=%zu group_count=%zu",
              stage_name, batch->worker->gpu_id,
              batch->slot_index, batch->start_group, batch->group_count);

    int err = pthread_create(&batch->thread, NULL, entry, batch);
    if (err != 0)
    {
        LOG_ERROR("thread_create_failed",
                  "phase=%s gpu=%d slot=%d start_group=%zu group_count=%zu",
                  stage_name, batch->worker->gpu_id,
                  batch->slot_index, batch->start_group, batch->group_count);
        return -1;
    }
    batch->thread_active = 1;
    return 0;
}

static int JoinBatchStage(WorkerBatch *batch)
{
    const char *stage_name = batch->stage_name != NULL ? batch->stage_name : "unknown";
    if (!batch->thread_active)
    {
        return 0;
    }

    if (pthread_join(batch->thread, NULL) != 0)
    {
        LOG_ERROR("thread_join_failed",
                  "phase=%s gpu=%d slot=%d start_group=%zu group_count=%zu",
                  stage_name, batch->worker->gpu_id,
                  batch->slot_index, batch->start_group, batch->group_count);
        batch->thread_active = 0;
        return -1;
    }
    batch->thread_active = 0;

    if (batch->status != 0)
    {
        LOG_ERROR("batch_stage_failed",
                  "stage=%s gpu=%d slot=%d start_group=%zu group_count=%zu status=%d",
                  stage_name, batch->worker->gpu_id, batch->slot_index,
                  batch->start_group, batch->group_count, batch->status);
        return -1;
    }

    LOG_DEBUG("batch_stage_done",
              "stage=%s gpu=%d slot=%d start_group=%zu group_count=%zu",
              stage_name, batch->worker->gpu_id,
              batch->slot_index, batch->start_group, batch->group_count);
    return 0;
}

static int RunBatchStage(WorkerBatch *batch, const char *stage_name,
                         BatchStageEntry entry)
{
    if (StartBatchStage(batch, stage_name, entry) != 0)
    {
        return -1;
    }
    return JoinBatchStage(batch);
}

static int ProcessWorkerBatch(GpuWorker *worker, size_t start_group, size_t group_count)
{
    WorkerBatch batch;
    InitWorkerBatch(&batch, worker, &worker->host_slots[0], start_group, group_count);

    if (RunBatchStage(&batch, "read_sac", LoadSacBatchStageMain) != 0)
    {
        return -1;
    }
    if (RunBatchStage(&batch, "compute", ComputeBatchStageMain) != 0)
    {
        return -1;
    }
    if (RunBatchStage(&batch, "write_output", WriteOutputBatchStageMain) != 0)
    {
        return -1;
    }

    return 0;
}

static int FindFreeHostSlot(const int *slot_busy, int slot_count)
{
    for (int i = 0; i < slot_count; i++)
    {
        if (!slot_busy[i])
        {
            return i;
        }
    }
    return -1;
}

static int ScheduleLazyRead(GpuWorker *worker, WorkerBatch *batches,
                            int *slot_busy, WorkerBatch **out_batch)
{
    size_t start_group = 0;
    size_t group_count = 0;
    *out_batch = NULL;

    if (!TaskQueuePop(worker->queue, worker->capacity, &start_group, &group_count))
    {
        return 0;
    }

    int slot_index = FindFreeHostSlot(slot_busy, worker->host_slot_count);
    if (slot_index < 0)
    {
        LOG_ERROR("lazy_async_no_free_slot",
                  "gpu=%d host_slots=%d start_group=%zu group_count=%zu",
                  worker->gpu_id, worker->host_slot_count,
                  start_group, group_count);
        return -1;
    }

    slot_busy[slot_index] = 1;
    InitWorkerBatch(&batches[slot_index], worker, &worker->host_slots[slot_index],
                    start_group, group_count);

    LOG_INFO("lazy_async_read_scheduled",
             "gpu=%d slot=%d start_group=%zu group_count=%zu",
             worker->gpu_id, slot_index, start_group, group_count);

    if (StartBatchStage(&batches[slot_index], "read_sac", LoadSacBatchStageMain) != 0)
    {
        slot_busy[slot_index] = 0;
        return -1;
    }

    *out_batch = &batches[slot_index];
    return 1;
}

static int ReleaseLazyWrite(WorkerBatch **write_batch, int *slot_busy)
{
    if (*write_batch == NULL)
    {
        return 0;
    }

    if (JoinBatchStage(*write_batch) != 0)
    {
        return -1;
    }

    slot_busy[(*write_batch)->slot_index] = 0;
    LOG_INFO("lazy_async_slot_released",
             "gpu=%d slot=%d start_group=%zu group_count=%zu",
             (*write_batch)->worker->gpu_id, (*write_batch)->slot_index,
             (*write_batch)->start_group, (*write_batch)->group_count);
    *write_batch = NULL;
    return 0;
}

static int ProcessWorkerBatchesLazyAsync(GpuWorker *worker)
{
    int slot_count = worker->host_slot_count;
    WorkerBatch *batches = (WorkerBatch *)calloc((size_t)slot_count, sizeof(WorkerBatch));
    int *slot_busy = (int *)calloc((size_t)slot_count, sizeof(int));
    if (batches == NULL || slot_busy == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=lazy_async_state slots=%d", slot_count);
        free(batches);
        free(slot_busy);
        return -1;
    }

    LOG_INFO("lazy_async_worker_start",
             "gpu=%d host_slots=%d file_capacity=%zu",
             worker->gpu_id, slot_count, worker->capacity);

    WorkerBatch *current = NULL;
    WorkerBatch *write_batch = NULL;

    int rc = ScheduleLazyRead(worker, batches, slot_busy, &current);
    if (rc < 0)
    {
        free(batches);
        free(slot_busy);
        return -1;
    }
    if (rc == 0)
    {
        free(batches);
        free(slot_busy);
        return 0;
    }

    if (JoinBatchStage(current) != 0)
    {
        slot_busy[current->slot_index] = 0;
        free(batches);
        free(slot_busy);
        return -1;
    }

    while (current != NULL)
    {
        WorkerBatch *next = NULL;
        rc = ScheduleLazyRead(worker, batches, slot_busy, &next);
        if (rc < 0)
        {
            if (write_batch != NULL)
            {
                (void)ReleaseLazyWrite(&write_batch, slot_busy);
            }
            free(batches);
            free(slot_busy);
            return -1;
        }

        LOG_INFO("lazy_async_compute_current",
                 "gpu=%d slot=%d start_group=%zu group_count=%zu next_prefetch=%d",
                 worker->gpu_id, current->slot_index,
                 current->start_group, current->group_count,
                 next != NULL);

        if (ComputeWorkerBatch(current) != 0)
        {
            if (next != NULL)
            {
                (void)JoinBatchStage(next);
                slot_busy[next->slot_index] = 0;
            }
            if (write_batch != NULL)
            {
                (void)ReleaseLazyWrite(&write_batch, slot_busy);
            }
            free(batches);
            free(slot_busy);
            return -1;
        }

        if (ReleaseLazyWrite(&write_batch, slot_busy) != 0)
        {
            if (next != NULL)
            {
                (void)JoinBatchStage(next);
                slot_busy[next->slot_index] = 0;
            }
            free(batches);
            free(slot_busy);
            return -1;
        }

        if (StartBatchStage(current, "write_output", WriteOutputBatchStageMain) != 0)
        {
            if (next != NULL)
            {
                (void)JoinBatchStage(next);
                slot_busy[next->slot_index] = 0;
            }
            slot_busy[current->slot_index] = 0;
            free(batches);
            free(slot_busy);
            return -1;
        }
        write_batch = current;

        if (next != NULL)
        {
            if (JoinBatchStage(next) != 0)
            {
                (void)ReleaseLazyWrite(&write_batch, slot_busy);
                slot_busy[next->slot_index] = 0;
                free(batches);
                free(slot_busy);
                return -1;
            }
            current = next;
        }
        else
        {
            current = NULL;
        }
    }

    rc = ReleaseLazyWrite(&write_batch, slot_busy);
    LOG_INFO("lazy_async_worker_done", "gpu=%d status=%d", worker->gpu_id, rc);

    free(batches);
    free(slot_busy);
    return rc;
}

void *RunGpuWorkerThreadMain(void *arg)
{
    GpuWorker *worker = (GpuWorker *)arg;

    LOG_INFO("gpu_worker_start", "gpu=%d file_capacity=%zu io_threads=%d",
             worker->gpu_id, worker->capacity, worker->io_threads);
    CUDACHECK(cudaSetDevice(worker->gpu_id));

    InitWorkerHostMemory(worker);
    InitWorkerDeviceMemory(worker);

    if (worker->plan->lazy_async)
    {
        if (ProcessWorkerBatchesLazyAsync(worker) != 0)
        {
            worker->failed = 1;
        }
    }
    else
    {
        size_t start_group = 0;
        size_t group_count = 0;
        while (TaskQueuePop(worker->queue, worker->capacity, &start_group, &group_count))
        {
            LOG_INFO("gpu_worker_batch_start",
                     "gpu=%d start_group=%zu group_count=%zu file_capacity=%zu",
                     worker->gpu_id, start_group, group_count, worker->capacity);
            if (ProcessWorkerBatch(worker, start_group, group_count) != 0)
            {
                worker->failed = 1;
                break;
            }
            LOG_INFO("gpu_worker_batch_done",
                     "gpu=%d start_group=%zu group_count=%zu",
                     worker->gpu_id, start_group, group_count);
        }
    }

    CUDACHECK(cudaDeviceSynchronize());
    FreeWorkerDeviceMemory(worker);
    FreeWorkerHostMemory(worker);

    LOG_INFO("gpu_worker_done", "gpu=%d failed=%d", worker->gpu_id, worker->failed);
    return NULL;
}
