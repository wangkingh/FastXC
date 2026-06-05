#include "worker_runtime.hpp"
#include "cuda.util.cuh"
#include "logger.h"
#include "sac2spec_plan.hpp"
#include "task_queue.hpp"

#include <stdlib.h>
#include <string.h>

static void InitWorkerBatch(WorkerBatch *batch, GpuWorker *worker,
                            WorkerHostSlot *slot,
                            size_t batch_seq,
                            size_t start_group, size_t group_count)
{
    const Sac2SpecPlan *plan = worker->plan;
    memset(batch, 0, sizeof(*batch));
    batch->worker = worker;
    batch->slot = slot;
    batch->slot_index = slot != NULL ? slot->slot_id : -1;
    batch->batch_seq = batch_seq;
    batch->start_group = start_group;
    batch->group_count = group_count;
    batch->file_rows = group_count * (size_t)plan->num_ch;
    batch->frame_batch = group_count * (size_t)plan->nstep_valid;
    batch->frame_rows = batch->frame_batch * (size_t)plan->num_ch;
    batch->plan_rows = worker->frame_capacity * (size_t)plan->num_ch;
}

static void *ReadBatchThreadMain(void *arg)
{
    WorkerBatch *batch = (WorkerBatch *)arg;
    batch->status = LoadWorkerBatchSac(batch);
    return NULL;
}

static void *WriteBatchThreadMain(void *arg)
{
    WorkerBatch *batch = (WorkerBatch *)arg;
    batch->status = WriteWorkerBatchOutput(batch);
    return NULL;
}

static int StartReadBatch(WorkerBatch *batch)
{
    batch->status = 0;
    batch->stage_name = "read_sac";
    LOG_DEBUG("pipelined_io_read_start",
              "gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu",
              batch->worker->gpu_id, batch->slot_index, batch->batch_seq,
              batch->start_group, batch->group_count);

    int err = pthread_create(&batch->thread, NULL, ReadBatchThreadMain, batch);
    if (err != 0)
    {
        LOG_ERROR("thread_create_failed",
                  "phase=read_sac gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu",
                  batch->worker->gpu_id, batch->slot_index,
                  batch->batch_seq, batch->start_group, batch->group_count);
        return -1;
    }
    batch->thread_active = 1;
    return 0;
}

static int StartWriteBatch(WorkerBatch *batch)
{
    batch->status = 0;
    batch->stage_name = "write_output";
    LOG_DEBUG("pipelined_io_write_start",
              "gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu",
              batch->worker->gpu_id, batch->slot_index, batch->batch_seq,
              batch->start_group, batch->group_count);

    int err = pthread_create(&batch->thread, NULL, WriteBatchThreadMain, batch);
    if (err != 0)
    {
        LOG_ERROR("thread_create_failed",
                  "phase=write_output gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu",
                  batch->worker->gpu_id, batch->slot_index,
                  batch->batch_seq, batch->start_group, batch->group_count);
        return -1;
    }
    batch->thread_active = 1;
    return 0;
}

static int WaitBatchThread(WorkerBatch *batch, const char *stage_name)
{
    const char *phase = batch->stage_name != NULL ? batch->stage_name : stage_name;
    if (!batch->thread_active)
    {
        return 0;
    }

    if (pthread_join(batch->thread, NULL) != 0)
    {
        LOG_ERROR("thread_join_failed",
                  "phase=%s gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu",
                  phase, batch->worker->gpu_id,
                  batch->slot_index, batch->batch_seq,
                  batch->start_group, batch->group_count);
        batch->thread_active = 0;
        return -1;
    }
    batch->thread_active = 0;

    if (batch->status != 0)
    {
        LOG_ERROR("pipelined_io_thread_failed",
                  "stage=%s gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu status=%d",
                  phase, batch->worker->gpu_id, batch->slot_index,
                  batch->batch_seq, batch->start_group, batch->group_count,
                  batch->status);
        return -1;
    }

    LOG_DEBUG("pipelined_io_thread_done",
              "stage=%s gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu",
              phase, batch->worker->gpu_id,
              batch->slot_index, batch->batch_seq,
              batch->start_group, batch->group_count);
    return 0;
}

static int WaitReadBatch(WorkerBatch *batch)
{
    return WaitBatchThread(batch, "read_sac");
}

static int WaitWriteBatch(WorkerBatch *batch)
{
    return WaitBatchThread(batch, "write_output");
}

static int RunWorkerSequentialBatches(GpuWorker *worker)
{
    size_t start_group = 0;
    size_t group_count = 0;
    size_t batch_seq = 0;
    while (TaskQueuePop(worker->queue, worker->capacity,
                        &start_group, &group_count, &batch_seq))
    {
        LOG_INFO("gpu_worker_batch_start",
                 "gpu=%d batch_seq=%zu start_group=%zu group_count=%zu file_capacity=%zu",
                 worker->gpu_id, batch_seq, start_group, group_count, worker->capacity);

        WorkerBatch batch;
        InitWorkerBatch(&batch, worker, &worker->host_slots[0],
                        batch_seq, start_group, group_count);

        if (LoadWorkerBatchSac(&batch) != 0 ||
            ComputeWorkerBatch(&batch) != 0 ||
            WriteWorkerBatchOutput(&batch) != 0)
        {
            return -1;
        }

        LOG_INFO("gpu_worker_batch_done",
                 "gpu=%d batch_seq=%zu start_group=%zu group_count=%zu",
                 worker->gpu_id, batch_seq, start_group, group_count);
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

static void MarkBatchSlotFree(WorkerBatch *batch, int *slot_busy)
{
    if (batch != NULL && batch->slot_index >= 0)
    {
        slot_busy[batch->slot_index] = 0;
    }
}

static int StartNextReadBatch(GpuWorker *worker, WorkerBatch *batches,
                              int *slot_busy, WorkerBatch **out_batch)
{
    size_t start_group = 0;
    size_t group_count = 0;
    size_t batch_seq = 0;
    *out_batch = NULL;

    if (!TaskQueuePop(worker->queue, worker->capacity,
                      &start_group, &group_count, &batch_seq))
    {
        return 0;
    }

    int slot_index = FindFreeHostSlot(slot_busy, worker->host_slot_count);
    if (slot_index < 0)
    {
        LOG_ERROR("pipelined_io_no_free_slot",
                  "gpu=%d host_slots=%d start_group=%zu group_count=%zu",
                  worker->gpu_id, worker->host_slot_count,
                  start_group, group_count);
        return -1;
    }

    slot_busy[slot_index] = 1;
    InitWorkerBatch(&batches[slot_index], worker, &worker->host_slots[slot_index],
                    batch_seq, start_group, group_count);

    LOG_INFO("pipelined_io_read_scheduled",
             "gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu",
             worker->gpu_id, slot_index, batch_seq, start_group, group_count);

    if (StartReadBatch(&batches[slot_index]) != 0)
    {
        slot_busy[slot_index] = 0;
        return -1;
    }

    *out_batch = &batches[slot_index];
    return 1;
}

static int WaitWriteBatchAndReleaseSlot(WorkerBatch **write_batch, int *slot_busy)
{
    if (*write_batch == NULL)
    {
        return 0;
    }

    int rc = WaitWriteBatch(*write_batch);
    MarkBatchSlotFree(*write_batch, slot_busy);
    LOG_INFO("pipelined_io_slot_released",
             "gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu",
             (*write_batch)->worker->gpu_id, (*write_batch)->slot_index,
             (*write_batch)->batch_seq,
             (*write_batch)->start_group, (*write_batch)->group_count);
    *write_batch = NULL;
    return rc;
}

static int RunWorkerPipelinedBatches(GpuWorker *worker)
{
    int slot_count = worker->host_slot_count;
    if (slot_count < 3)
    {
        LOG_ERROR("pipelined_io_requires_host_slots",
                  "gpu=%d host_slots=%d required=3",
                  worker->gpu_id, slot_count);
        return -1;
    }

    WorkerBatch *batches = (WorkerBatch *)calloc((size_t)slot_count, sizeof(WorkerBatch));
    int *slot_busy = (int *)calloc((size_t)slot_count, sizeof(int));
    if (batches == NULL || slot_busy == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=pipelined_io_state slots=%d", slot_count);
        free(batches);
        free(slot_busy);
        return -1;
    }

    /* Pipelined mode keeps three host slots in flight:
     * read next, compute current, and write previous.
     */
    LOG_INFO("pipelined_io_worker_start",
             "gpu=%d host_slots=%d file_capacity=%zu",
             worker->gpu_id, slot_count, worker->capacity);

    WorkerBatch *current = NULL;
    WorkerBatch *write_batch = NULL;

    int rc = StartNextReadBatch(worker, batches, slot_busy, &current);
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

    if (WaitReadBatch(current) != 0)
    {
        MarkBatchSlotFree(current, slot_busy);
        free(batches);
        free(slot_busy);
        return -1;
    }

    while (current != NULL)
    {
        WorkerBatch *next = NULL;
        rc = StartNextReadBatch(worker, batches, slot_busy, &next);
        if (rc < 0)
        {
            if (write_batch != NULL)
            {
                (void)WaitWriteBatchAndReleaseSlot(&write_batch, slot_busy);
            }
            MarkBatchSlotFree(current, slot_busy);
            free(batches);
            free(slot_busy);
            return -1;
        }

        LOG_INFO("pipelined_io_compute_current",
                 "gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu next_prefetch=%d",
                 worker->gpu_id, current->slot_index,
                 current->batch_seq, current->start_group, current->group_count,
                 next != NULL);

        if (ComputeWorkerBatch(current) != 0)
        {
            if (next != NULL)
            {
                (void)WaitReadBatch(next);
                MarkBatchSlotFree(next, slot_busy);
            }
            if (write_batch != NULL)
            {
                (void)WaitWriteBatchAndReleaseSlot(&write_batch, slot_busy);
            }
            MarkBatchSlotFree(current, slot_busy);
            free(batches);
            free(slot_busy);
            return -1;
        }

        if (WaitWriteBatchAndReleaseSlot(&write_batch, slot_busy) != 0)
        {
            if (next != NULL)
            {
                (void)WaitReadBatch(next);
                MarkBatchSlotFree(next, slot_busy);
            }
            MarkBatchSlotFree(current, slot_busy);
            free(batches);
            free(slot_busy);
            return -1;
        }

        if (StartWriteBatch(current) != 0)
        {
            if (next != NULL)
            {
                (void)WaitReadBatch(next);
                MarkBatchSlotFree(next, slot_busy);
            }
            MarkBatchSlotFree(current, slot_busy);
            free(batches);
            free(slot_busy);
            return -1;
        }
        write_batch = current;

        if (next != NULL)
        {
            if (WaitReadBatch(next) != 0)
            {
                (void)WaitWriteBatchAndReleaseSlot(&write_batch, slot_busy);
                MarkBatchSlotFree(next, slot_busy);
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

    rc = WaitWriteBatchAndReleaseSlot(&write_batch, slot_busy);
    LOG_INFO("pipelined_io_worker_done", "gpu=%d status=%d", worker->gpu_id, rc);

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
        if (RunWorkerPipelinedBatches(worker) != 0)
        {
            worker->failed = 1;
        }
    }
    else
    {
        if (RunWorkerSequentialBatches(worker) != 0)
        {
            worker->failed = 1;
        }
    }

    CUDACHECK(cudaDeviceSynchronize());
    FreeWorkerDeviceMemory(worker);
    FreeWorkerHostMemory(worker);

    LOG_INFO("gpu_worker_done", "gpu=%d failed=%d", worker->gpu_id, worker->failed);
    return NULL;
}
