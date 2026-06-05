#include "worker_runtime.hpp"
#include "logger.h"
#include "progress.hpp"
#include "sac2spec_plan.hpp"
#include "parallel_io.h"
#include "stepack_writer.h"

#include <string.h>

int LoadWorkerBatchSac(WorkerBatch *batch)
{
    GpuWorker *worker = batch->worker;
    WorkerHostSlot *slot = batch->slot;
    const Sac2SpecPlan *plan = worker->plan;
    size_t finish_cnt = batch->start_group * (size_t)plan->num_ch;

    if (slot == NULL)
    {
        LOG_ERROR("worker_slot_missing", "stage=read_sac gpu=%d", worker->gpu_id);
        return -1;
    }

    for (size_t i = finish_cnt, j = 0; j < batch->file_rows; i++, j++)
    {
        slot->nodes[j].sacpath = plan->in_paths.paths[i];
        slot->nodes[j].meta = &plan->meta.values[i];
    }

    LOG_INFO("worker_batch_sac_load_start",
             "gpu=%d slot=%d start_group=%zu group_count=%zu file_rows=%zu first_sac=\"%s\" first_timestamp=\"%s\"",
             worker->gpu_id, batch->slot_index, batch->start_group,
             batch->group_count, batch->file_rows,
             batch->file_rows > 0 ? slot->nodes[0].sacpath : "",
             batch->file_rows > 0 ? slot->nodes[0].meta->timestamp : "");

    memset(slot->h_sacdata, 0, batch->file_rows * plan->npts * sizeof(float));
    memset(slot->h_spectrum, 0,
           batch->file_rows * plan->nstep_valid * plan->nspec_output * sizeof(complex));

    if (ReadSacBatchParallel(worker->read_pool, batch->file_rows,
                             slot->nodes, worker->io_threads,
                             plan->npts, plan->delta) != 0)
    {
        LOG_ERROR("worker_sac_load_failed", "gpu=%d start_group=%zu count=%zu",
                  worker->gpu_id, batch->start_group, batch->group_count);
        return -1;
    }

    LOG_INFO("worker_batch_sac_load_done",
             "gpu=%d slot=%d start_group=%zu group_count=%zu file_rows=%zu npts=%d",
             worker->gpu_id, batch->slot_index, batch->start_group, batch->group_count,
             batch->file_rows, plan->npts);
    return 0;
}

int WriteWorkerBatchOutput(WorkerBatch *batch)
{
    GpuWorker *worker = batch->worker;
    WorkerHostSlot *slot = batch->slot;

    if (slot == NULL)
    {
        LOG_ERROR("worker_slot_missing", "stage=write_output gpu=%d", worker->gpu_id);
        return -1;
    }

    LOG_INFO("worker_batch_output_write_start",
             "gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu file_rows=%zu first_timestamp=\"%s\"",
             worker->gpu_id, batch->slot_index, batch->batch_seq,
             batch->start_group, batch->group_count, batch->file_rows,
             batch->file_rows > 0 ? slot->nodes[0].meta->timestamp : "");

    int status = StepackWriterAppendBatch(worker->stepack_writer, slot->nodes,
                                          batch->file_rows, batch->batch_seq,
                                          batch->start_group, batch->group_count);

    if (status != 0)
    {
        LOG_ERROR("worker_output_write_failed", "gpu=%d start_group=%zu count=%zu",
                  worker->gpu_id, batch->start_group, batch->group_count);
        return -1;
    }

    ProgressAdd(worker->plan->progress, batch->file_rows, worker->gpu_id,
                batch->start_group, batch->group_count);

    LOG_INFO("worker_batch_output_write_done",
             "gpu=%d slot=%d batch_seq=%zu start_group=%zu group_count=%zu file_rows=%zu",
             worker->gpu_id, batch->slot_index, batch->batch_seq,
             batch->start_group, batch->group_count, batch->file_rows);
    return 0;
}
