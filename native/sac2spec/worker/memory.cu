#include "worker_runtime.hpp"
#include "sac2spec_plan.hpp"
#include "core/filter_response.h"
#include "include/device_memory.cuh"
#include "io/parallel_io.h"
#include "io/stepack_writer.h"

#include <limits.h>
#include <stdlib.h>

extern "C"
{
#include "include/util.h"
}

static void InitWorkerHostSlot(GpuWorker *worker, WorkerHostSlot *slot, int slot_id)
{
    const Sac2SpecPlan *plan = worker->plan;
    slot->slot_id = slot_id;
    slot->node_count = worker->capacity * (size_t)plan->num_ch;

    CpuMalloc((void **)&slot->nodes, slot->node_count * plan->unit_InOutNode_size);
    CpuMalloc((void **)&slot->h_sacdata, slot->node_count * plan->unit_sacdata_size);
    CpuMalloc((void **)&slot->h_spectrum, slot->node_count * plan->unit_spectrum_size);

    size_t sachdSize = sizeof(SACHEAD);
    for (size_t i = 0; i < slot->node_count; i++)
    {
        CpuMalloc((void **)&(slot->nodes[i].sac_hd), sachdSize);

        slot->nodes[i].sac_data = slot->h_sacdata + i * plan->npts;
        slot->nodes[i].spectrum = slot->h_spectrum + i * plan->nspec_output;

        slot->nodes[i].nspec = plan->nspec_output;
        slot->nodes[i].nstep = plan->nstep_valid;
        slot->nodes[i].df = plan->df_output;
        slot->nodes[i].dt = plan->delta;
    }
}

static void FreeWorkerHostSlot(WorkerHostSlot *slot)
{
    if (slot->nodes != NULL)
    {
        for (size_t i = 0; i < slot->node_count; i++)
        {
            if (slot->nodes[i].sac_hd != NULL)
            {
                CpuFree((void **)&slot->nodes[i].sac_hd);
            }
        }
        CpuFree((void **)&slot->nodes);
    }
    if (slot->h_sacdata != NULL)
    {
        CpuFree((void **)&slot->h_sacdata);
    }
    if (slot->h_spectrum != NULL)
    {
        CpuFree((void **)&slot->h_spectrum);
    }
}

void InitWorkerHostMemory(GpuWorker *worker)
{
    const Sac2SpecPlan *plan = worker->plan;
    worker->host_slot_count = plan->host_slot_count;
    if (worker->host_slot_count < 1)
    {
        worker->host_slot_count = 1;
    }

    worker->host_slots = (WorkerHostSlot *)calloc((size_t)worker->host_slot_count,
                                                  sizeof(WorkerHostSlot));
    if (worker->host_slots == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=host_slots count=%d",
                  worker->host_slot_count);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < worker->host_slot_count; i++)
    {
        InitWorkerHostSlot(worker, &worker->host_slots[i], i);
    }

    worker->read_pool = CreateReadIoPool((size_t)worker->io_threads);
    worker->stepack_writer = CreateStepackWriter(plan->stepack_root,
                                             worker->worker_index);
    if (worker->read_pool == NULL || worker->stepack_writer == NULL)
    {
        LOG_ERROR("worker_io_group_init_failed", "gpu=%d io_threads=%d",
                  worker->gpu_id, worker->io_threads);
        exit(EXIT_FAILURE);
    }
}

void FreeWorkerHostMemory(GpuWorker *worker)
{
    if (worker->stepack_writer != NULL)
    {
        DestroyStepackWriter(worker->stepack_writer);
        worker->stepack_writer = NULL;
    }
    if (worker->read_pool != NULL)
    {
        DestroyReadIoPool(worker->read_pool);
        worker->read_pool = NULL;
    }
    if (worker->host_slots != NULL)
    {
        for (int i = 0; i < worker->host_slot_count; i++)
        {
            FreeWorkerHostSlot(&worker->host_slots[i]);
        }
        free(worker->host_slots);
        worker->host_slots = NULL;
    }
}

void InitWorkerDeviceMemory(GpuWorker *worker)
{
    const Sac2SpecPlan *plan = worker->plan;
    worker->frame_capacity = worker->capacity * (size_t)plan->nstep_valid;
    if (worker->frame_capacity > (size_t)INT_MAX ||
        worker->frame_capacity > (size_t)INT_MAX / (size_t)plan->num_ch)
    {
        LOG_ERROR("worker_frame_capacity_too_large",
                  "gpu=%d file_capacity=%zu nstep_valid=%d frame_capacity=%zu",
                  worker->gpu_id, worker->capacity, plan->nstep_valid,
                  worker->frame_capacity);
        exit(EXIT_FAILURE);
    }

    CUDACHECK(cudaSetDevice(worker->gpu_id));
    AllocateGpuMemory((int)worker->frame_capacity, plan->segment_pts, plan->filter_nfft, plan->output_nfft,
                      plan->num_ch, plan->do_runabs, plan->do_runabs_mf,
                      plan->wh_flag,
                      &worker->d_sacdata, &worker->d_spectrum,
                      &worker->d_padded_sacdata, &worker->d_padded_spectrum,
                      &worker->d_base_padded_spectrum,
                      &worker->d_filtered_sacdata,
                      &worker->d_total_sacdata,
                      &worker->d_responses, &worker->d_tmp,
                      &worker->d_weight, &worker->d_tmp_weight,
                      plan->filter_count, &worker->d_sum, &worker->d_isum,
                      &worker->d_cufft_work,
                      &worker->planfwd, &worker->planinv,
                      &worker->planfwd_filter, &worker->planinv_filter,
                      &worker->planfwd_output);
    worker->plans_created = 1;

    for (int i = 0; i < plan->filter_count; i++)
    {
        CUDACHECK(cudaMemcpy(worker->d_responses + i * plan->filter_nfft,
                             plan->filter_responses[i].response,
                             plan->filter_nfft * sizeof(float),
                             cudaMemcpyHostToDevice));
    }
}

void FreeWorkerDeviceMemory(GpuWorker *worker)
{
    CUDACHECK(cudaSetDevice(worker->gpu_id));
    if (worker->plans_created)
    {
        cufftDestroy(worker->planfwd);
        cufftDestroy(worker->planinv);
        cufftDestroy(worker->planfwd_filter);
        cufftDestroy(worker->planinv_filter);
        cufftDestroy(worker->planfwd_output);
        worker->plans_created = 0;
    }
    GpuFree((void **)&worker->d_cufft_work);
    GpuFree((void **)&worker->d_sacdata);
    GpuFree((void **)&worker->d_spectrum);
    GpuFree((void **)&worker->d_padded_sacdata);
    GpuFree((void **)&worker->d_padded_spectrum);
    GpuFree((void **)&worker->d_base_padded_spectrum);
    GpuFree((void **)&worker->d_filtered_sacdata);
    GpuFree((void **)&worker->d_total_sacdata);
    GpuFree((void **)&worker->d_responses);
    GpuFree((void **)&worker->d_weight);
    GpuFree((void **)&worker->d_tmp);
    GpuFree((void **)&worker->d_tmp_weight);
    GpuFree((void **)&worker->d_sum);
    GpuFree((void **)&worker->d_isum);
}
