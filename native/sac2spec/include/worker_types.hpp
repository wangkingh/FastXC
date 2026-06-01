#ifndef SAC2SPEC_WORKER_TYPES_HPP
#define SAC2SPEC_WORKER_TYPES_HPP

#include "complex.h"

#include <cstddef>
#include <cuComplex.h>
#include <cufft.h>
#include <pthread.h>

typedef struct InOutNode InOutNode;
typedef struct Sac2SpecPlan Sac2SpecPlan;
typedef struct TaskQueue TaskQueue;
typedef struct ThreadPoolRead ThreadPoolRead;
typedef struct SpackWriter SpackWriter;

typedef struct WorkerCapacityPlan
{
    int gpu_id;
    size_t gpu_frame_cap;
    size_t gpu_cap;
    size_t capacity;
    int io_threads;
    int enabled;
} WorkerCapacityPlan;

typedef struct WorkerHostSlot
{
    InOutNode *nodes;
    float *h_sacdata;
    complex *h_spectrum;
    size_t node_count;
    int slot_id;
} WorkerHostSlot;

typedef struct GpuWorker
{
    const Sac2SpecPlan *plan;
    TaskQueue *queue;
    int gpu_id;
    int worker_index;
    size_t capacity;
    size_t frame_capacity;
    int io_threads;
    int failed;

    WorkerHostSlot *host_slots;
    int host_slot_count;

    ThreadPoolRead *read_pool;
    SpackWriter *spack_writer;

    float *d_sacdata;
    float *d_sacdata_2x;
    float *d_filtered_sacdata;
    float *d_total_sacdata;
    cuComplex *d_spectrum;
    cuComplex *d_spectrum_2x;
    cuComplex *d_base_spectrum_2x;
    float *d_weight;
    float *d_tmp;
    float *d_tmp_weight;
    float *d_responses;
    double *d_sum;
    double *d_isum;
    void *d_cufft_work;

    cufftHandle planfwd;
    cufftHandle planinv;
    cufftHandle planfwd_filter;
    cufftHandle planinv_filter;
    cufftHandle planfwd_output;
    int plans_created;
} GpuWorker;

typedef struct GpuWorkerRuntime
{
    const Sac2SpecPlan *plan;
    TaskQueue *queue;
    GpuWorker *workers;
    pthread_t *threads;
    int worker_count;
    int started_count;
} GpuWorkerRuntime;

typedef struct WorkerBatch
{
    GpuWorker *worker;
    WorkerHostSlot *slot;
    size_t start_group;
    size_t group_count;
    size_t file_rows;
    size_t frame_batch;
    size_t frame_rows;
    size_t plan_rows;
    int status;
    int slot_index;
    int thread_active;
    const char *stage_name;
    pthread_t thread;
} WorkerBatch;

#endif
