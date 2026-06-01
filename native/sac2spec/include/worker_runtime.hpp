#ifndef SAC2SPEC_WORKER_RUNTIME_HPP
#define SAC2SPEC_WORKER_RUNTIME_HPP

#include "worker_types.hpp"

int PlanWorkerCapacities(WorkerCapacityPlan *plans, int gpu_worker_count,
                         size_t total_groups, size_t host_cap,
                         int total_io_threads);

void InitWorkerHostMemory(GpuWorker *worker);
void FreeWorkerHostMemory(GpuWorker *worker);
void InitWorkerDeviceMemory(GpuWorker *worker);
void FreeWorkerDeviceMemory(GpuWorker *worker);

int LoadWorkerBatchSac(WorkerBatch *batch);
int ComputeWorkerBatch(WorkerBatch *batch);
int WriteWorkerBatchOutput(WorkerBatch *batch);

int InitGpuWorkerRuntime(GpuWorkerRuntime *runtime,
                         const Sac2SpecPlan *plan,
                         TaskQueue *queue,
                         const WorkerCapacityPlan *plans,
                         int worker_count);
int StartGpuWorkerThreads(GpuWorkerRuntime *runtime);
int JoinGpuWorkerThreads(GpuWorkerRuntime *runtime);
void DestroyGpuWorkerRuntime(GpuWorkerRuntime *runtime);

void *RunGpuWorkerThreadMain(void *arg);

#endif
