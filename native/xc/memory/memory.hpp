#ifndef MEMORY_HPP
#define MEMORY_HPP

#include "runtime.hpp"

extern "C"
{
#include "arguproc.h"
}

bool compute_memory_plan(size_t block_files,
                         const RuntimeShape *shape,
                         size_t lazy_write_depth,
                         MemoryPlan *plan);

bool compute_pair_capacity_for_block(size_t block_files, size_t *pair_capacity);

size_t estimate_block_files_for_worker(const ARGUTYPE *args,
                                       size_t worker_index,
                                       const RuntimeShape *shape);

int init_gpu_buffers(GpuBuffers *buf,
                     const WorkerConfig *cfg,
                     const RuntimeShape *shape);

void free_gpu_buffers(GpuBuffers *buf);

#endif
