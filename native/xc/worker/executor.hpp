#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "runtime.hpp"

void finalize_xc_batch(GpuBuffers *buf,
                       const RuntimeShape *shape,
                       size_t task_count);

#endif
