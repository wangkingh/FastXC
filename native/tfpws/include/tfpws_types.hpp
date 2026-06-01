#ifndef TFPWS_TYPES_HPP
#define TFPWS_TYPES_HPP

#include <cstddef>
struct GpuWorkerConfig
{
    std::size_t worker_index;
    int device_id;
    std::size_t memory_budget_bytes;
};

#endif
