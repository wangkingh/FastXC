#ifndef GPU_BUDGET_HPP
#define GPU_BUDGET_HPP

#include <cstddef>

static inline std::size_t mib_to_bytes(std::size_t mib)
{
    return mib * 1024 * 1024;
}

static inline std::size_t estimate_auto_worker_gpu_budget_bytes(std::size_t gpu_free_bytes,
                                                                float auto_fraction,
                                                                std::size_t physical_worker_count)
{
    std::size_t auto_available = (std::size_t)(gpu_free_bytes * auto_fraction);
    if (physical_worker_count > 1)
        auto_available /= physical_worker_count;
    return auto_available;
}

static inline std::size_t estimate_worker_gpu_budget_bytes(std::size_t gpu_free_bytes,
                                                           float auto_fraction,
                                                           std::size_t physical_worker_count,
                                                           std::size_t worker_memory_limit_mib)
{
    std::size_t available = estimate_auto_worker_gpu_budget_bytes(gpu_free_bytes,
                                                                  auto_fraction,
                                                                  physical_worker_count);

    if (worker_memory_limit_mib > 0)
    {
        std::size_t manual_limit = mib_to_bytes(worker_memory_limit_mib);
        if (manual_limit < available)
            available = manual_limit;
    }

    return available;
}

#endif
