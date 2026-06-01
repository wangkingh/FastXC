#ifndef MEMORY_ESTIMATE_HPP
#define MEMORY_ESTIMATE_HPP

#include <algorithm>
#include <cstddef>

#include "cuda.util.cuh"

static inline std::size_t estimate_pws_batch_gpu_bytes(std::size_t nsamples,
                                                       std::size_t total_groups,
                                                       std::size_t num_pairs)
{
    std::size_t n = nsamples;
    std::size_t t = total_groups;
    std::size_t f = num_pairs;
    std::size_t r2c_ws = EstimateCufftWorkspace1D(n, t, CUFFT_R2C);
    std::size_t c2c_ws = EstimateCufftWorkspace1D(n, t, CUFFT_C2C);

    std::size_t forward_stage = 4 * f * n + 12 * t * n + 4 * t + r2c_ws;
    std::size_t pws_stage = 16 * f * n + 16 * t * n + 4 * t + c2c_ws;

    return std::max(forward_stage, pws_stage);
}

#endif
