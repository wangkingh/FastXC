#ifndef CUDA_FILTER_CUH
#include <cstddef>

__global__ void filterTKernel(float *d_sac, float *d_filtered, double *a, double *b, float *sac_hist, float *filtered_hist, size_t width, size_t height);
__global__ void reverseKernel(const float *d_input, float *d_output, size_t width, size_t height);
#endif
