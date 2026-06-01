#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cstddef>
#include <cuComplex.h>
#include <cuda_runtime.h>

__global__ void generateSignVector(int *sgn_vec, size_t width);

__global__ void accumulateStepXc2DKernel(const cuComplex *__restrict__ d_spec_buffer,
                                         const size_t *__restrict__ src_idx_list,
                                         const size_t *__restrict__ rec_idx_list,
                                         cuComplex *__restrict__ d_stack,
                                         size_t nspec, float scale, size_t pair_count);

__global__ void applyPhaseShiftKernel(cuComplex *ncf_vec, int *sgn_vec,
                                      size_t spitch, size_t width, size_t height);

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt);

#endif
