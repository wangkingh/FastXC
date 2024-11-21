#ifndef _CUDA_XC_DUAL_CUH
#define _CUDA_XC_DUAL_CUH

#include "cuda.util.cuh"
#include <cuComplex.h>
#include <cuda_runtime.h>

__global__ void complexMul2DKernel(cuComplex *src_vec, cuComplex *sta_vec, size_t spitch,
                                   cuComplex *ncf_vec, size_t dpitch,
                                   size_t width, size_t height);

__global__ void generateSignVector(int *sgn_vec, size_t width);

__global__ void applyPhaseShiftKernel(cuComplex *result_vec, int *sign_vec, size_t spitch, size_t width, size_t height);

__global__ void sum2DKernel(float *d_finalccvec, int dpitch, float *d_segncfvec,
                            int spitch, size_t width, size_t height, int nstep);

__global__ void csum2DKernel(cuComplex *d_total_spectrum, int dpitch,
                             cuComplex *d_segment_spectrum, int spitch,
                             size_t width, size_t height, int step_idx, int nstep);

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt);

#endif