#ifndef _SAC2SPEC_REAL_MATRIX_CUH
#define _SAC2SPEC_REAL_MATRIX_CUH

#include "cuda.util.cuh"
#include <cstddef>

__global__ void abs2DKernel(float *d_data, size_t pitch, size_t width,
                            size_t height);

__global__ void clampmin2DKernel(float *d_data, size_t pitch, size_t width,
                                 size_t height, float minval);

__global__ void onebit2DKernel(float *d_data, size_t pitch, size_t width,
                               size_t height);

__global__ void cutmax2DKernel(float *d_data, size_t pitch, size_t width,
                               size_t height, float maxval);

__global__ void isnan2DKernel(float *d_data, size_t pitch, size_t width,
                              size_t height);

__global__ void div2DKernel(float *d_data, size_t dpitch, float *d_divisor,
                            size_t spitch, size_t width, size_t height);

__global__ void sum2DKernel(float *d_sum, size_t dpitch, float *d_in,
                            size_t spitch, size_t width, size_t height);

__global__ void expandSharedWeight2DKernel(float *d_weight_full, size_t full_pitch,
                                           const float *d_weight_shared, size_t shared_pitch,
                                           size_t width, size_t height, int num_ch);

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt);

__global__ void smoothRowsRollingKernel(float *d_out, int dpitch, const float *d_tmp,
                                        int spitch, int width, int height, int winsize);

#endif
