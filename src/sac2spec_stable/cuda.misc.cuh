#ifndef _CUDA_MISC_CUH
#define _CUDA_MISC_CUH
#include "cuda.util.cuh"
#include <cstddef>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

__global__ void cisnan2DKernel(cuComplex *d_data, size_t pitch, size_t width,
                               size_t height);

__global__ void amp2DKernel(float *d_amp, size_t dpitch, cuComplex *d_data,
                            size_t spitch, size_t width, size_t height);

__global__ void div2DKernel(float *d_data, size_t dpitch, float *d_divisor,
                            size_t spitch, size_t width, size_t height);

__global__ void cdiv2DKernel(cuComplex *d_data, size_t dpitch, float *d_divisor,
                             size_t spitch, size_t width, size_t height);

__global__ void sum2DKernel(float *d_sum, size_t dpitch, float *d_in,
                            size_t spitch, size_t width, size_t height);

__global__ void filterKernel(cuComplex *d_spectrum, cuComplex *d_response, size_t pitch, size_t width, size_t height);

__global__ void onebit2DKernel(float *d_data, size_t pitch, size_t width, size_t height);

__global__ void FwdNormalize2DKernel(cuComplex *d_segspec, size_t pitch, size_t width, size_t height, float dt);

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch, size_t width, size_t height, float dt);

__global__ void smooth2DKernel(float *d_out, int dpitch, float *d_tmp, int spitch, int width, int height, int winsize);

#endif