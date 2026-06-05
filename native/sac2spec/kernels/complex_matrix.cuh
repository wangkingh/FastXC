#ifndef _SAC2SPEC_COMPLEX_MATRIX_CUH
#define _SAC2SPEC_COMPLEX_MATRIX_CUH

#include "cuda.util.cuh"
#include <cstddef>
#include <cuComplex.h>

__global__ void cisnan2DKernel(cuComplex *d_data, size_t pitch, size_t width,
                               size_t height);

__global__ void amp2DKernel(float *d_amp, size_t dpitch, cuComplex *d_data,
                            size_t spitch, size_t width, size_t height);

__global__ void cdiv2DKernel(cuComplex *d_data, size_t dpitch, float *d_divisor,
                             size_t spitch, size_t width, size_t height);

__global__ void spectralOnebit2DKernel(cuComplex *d_data, size_t pitch,
                                       size_t width, size_t height,
                                       float minval);

__global__ void filterKernel(cuComplex *d_spectrum, const float *d_response,
                             size_t pitch, size_t width, size_t height);

__global__ void FwdNormalize2DKernel(cuComplex *d_segspec, size_t pitch,
                                     size_t width, size_t height, float dt);

#endif
