#ifndef _CUDA_RDCRTR_CUH
#define _CUDA_RDCRTR_CUH

#include "cuda.util.cuh"
#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void sumSingleBlock2DKernel(double *d_sum, int dpitch,
                                       const float *d_data, int spitch,
                                       int width, int height);

__global__ void isumSingleBlock2DKernel(double *d_isum, int dpitch,
                                        const float *d_data, int spitch,
                                        int width, int height);

__global__ void rdc2DKernel(float *d_data, int pitch, int width, int height,
                            double *d_sum);

__global__ void rtr2DKernel(float *d_data, int pitch, int width, int height,
                            double *d_sum, double *d_isum);

#endif