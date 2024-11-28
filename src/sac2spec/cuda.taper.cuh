#ifndef _CUDA_TAPER_H
#define _CUDA_TAPER_H
#include <cuComplex.h>
#include "config.h"
#include "cuda.util.cuh"

__global__ void specTaper2DKernel(cuComplex *d_segspec, size_t pitch,
                                  size_t width, size_t height, int np,
                                  int idx1, int idx2, int idx3, int idx4);

__global__ void timetaper2DKernel(float *d_data, int pitch, int width, int height, int taper_size);

#endif
