#ifndef _CU_PWS_UTIL_H_
#define _CU_PWS_UTIL_H_
#include <cuComplex.h>
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void cudaMean(cufftComplex *hilbert_complex, cufftComplex *mean, size_t num_trace, size_t nfft);

__global__ void cudaNormalizeComplex(cufftComplex *hilbert_complex, size_t data_num, size_t nfft);

__global__ void cudaMultiply(float *linear_stack, cuComplex *weight, float *pws_stack, size_t nfft);

#endif