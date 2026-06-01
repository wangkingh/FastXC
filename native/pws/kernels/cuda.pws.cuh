#ifndef _CU_PWS_H_
#define _CU_PWS_H_
#include <cstddef>
#include <cuComplex.h>
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void hilbertTransformKernel(cufftComplex *d_inputSpectrum,
                                       size_t freqDomainSize,
                                       size_t nTraces);

__global__ void cudaWeightedMeanBatch(cufftComplex *hilbert_complex,
                                      const float *trace_weights,
                                      const size_t *pair_group_offsets,
                                      const size_t *pair_group_counts,
                                      cufftComplex *mean,
                                      size_t num_pairs,
                                      size_t nfft);

__global__ void cudaNormalizeComplex(cufftComplex *hilbert_complex, size_t data_num, size_t nfft);

__global__ void cudaMultiplyBatch(float *linear_stack,
                                  cuComplex *weight,
                                  float *pws_stack,
                                  size_t num_pairs,
                                  size_t nfft);

#endif
