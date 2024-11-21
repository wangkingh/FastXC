#ifndef _CUDA_UTIL_CUH
#define _CUDA_UTIL_CUH
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <stdio.h>
#include "config.h"
#define BLOCKX 32
#define BLOCKY 32
#define BLOCKMAX 1024

#define CUDACHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        cudaError_t e = cmd;                                   \
        if (e != cudaSuccess)                                  \
        {                                                      \
            printf("Failed: Cuda error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#define CUFFTCHECK(cmd)                              \
    do                                               \
    {                                                \
        cufftResult_t e = cmd;                       \
        if (e != CUFFT_SUCCESS)                      \
        {                                            \
            printf("Failed: CuFFT error %s:%d %d\n", \
                   __FILE__, __LINE__, e);           \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while (0)

/* Compute the block and dimension for cuda kernel fucntion*/
void DimCompute(dim3 *, dim3 *, size_t, size_t);

/* Free cuda pointer */
void GpuFree(void **pptr);

/* Get the infomation of available GPU ram of a certain device */
size_t QueryAvailGpuRam(size_t);

/* Estimate the batch of processing data */
size_t EstimateGpuBatch(size_t gpu_id, int nseg, int num_ch);

/* Allocating Memory of Processing data */
void AllocateGpuMemory(int batch, int nseg, int num_ch,
                       float **d_sacdata, float **d_sac_sum,
                       cuComplex **d_spectrum,
                       float **d_sacdata_2x, cuComplex **d_spectrum_2x,
                       float **d_filtered, float **d_filtered_tmp,
                       float **sac_hist, float **filtered_hist,
                       double **a, double **b, float **d_tmp,
                       float **d_weight, float **d_tmp_weight,
                       int filterCount, double **d_sum, double **d_isum,
                       cufftHandle *planfwd, cufftHandle *planinv, cufftHandle *planfwd_2x);

#endif