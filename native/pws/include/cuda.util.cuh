#ifndef _CUDA_UTIL_CUH
#define _CUDA_UTIL_CUH

#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "logger.h"

// 3060 gpu do not have enough resources, so I try modify the block size to 16
#define BLOCKX1D 256
#define BLOCKX2D 16
#define BLOCKY2D 16

#define CUDACHECK(cmd)                                              \
  do                                                                \
  {                                                                 \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess)                                           \
    {                                                               \
      LOG_ERROR("cuda_call_failed", "file=%s line=%d error=\"%s\"", __FILE__, __LINE__, \
                cudaGetErrorString(e));                             \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#define CUFFTCHECK(cmd)                                                \
  do                                                                   \
  {                                                                    \
    cufftResult_t e = cmd;                                             \
    if (e != CUFFT_SUCCESS)                                            \
    {                                                                  \
      LOG_ERROR("cufft_call_failed", "file=%s line=%d code=%d", __FILE__, __LINE__, (int)e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

/* Compute the block and dimension for cuda kernel fucntion*/
void DimCompute1D(dim3 *pdimgrd, dim3 *pdimblk, size_t width);
void DimCompute2D(dim3 *, dim3 *, size_t, size_t);

size_t EstimateCufftWorkspace1D(size_t nsamples, size_t batch, cufftType type);
size_t QueryGpuFreeBytes(int gpu_id);
void CudaCheckLastKernel(const char *name);

void CufftPlanAlloc(cufftHandle *, int, int *, int *, int, int, int *, int, int,
                    cufftType, int);
void GpuMalloc(void **, size_t);
void GpuFree(void **);

#endif
