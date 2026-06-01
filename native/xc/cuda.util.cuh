#ifndef _CUDA_UTIL_CUH
#define _CUDA_UTIL_CUH

#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include "logger.h"

#define BLOCKX 32
#define BLOCKY 32
#define BLOCKX1D 256

#define CUDACHECK(cmd)                                              \
  do                                                                \
  {                                                                 \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess)                                           \
    {                                                               \
      LOG_ERROR("cuda_error", "cuda_status=\"%s\"",                 \
                cudaGetErrorString(e));                              \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#define CUFFTCHECK(cmd)                                                \
  do                                                                   \
  {                                                                    \
    cufftResult_t e = cmd;                                             \
    if (e != CUFFT_SUCCESS)                                            \
    {                                                                  \
      LOG_ERROR("cufft_error", "cufft_status=%d", (int)e);          \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

void DimCompute1D(dim3 *pdimgrd, dim3 *pdimblk, size_t width);
void DimCompute(dim3 *, dim3 *, size_t, size_t);

void GpuFree(void **);

#endif
