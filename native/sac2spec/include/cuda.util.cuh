#ifndef _CUDA_UTIL_CUH
#define _CUDA_UTIL_CUH
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <stdio.h>
#include "config.h"
#include "logger.h"
#define BLOCKX 32
#define BLOCKY 32
#define BLOCKMAX 1024

static inline const char *cufftResultName(cufftResult_t result)
{
    switch (result)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
        return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:
        return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:
        return "CUFFT_NOT_SUPPORTED";
    default:
        return "CUFFT_UNKNOWN_ERROR";
    }
}

#define CUDACHECK(cmd)                                                             \
    do                                                                             \
    {                                                                              \
        cudaError_t e = (cmd);                                                     \
        if (e != cudaSuccess)                                                      \
        {                                                                          \
            LOG_ERROR("cuda_api_error",                                            \
                      "expr=\"%s\" code=%d name=%s message=\"%s\"",               \
                      #cmd, (int)e, cudaGetErrorName(e), cudaGetErrorString(e));   \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                         \
    do                                                                             \
    {                                                                              \
        cudaError_t e = cudaGetLastError();                                         \
        if (e != cudaSuccess)                                                      \
        {                                                                          \
            LOG_ERROR("cuda_kernel_launch_error",                                  \
                      "code=%d name=%s message=\"%s\"",                            \
                      (int)e, cudaGetErrorName(e), cudaGetErrorString(e));         \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

#define CUFFTCHECK(cmd)                                                            \
    do                                                                             \
    {                                                                              \
        cufftResult_t e = (cmd);                                                    \
        if (e != CUFFT_SUCCESS)                                                    \
        {                                                                          \
            LOG_ERROR("cufft_error", "expr=\"%s\" code=%d name=%s",              \
                      #cmd, (int)e, cufftResultName(e));                           \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

/* Compute grid and block dimensions for a 2D CUDA kernel. */
void DimCompute(dim3 *, dim3 *, size_t, size_t);

/* Free a CUDA pointer and reset it to NULL. */
void GpuFree(void **pptr);

#endif
