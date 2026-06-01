#ifndef SAC2SPEC_DEVICE_MEMORY_INTERNAL_CUH
#define SAC2SPEC_DEVICE_MEMORY_INTERNAL_CUH

#include "device_memory.cuh"

static inline double DeviceMemoryBytesToMiB(size_t bytes)
{
    return bytes / (1024.0 * 1024.0);
}

static inline size_t DeviceMemoryMaxSize(size_t a, size_t b)
{
    return (a > b) ? a : b;
}

typedef struct CufftWorkspaceBreakdown
{
    size_t fwd_1x;
    size_t inv_1x;
    size_t fwd_filter;
    size_t inv_filter;
    size_t fwd_output;
    size_t shared_max;
    size_t sum_if_auto;
} CufftWorkspaceBreakdown;

static cufftResult_t CreateCufftPlanNoAuto(cufftHandle *plan, int nfft,
                                           cufftType type, int batch,
                                           size_t *work_size)
{
    int rank = 1;
    int n[1] = {nfft};
    int inembed[1] = {nfft};
    int onembed[1] = {nfft};
    int istride = 1;
    int idist = nfft;
    int ostride = 1;
    int odist = nfft;

    *plan = 0;
    *work_size = 0;

    cufftResult_t err = cufftCreate(plan);
    if (err != CUFFT_SUCCESS)
    {
        return err;
    }

    err = cufftSetAutoAllocation(*plan, 0);
    if (err != CUFFT_SUCCESS)
    {
        cufftDestroy(*plan);
        *plan = 0;
        return err;
    }

    err = cufftMakePlanMany(*plan, rank, n, inembed, istride, idist,
                            onembed, ostride, odist, type, batch,
                            work_size);
    if (err != CUFFT_SUCCESS)
    {
        cufftDestroy(*plan);
        *plan = 0;
    }
    return err;
}

#endif
