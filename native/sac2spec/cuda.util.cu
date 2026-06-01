#include "cuda.util.cuh"

void DimCompute(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
{
    pdimblk->x = BLOCKX;
    pdimblk->y = BLOCKY;

    pdimgrd->x = (width + BLOCKX - 1) / BLOCKX;
    pdimgrd->y = (height + BLOCKY - 1) / BLOCKY;
}

void GpuFree(void **pptr)
{
    if (*pptr != NULL)
    {
        cudaError_t err = cudaFree(*pptr);
        if (err != cudaSuccess)
        {
            LOG_WARN("cuda_free_failed", "code=%d name=%s message=\"%s\"",
                     (int)err, cudaGetErrorName(err), cudaGetErrorString(err));
        }
        *pptr = NULL;
    }
}
