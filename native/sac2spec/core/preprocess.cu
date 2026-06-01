#include "preprocess.cuh"

#include "../kernels/misc.cuh"
#include "../kernels/rdcrtr.cuh"
#include "../kernels/taper.cuh"
#include "cuda.util.cuh"

#include <limits.h>

static int kernelOk(const char *op)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        LOG_ERROR("core_cuda_kernel_failed",
                  "op=%s code=%d name=%s message=\"%s\"",
                  op, (int)err, cudaGetErrorName(err), cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int preprocess(float *d_sacdata, double *d_sum, double *d_isum,
                               int pitch, size_t row_count,
                               float freq_low, float delta)
{
    if (d_sacdata == NULL || d_sum == NULL || d_isum == NULL ||
        pitch <= 0 || row_count == 0 || row_count > (size_t)INT_MAX ||
        freq_low <= 0.0f || delta <= 0.0f)
    {
        LOG_ERROR("core_preprocess_invalid_input",
                  "data=%p sum=%p isum=%p pitch=%d rows=%zu freq_low=%.8g delta=%.8g",
                  d_sacdata, d_sum, d_isum, pitch, row_count, freq_low, delta);
        return -1;
    }

    size_t width = (size_t)pitch;
    size_t height = row_count;
    int height_i = (int)height;
    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);

    dim3 dimgrd2, dimblk2;
    dimblk2.x = BLOCKMAX;
    dimblk2.y = 1;
    dimgrd2.x = 1;
    dimgrd2.y = height;

    isnan2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, pitch, height_i);
    if (kernelOk("preprocess_isnan") != 0)
    {
        return -1;
    }

    sumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                             dimblk2.x * dimblk2.y * sizeof(double)>>>
        (d_sum, 1, d_sacdata, pitch, pitch, height_i);
    if (kernelOk("preprocess_sum_for_rdc") != 0)
    {
        return -1;
    }

    rdc2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, pitch, height_i, d_sum);
    if (kernelOk("preprocess_rdc") != 0)
    {
        return -1;
    }

    sumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                             dimblk2.x * dimblk2.y * sizeof(double)>>>
        (d_sum, 1, d_sacdata, pitch, pitch, height_i);
    if (kernelOk("preprocess_sum_for_rtr") != 0)
    {
        return -1;
    }

    isumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                              dimblk2.x * dimblk2.y * sizeof(double)>>>
        (d_isum, 1, d_sacdata, pitch, pitch, height_i);
    if (kernelOk("preprocess_isum_for_rtr") != 0)
    {
        return -1;
    }

    rtr2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, pitch, height_i, d_sum, d_isum);
    if (kernelOk("preprocess_rtr") != 0)
    {
        return -1;
    }

    size_t taper_size = (size_t)(2.0f * (1.0f / freq_low) / delta);
    if (taper_size > (size_t)INT_MAX)
    {
        LOG_ERROR("core_preprocess_taper_too_large",
                  "taper_size=%zu max=%d", taper_size, INT_MAX);
        return -1;
    }
    timetaper2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, pitch, height_i, (int)taper_size);
    if (kernelOk("preprocess_timetaper") != 0)
    {
        return -1;
    }

    return 0;
}
