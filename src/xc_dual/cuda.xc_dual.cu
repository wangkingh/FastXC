#include "cuda.xc_dual.cuh"
#include <stdio.h>

__global__ void complexMul2DKernel(cuComplex *src_vec, cuComplex *sta_vec, size_t spitch,
                                   cuComplex *ncf_vec, size_t dpitch,
                                   size_t width, size_t height)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height)
    {
        size_t sidx = row * spitch + col;
        size_t didx = row * dpitch + col; // desitnation 包含一部分0,dpitch ~ 2 *spitch

        cuComplex src_conj_value = make_cuComplex(src_vec[sidx].x, -src_vec[sidx].y);
        cuComplex sta_value = sta_vec[sidx];

        ncf_vec[didx] = cuCmulf(src_conj_value, sta_value);
    }
}

__global__ void generateSignVector(int *sgn_vec, size_t width)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width)
    {
        if (col == 0)
        {
            sgn_vec[col] = 0;
        }
        sgn_vec[col] = (col % 2 == 0) ? 1 : -1;
    }
}

__global__ void applyPhaseShiftKernel(cuComplex *ncf_vec, int *sgn_vec, size_t spitch, size_t width, size_t height)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height)
    {
        size_t idx = row * spitch + col;
        int sign = sgn_vec[col]; // 使用一维向量
        ncf_vec[idx].x *= sign;
        ncf_vec[idx].y *= sign;
    }
}

// sum2dKernel is used to sum the 2D array of float, not used in the current version
__global__ void sum2DKernel(float *d_finalccvec, int dpitch, float *d_segncfvec,
                            int spitch, size_t width, size_t height,
                            int nstep)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int sidx = row * spitch + col;
        int didx = row * dpitch + col;
        d_finalccvec[didx] += (d_segncfvec[sidx] / nstep);
    }
}

__global__ void csum2DKernel(cuComplex *d_total_spectrum, int dpitch,
                             cuComplex *d_segment_spectrum, int spitch,
                             size_t width, size_t height, int step_idx, int nstep)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int sidx = row * spitch * nstep + step_idx * spitch + col;
        int didx = row * dpitch + col;
        cuComplex temp = d_segment_spectrum[sidx];
        temp.x /= nstep; // divide the real part by nstep
        temp.y /= nstep; // divide the imaginary part by nstep

        d_total_spectrum[didx] = cuCaddf(d_total_spectrum[didx], temp);
    }
}

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    double weight = 1.0 / (width * dt);
    if (row < height && col < width)
    {
        size_t idx = row * pitch + col;
        d_segdata[idx] *= weight;
    }
}