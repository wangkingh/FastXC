#include "cuda.pws_util.cuh"
#include <stdio.h>

__global__ void cudaMean(cufftComplex *hilbert_complex, cufftComplex *mean, size_t num_trace, size_t nfft)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < nfft)
    {
        float sum_real = 0.0f, sum_img = 0.0f;
        // 计算当前时间点上所有信号的实部总和
        for (size_t j = 0; j < num_trace; ++j)
        {
            sum_real += hilbert_complex[j * nfft + col].x;
            sum_img += hilbert_complex[j * nfft + col].y;
        }
        // // 计算平均值
        mean[col].x = sum_real / num_trace;
        mean[col].y = sum_img / num_trace;
    }
}

__global__ void cudaNormalizeComplex(cufftComplex *hilbert_complex, size_t data_num, size_t nfft)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // 一维索引
    if (idx < data_num)
    {
        float real = hilbert_complex[idx].x / nfft;
        float imag = hilbert_complex[idx].y / nfft;

        float modulus = sqrtf(real * real + imag * imag);
        modulus = (modulus > 1e-7f) ? modulus : 1e-7f;
        hilbert_complex[idx].x = real / modulus;
        hilbert_complex[idx].y = imag / modulus;
    }
}

__global__ void cudaMultiply(float *linear_stack, cuComplex *weight, float *pws_stack, size_t nfft)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nfft)
    {
        float weight_value = sqrtf(weight[idx].x * weight[idx].x + weight[idx].y * weight[idx].y);
        pws_stack[idx] = linear_stack[idx] * weight_value;
    }
}