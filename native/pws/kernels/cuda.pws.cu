#include "cuda.pws.cuh"

__global__ void hilbertTransformKernel(cufftComplex *d_inputSpectrum,
                                       size_t freqDomainSize,
                                       size_t nTraces)
{
    size_t freqIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t traceIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (freqIdx < freqDomainSize && traceIdx < nTraces)
    {
        size_t idx = traceIdx * freqDomainSize + freqIdx;
        size_t halfNfft = (freqDomainSize + 1) / 2;

        if (freqIdx == 0 || freqIdx >= halfNfft)
        {
            d_inputSpectrum[idx].x = 0.0f;
            d_inputSpectrum[idx].y = 0.0f;
        }
        else
        {
            d_inputSpectrum[idx].x *= 2.0f;
            d_inputSpectrum[idx].y *= 2.0f;
        }
    }
}

__global__ void cudaWeightedMeanBatch(cufftComplex *hilbert_complex,
                                      const float *trace_weights,
                                      const size_t *pair_group_offsets,
                                      const size_t *pair_group_counts,
                                      cufftComplex *mean,
                                      size_t num_pairs,
                                      size_t nfft)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pair = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < nfft && pair < num_pairs)
    {
        size_t group_offset = pair_group_offsets[pair];
        size_t group_count = pair_group_counts[pair];
        float sum_real = 0.0f;
        float sum_img = 0.0f;

        for (size_t j = 0; j < group_count; ++j)
        {
            size_t group = group_offset + j;
            float weight = trace_weights[group];
            size_t idx = group * nfft + col;
            sum_real += hilbert_complex[idx].x * weight;
            sum_img += hilbert_complex[idx].y * weight;
        }

        size_t out_idx = pair * nfft + col;
        mean[out_idx].x = sum_real;
        mean[out_idx].y = sum_img;
    }
}

__global__ void cudaNormalizeComplex(cufftComplex *hilbert_complex,
                                     size_t data_num,
                                     size_t nfft)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void cudaMultiplyBatch(float *linear_stack,
                                  cuComplex *weight,
                                  float *pws_stack,
                                  size_t num_pairs,
                                  size_t nfft)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pair = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nfft && pair < num_pairs)
    {
        size_t out_idx = pair * nfft + idx;
        float weight_value = sqrtf(weight[out_idx].x * weight[out_idx].x +
                                   weight[out_idx].y * weight[out_idx].y);
        pws_stack[out_idx] = linear_stack[out_idx] * weight_value;
    }
}
