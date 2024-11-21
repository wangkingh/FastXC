#include <cstddef>
#include <cuComplex.h>
#include <cufft.h>
#include <math_constants.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.stransform.cuh"

__global__ void copyRealToComplex(float *realArray, cufftComplex *complexArray, size_t width, size_t height)
{
    // width : nfft,           序列(频谱)点数, 用[col]索引
    // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引, 在这个函数中没有涉及
    // height: num_trace,      数据道数,       用[row]索引
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // 对应 nfft
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // 对应 num_trace
    if (col < width && row < height)
    {
        size_t idx = row * width + col;
        complexArray[idx].x = realArray[idx];
        complexArray[idx].y = 0.0f;
    }
}

__global__ void compute_g_matrix_kernel(float *g_matrix, size_t nfft, size_t nfreq, float scale)
{
    // width : nfft,           序列(频谱)点数, 用[col]索引
    // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引
    // height: num_trace,      数据道数,       用[row]索引,在这里是 1（未在核函数参数中直接使用）

    // col 索引应该对应 nfft 的点数，即宽度 width
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // nfft index, 对应 width
    // depth 索引应该对应 nfreq 的点数，即深度 depth
    size_t depth = blockIdx.y * blockDim.y + threadIdx.y; // nfreq index, 对应 depth

    float pi2 = -2.0f * CUDART_PI_F * CUDART_PI_F;
    size_t half_nfft = (nfft + 1) / 2; // 正确索引中间点, Use floor division for correct indexing
    if (col < half_nfft && depth < nfreq)
    {
        size_t idx = depth * nfft + col; // 第depth(nfreq_idx)行, 第col(nfft_idx)列
        size_t mirror_idx = depth * nfft + (nfft - col);

        if (depth == 0) // 第一个调制频率(调制直流分量)，整个高斯窗设置为1
        {
            g_matrix[idx] = 1.0f;
            g_matrix[mirror_idx] = 1.0f;
            return;
        }

        if (col == 0) // 所有调制频率的高斯窗的第一个值(直流分量)置零
        {
            g_matrix[idx] = 0.0f;
        }
        else
        {
            float value = expf(pi2 * col * col * scale / (depth * depth));
            g_matrix[idx] = value;
            if (col <= half_nfft) // Ensure not to overwrite in the Nyquist frequency scenario for even nfft
            {
                g_matrix[mirror_idx] = value;
            }
        }
    }
}

__global__ void hilbertTransformKernel(cufftComplex *d_spectrum, size_t nfft, size_t num_trace)
{
    // width : nfft,           序列(频谱)点数, 用[col]索引
    // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引,在这里是 1（未在核函数参数中直接使用）
    // height: num_trace,      数据道数,       用[row]索引

    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // nfft index, 对应 width
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // trace index, 对应 height

    if (col < nfft && row < num_trace)
    {
        size_t idx = row * nfft + col; // 第row(trace_idx)行, 第col(nfft_idx)列
        int half_nfft = (nfft + 1) / 2;

        if (col >= half_nfft || col == 0)
        {
            // 将高于一半频率的部分置零,直流分量置零
            d_spectrum[idx].x = 0.0f;
            d_spectrum[idx].y = 0.0f;
        }
        else
        {
            d_spectrum[idx].x *= 2.0f;
            d_spectrum[idx].y *= 2.0f;
        }
    }
}

__global__ void gaussianModulate(cufftComplex *d_spectrum, cufftComplex *modulatedSpectrum, float *g_matrix,
                                 size_t num_trace, size_t nfreq, size_t nfft)
{
    // width : nfft,           序列(频谱)点数, 用[col]索引
    // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引
    // height: num_trace,      数据道数,       用[row]索引
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;   // 对应 nfft
    size_t depth = blockIdx.y * blockDim.y + threadIdx.y; // 对应 nfreq
    size_t row = blockIdx.z * blockDim.z + threadIdx.z;   // 对应 num_trace
    if (row < num_trace && depth < nfreq && col < nfft)
    {
        size_t modulated_spec_idx = depth + col;
        if (modulated_spec_idx >= nfft)
            modulated_spec_idx -= nfft;                         // 环形索引处理
        size_t originalIndex = row * nfft + modulated_spec_idx; // 定位原始频谱的位置
        size_t gaussianValueIndex = depth * nfft + col;         // 定位调制幅度的位置

        cufftComplex originalValue = d_spectrum[originalIndex]; // 原始频谱值
        float modulationFactor = g_matrix[gaussianValueIndex];  // 高斯调制因子

        int modulatedIndex = row * nfreq * nfft + depth * nfft + col; // 定位调制后频谱的位置

        // 应用高斯调制
        modulatedSpectrum[modulatedIndex].x = originalValue.x * modulationFactor;
        modulatedSpectrum[modulatedIndex].y = originalValue.y * modulationFactor;
    }
}

__global__ void normalizeModulatedAnalysis(cufftComplex *modulatedAnalysis, size_t nfft, size_t nfreq, size_t num_trace)
{
    // width : nfft,           序列(频谱)点数, 用[col]索引
    // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引
    // height: num_trace,      数据道数,       用[row]索引
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;   // 对应 nfft
    size_t depth = blockIdx.y * blockDim.y + threadIdx.y; // 对应 nfreq
    size_t row = blockIdx.z * blockDim.z + threadIdx.z;   // 对应 num_trace

    if (row < num_trace && depth < nfreq && col < nfft)
    {
        size_t idx = row * nfreq * nfft + depth * nfft + col;
        modulatedAnalysis[idx].x /= nfft;
        modulatedAnalysis[idx].y /= nfft;
    }
}

__global__ void calculateWeight(cufftComplex *trans_all, cuComplex *weight_complex, size_t nfft, size_t nfreq, size_t num_trace)
{
    // width : nfft,           序列(频谱)点数, 用[col]索引
    // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引
    // height: num_trace,      数据道数,       用[row]索引
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;   // 对应 nfft
    size_t depth = blockIdx.y * blockDim.y + threadIdx.y; // 对应 nfreq
                                                          //    float normalization_factor = num_trace * nfft;        // 定义归一化因子
    float normalization_factor = nfft * num_trace;        // 定义归一化因子

    if (col < nfft && depth < nfreq)
    {
        size_t out_idx = depth * nfft + col; // 第depthf(nfreq_idx个调制频率), 第col列(nfft_idx个频率)
        for (size_t row = 0; row < num_trace;row++)
        {
            size_t idx = row * nfreq * nfft + depth * nfft + col;
            float fa = sqrtf(trans_all[idx].x * trans_all[idx].x + trans_all[idx].y * trans_all[idx].y);
            fa = (fa > 10e-7f) ? fa : 1.0f;
            // 三重归一化: 叠加求均值fileCount、反傅里叶变换nfft、复数的模fa
            weight_complex[out_idx].x += (trans_all[idx].x) / (fa * normalization_factor);
            weight_complex[out_idx].y += (trans_all[idx].y) / (fa * normalization_factor);
        }
    }
}

__global__ void linearSumTraces(float *d_sacdata, float *d_output, size_t num_trace, size_t nfft)
{
    // width : nfft,           序列(频谱)点数, 用[col]索引
    // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引, 这里没有这个概念
    // height: num_trace,      数据道数,       用[row]索引
    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // 全局索引

    if (col < nfft)
    {
        float sum = 0.0f;
        for (size_t row = 0; row < num_trace; row++)
        {
            sum += d_sacdata[row * nfft + col];
        }
        d_output[col] = sum;
    }
}

__global__ void applyWeight(cufftComplex *analysisSignal, cuComplex *weight, size_t nfreq, size_t nfft, float weight_order)
{
    // width : nfft,           序列(频谱)点数, 用[col]索引
    // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引
    // height: num_trace,      数据道数,不适用 用[row]索引,这里总共只有一道数据
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;   // 对应序列点数，使用nfft宽度索引
    size_t depth = blockIdx.y * blockDim.y + threadIdx.y; // 对应调制频率点数，使用nfreq深度索引

    if (depth < nfreq && col < nfft)
    {
        size_t idx = depth * nfft + col; // 计算一维数组中的索引位置

        float new_weight = powf((powf(weight[idx].x, 2.0f) + powf(weight[idx].y, 2.0f)), weight_order / 2.0f); // 不使用权重
        analysisSignal[idx].x *= new_weight;                                                                   // 更新实部
        analysisSignal[idx].y *= new_weight;                                                                   // 更新虚部
    }
}

__global__ void sumComplexSpectrumSimple(cufftComplex *analysisSignal, cufftComplex *d_pws_spectrum, size_t nfreq, size_t nfft)
{
    // width : nfft,           序列(频谱)点数, 用[col]索引
    // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引
    // height: num_trace,      数据道数,不适用 用[row]索引,这里总共只有一道数据
    size_t depth = blockIdx.x * blockDim.x + threadIdx.x; // 使用线程索引来表示频率点

    if (depth < nfreq)
    {
        size_t idx = depth;
        float sumReal = 0.0f;
        float sumImag = 0.0f;

        // 累加当前频率点的所有nfft点的实部和虚部
        for (size_t col = 0; col < nfft; col++)
        {
            size_t value_idx = depth * nfft + col;
            sumReal += analysisSignal[value_idx].x;
            sumImag += analysisSignal[value_idx].y;
        }

        // 写入结果数组
        d_pws_spectrum[idx].x = sumReal;
        d_pws_spectrum[idx].y = sumImag;
    }
}

__global__ void extractReal(float *d_tfpw_stack, cufftComplex *d_tfpw_stack_complex, size_t nfft)
{
    // width : nfft,           序列(频谱)点数, 用[col]索引, 这里是解析时间信号的索引
    // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引, 这里没有调制
    // height: num_trace,      数据道数,不适用 用[row]索引,这里总共只有一道数据
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < nfft)
    {
        d_tfpw_stack[col] = d_tfpw_stack_complex[col].x / nfft;
    }
}
