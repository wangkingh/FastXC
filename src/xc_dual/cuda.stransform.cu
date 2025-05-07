#include <cstddef>
#include <cuComplex.h>
#include <cufft.h>
#include <math_constants.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.stransform.cuh"

__global__ void hilbertTransformKernel(cufftComplex *d_inputspectrum, size_t freqDomainSize, size_t nTraces)
{
    // 这里采用 2D grid: (freqShiftIdx, traceIdx)
    //   blockIdx.x*blockDim.x + threadIdx.x => freqShiftIdx in [0..freqDomainSize)
    //   blockIdx.y*blockDim.y + threadIdx.y => traceIdx in [0..nTraces)

    size_t freqShiftIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t traceIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (freqShiftIdx < freqDomainSize && traceIdx < nTraces)
    {
        // 每个道(traceIdx)对应 freqDomainSize 个频点(freqShiftIdx)
        size_t idx = traceIdx * freqDomainSize + freqShiftIdx;
        int half_nfft = (freqDomainSize + 1) / 2;

        // 当 freqShiftIdx == 0 (DC) 或 freqShiftIdx >= half_nfft (负频) => 置零
        // 否则(正频区) => 乘以2
        if (freqShiftIdx >= (size_t)half_nfft || freqShiftIdx == 0)
        {
            d_inputspectrum[idx].x = 0.0f;
            d_inputspectrum[idx].y = 0.0f;
        }
        else
        {
            d_inputspectrum[idx].x *= 2.0f;
            d_inputspectrum[idx].y *= 2.0f;
        }
    }
}

/**
 * @brief 在频-频域上，对 [chunkStartFreq, chunkStartFreq + chunkFreqCount) 这段原始频率
 *        做高斯窗调制（环形移频），输出到 d_modulatedSubChunk。
 *
 * @param d_inputSpectrum       [in ] 大小 = nTraces * freqDomainLen
 *                              (每道在 freqDomainLen 上的解析频谱)
 * @param d_modulatedSubChunk   [out] 大小 = nTraces * chunkFreqCount * freqDomainLen
 *                              (存储分块频率 x 移频后的新频谱)
 * @param nTraces              道数
 * @param freqDomainLen        频域的长度 (通常 = nfft，用于环形移频)
 * @param chunkStartFreq       本分块起始频率下标
 * @param chunkFreqCount       本分块的频率数量
 * @param scale                高斯窗宽度因子
 *
 * 线程索引:
 *   - x => freqShiftIdx in [0..freqDomainLen)
 *   - y => freqInIdx     in [0..chunkFreqCount)
 *   - z => traceIdx      in [0..nTraces)
 */
__global__ void gaussianModulateSub(
    const cufftComplex *__restrict__ d_inputSpectrum,
    cufftComplex *__restrict__ d_modulatedSubChunk,
    size_t nTraces,
    size_t freqDomainLen,
    int chunkStartFreq,
    int chunkFreqCount,
    float scale)
{
    // freqShiftIdx: [0.. freqDomainLen)
    size_t freqShiftIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // freqInIdx: [0.. chunkFreqCount)
    size_t freqInIdx = blockIdx.y * blockDim.y + threadIdx.y;
    // traceIdx: [0.. nTraces)
    size_t traceIdx = blockIdx.z * blockDim.z + threadIdx.z;

    if (traceIdx < nTraces && freqInIdx < (size_t)chunkFreqCount && freqShiftIdx < freqDomainLen)
    {
        // 1) 计算分块内的“全局原始频率”
        int globalFreqIn = chunkStartFreq + freqInIdx; // 0 <= globalFreqIn < totalFreqBins

        // 2) 计算环形移频 => freqOutIdx = globalFreqIn + freqShiftIdx (mod freqDomainLen)
        size_t freqOutIdx = globalFreqIn + freqShiftIdx;
        if (freqOutIdx >= freqDomainLen)
        {
            freqOutIdx -= freqDomainLen;
        }

        // 3) 读取输入数据: d_inputSpectrum[traceIdx, freqOutIdx]
        //    => index = traceIdx * freqDomainLen + freqOutIdx
        size_t inIndex = traceIdx * freqDomainLen + freqOutIdx;
        cufftComplex valIn = d_inputSpectrum[inIndex];

        // 4) 计算高斯调制因子
        //    - freqShiftIdx 其实是“移频坐标”
        //    - globalFreqIn 是“原始频率”
        float modulationFactor;
        int halfLen = (freqDomainLen + 1) / 2;
        float pi2 = -2.0f * CUDART_PI_F * CUDART_PI_F;

        if (globalFreqIn == 0)
        {
            // 中心频率=0 => 全部设为1
            modulationFactor = 1.0f;
        }
        else if (freqShiftIdx == 0)
        {
            // 移频索引=0 => 可能 amplitude=0
            modulationFactor = 0.0f;
        }
        else
        {
            int c = (int)freqShiftIdx;
            int f = (int)globalFreqIn;

            if (c < halfLen)
            {
                // “正频端”
                float val = expf(pi2 * c * c * scale / (f * (float)f));
                modulationFactor = val;
            }
            else
            {
                // freqShiftIdx >= halfLen => 做镜像 c' = freqDomainLen - c
                int mirrorCol = (int)freqDomainLen - c;
                if (mirrorCol < halfLen && mirrorCol != 0 && f != 0)
                {
                    float val = expf(pi2 * mirrorCol * mirrorCol * scale / (f * (float)f));
                    modulationFactor = val;
                }
                else if (f == 0)
                {
                    modulationFactor = 1.0f;
                }
                else if (mirrorCol == 0)
                {
                    modulationFactor = 0.0f;
                }
                else
                {
                    modulationFactor = 0.0f;
                }
            }
        }

        // 5) 写入输出 d_modulatedSubChunk
        //    大小: [nTraces, chunkFreqCount, freqDomainLen]
        //    => outIndex = traceIdx*(chunkFreqCount*freqDomainLen)
        //                 + freqInIdx*(freqDomainLen) + freqShiftIdx
        size_t outIndex = traceIdx * (size_t)(chunkFreqCount * freqDomainLen) + freqInIdx * freqDomainLen + freqShiftIdx;

        d_modulatedSubChunk[outIndex].x = valIn.x * modulationFactor;
        d_modulatedSubChunk[outIndex].y = valIn.y * modulationFactor;
    }
}

__global__ void calculateWeightSub(
    const cufftComplex *__restrict__ d_subTransformAll, // [nTraces, freqChunkSize, freqDomainSize] after IFFT
    cuComplex *__restrict__ d_weightMatrix,             // 全局权重, 大小= nfreq * freqDomainSize
    size_t freqDomainSize,                              // 总的频率点数
    size_t nTraces,
    int freqChunkStartIdx, // 本次处理频段起始
    int freqChunkSize      // 本次处理频段的大小
)
{
    // 局部: col=[0..freqDomainSize), depth=[0..freqChunkSize)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < freqDomainSize && depth < freqChunkSize)
    {
        // 1) 全局频率下标
        int global_f = freqChunkStartIdx + depth; // 0 <= global_f < nfreq
        // 2) 计算输出索引
        size_t out_idx = (size_t)global_f * freqDomainSize + col;

        float normalization_factor = (float)(freqDomainSize * nTraces);

        // 3) 遍历道数累加
        cuComplex accum;
        accum.x = 0.0f;
        accum.y = 0.0f;
        for (size_t row = 0; row < nTraces; row++)
        {
            // 输入形状: [nTraces, freqChunkSize, freqDomainSize]
            // 索引: row*(freqChunkSize*freqDomainSize) + depth*freqDomainSize + col
            size_t idx = row * ((size_t)freqChunkSize * freqDomainSize) + (size_t)depth * freqDomainSize + col;

            float x = d_subTransformAll[idx].x;
            float y = d_subTransformAll[idx].y;
            float fa = sqrtf(x * x + y * y);
            if (fa <= 1e-7f)
                fa = 1.0f;

            accum.x += x / (fa * normalization_factor);
            accum.y += y / (fa * normalization_factor);
        }

        d_weightMatrix[out_idx].x = accum.x;
        d_weightMatrix[out_idx].y = accum.y;
    }
}

__global__ void applyWeight(
    cufftComplex *d_tfAnalysis, // [nfreq, freqDomainSize]
    cuComplex *d_weightMatrix,  // [nfreq, freqDomainSize]
    size_t nfreq,
    size_t freqDomainSize,
    float weight_order)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 对应 freqDomainSize
    int depth = blockIdx.y * blockDim.y + threadIdx.y; // 对应 nfreq

    if (col < freqDomainSize && depth < nfreq)
    {
        size_t idx = depth * freqDomainSize + col;
        float rw = d_weightMatrix[idx].x;
        float iw = d_weightMatrix[idx].y;
        float mag = powf(rw * rw + iw * iw, 0.5f * weight_order);

        d_tfAnalysis[idx].x *= mag;
        d_tfAnalysis[idx].y *= mag;
    }
}

// /**
//  * @brief 在二维复数数组 [nfreq, freqDomainSize] 上，对“nfreq”这一维进行累加，输出一维 [freqDomainSize]。
//  *
//  * 本函数最常用于将多个“调制频率/中心频率”或“局部频段”所对应的分析结果叠加到一起。
//  *
//  * - 第一维 nfreq:
//  *   通常表示**不同的调频通道**或**不同中心频率**（例如 TF-PWS 中为多种调制频率，或 Stockwell 变换里分段频率）。
//  *   我们对它进行求和/折叠，以得到合并后的谱。
//  *
//  * - 第二维 freqDomainSize:
//  *   常常是 FFT 或频域长度，也可能是环形移频后的索引范围。
//  *   在结果中保持这一维以得到最终的一维复数序列。
//  *
//  * 具体过程：
//  *   - 输入 d_tfAnalysis 大小=[nfreq, freqDomainSize]
//  *   - 输出 d_outSpectrum 大小=[freqDomainSize]
//  *   - 对每个 domainIdx ∈ [0..freqDomainSize)：
//  *       遍历 freqIndex ∈ [0..nfreq)，将对应复数数据依次累加到 sumR/sumI
//  *       最终写入 d_outSpectrum[domainIdx] 中。
//  *
//  * 使用场景：
//  *   - 当 d_tfAnalysis 同时包含多个调制频率（nfreq）与另一维度（freqDomainSize）上的数据，
//  *     而你只想把所有调制频率叠加，得到一条汇总谱或单道结果时，就可使用本函数。
//  *
//  * @param[in]  d_tfAnalysis     输入 2D 复数数组，形状=[nfreq, freqDomainSize]
//  * @param[out] d_outSpectrum    输出 1D 复数数组，形状=[freqDomainSize] (叠加结果)
//  * @param[in]  nfreq            第一维大小 (表示不同调制频率/中心频率的数目)
//  * @param[in]  freqDomainSize   第二维大小 (通常为域长度或 FFT 大小)
//  */
// __global__ void sumOverFreqAxisKernel(
//     const cufftComplex *d_tfAnalysis, // [nfreq, freqDomainSize]
//     cufftComplex *d_outSpectrum,      // [freqDomainSize]
//     size_t nfreq,
//     size_t freqDomainSize)
// {
//     // 将 thread 索引映射到 domainIdx
//     int domainIdx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (domainIdx < (int)freqDomainSize)
//     {
//         float sumR = 0.0f;
//         float sumI = 0.0f;

//         // 在第一维 freqIndex in [0..nfreq) 上做求和
//         for (int freqIndex = 0; freqIndex < (int)nfreq; freqIndex++)
//         {
//             size_t inIdx = (size_t)freqIndex * freqDomainSize + domainIdx;
//             sumR += d_tfAnalysis[inIdx].x;
//             sumI += d_tfAnalysis[inIdx].y;
//         }

//         // 将结果写入输出的一维数组
//         d_outSpectrum[domainIdx].x = sumR;
//         d_outSpectrum[domainIdx].y = sumI;
//     }
// }

// __global__ void sumOverFreqAxisKernel(const cufftComplex *analysisSignal, cufftComplex *d_pws_spectrum, size_t nfreq, size_t nfft)
// {
//     // width : nfft,           序列(频谱)点数, 用[col]索引
//     // depth : nfreq=nfft/2+1, 调制频率点数,   用[depth]索引
//     // height: num_trace,      数据道数,不适用 用[row]索引,这里总共只有一道数据
//     size_t depth = blockIdx.x * blockDim.x + threadIdx.x; // 使用线程索引来表示频率点

//     if (depth < nfreq)
//     {
//         size_t idx = depth;
//         float sumReal = 0.0f;
//         float sumImag = 0.0f;

//         // 累加当前频率点的所有nfft点的实部和虚部
//         for (size_t col = 0; col < nfft; col++)
//         {
//             size_t value_idx = depth * nfft + col;
//             sumReal += analysisSignal[value_idx].x;
//             sumImag += analysisSignal[value_idx].y;
//         }

//         // 写入结果数组
//         d_pws_spectrum[idx].x = sumReal;
//         d_pws_spectrum[idx].y = sumImag;
//     }
// }

/**
 * @brief
 *  在二维复数数组 [nfreq, ntime] 上，对 freq 维做累加，保留 time 维。
 *  形状说明:
 *    - d_input : [nfreq, ntime]
 *    - d_output: [ntime]
 *
 * @param d_input   输入 2D 复数数组
 * @param d_output  输出 1D 复数数组
 * @param nfreq     第一维大小 (频率数)
 * @param ntime     第二维大小 (时间采样数)
 */
__global__ void sumOverFreqAxisKernel(
    const cufftComplex *d_input,
    cufftComplex *d_output,
    size_t nfreq,
    size_t ntime)
{
    // timeIdx: 线程索引 (遍历第二维)
    size_t timeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (timeIdx < ntime)
    {
        float sumReal = 0.0f;
        float sumImag = 0.0f;

        // 在 freq 维(nfreq)上遍历
        for (size_t freqIdx = 0; freqIdx < nfreq; freqIdx++)
        {
            size_t inIdx = freqIdx * ntime + timeIdx;
            sumReal += d_input[inIdx].x;
            sumImag += d_input[inIdx].y;
        }

        // 写入输出 [ntime]
        d_output[timeIdx].x = sumReal;
        d_output[timeIdx].y = sumImag;
    }
}

/**
 * @brief
 *  在二维复数数组 [nfreq, ntime] 上，对 time 维做累加，保留 freq 维。
 *  形状说明:
 *    - d_input : [nfreq, ntime]
 *    - d_output: [nfreq]
 *
 * @param d_input   输入 2D 复数数组
 * @param d_output  输出 1D 复数数组
 * @param nfreq     第一维大小 (频率数)
 * @param ntime     第二维大小 (时间采样数)
 */
__global__ void sumOverTimeAxisKernel(
    const cufftComplex *d_input,
    cufftComplex *d_output,
    size_t nfreq,
    size_t ntime)
{
    // freqIdx: 线程索引 (遍历第一维)
    size_t freqIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (freqIdx < nfreq)
    {
        float sumReal = 0.0f;
        float sumImag = 0.0f;

        // 在 time 维(ntime)上遍历
        for (size_t timeIdx = 0; timeIdx < ntime; timeIdx++)
        {
            size_t inIdx = freqIdx * ntime + timeIdx;
            sumReal += d_input[inIdx].x;
            sumImag += d_input[inIdx].y;
        }

        // 写入输出 [nfreq]
        d_output[freqIdx].x = sumReal;
        d_output[freqIdx].y = sumImag;
    }
}

__global__ void extractReal(float *d_tfpw_stack,
                            cufftComplex *d_tfpw_stack_complex,
                            size_t nfft)
{
    // width : nfft             (时域长度)
    //        用 col in [0..nfft) 做索引
    // 这里表示：在逆变换后的时域序列中取实部
    // depth : nfreq=nfft/2+1   (若在频域中时,这里不使用)
    // height: nTraces          (道数,若仅处理一道则不需用row)

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < (int)nfft)
    {
        // 提取复数的实部，并做 / nfft 缩放（典型 IFFT 归一化）
        d_tfpw_stack[col] = d_tfpw_stack_complex[col].x / (float)nfft;
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