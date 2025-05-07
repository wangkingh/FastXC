#ifndef _CU_STRANSFORM_H_
#define _CU_STRANSFORM_H_

#include <cstddef>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 在频域中执行希尔伯特变换的核心步骤：
 *        将负频（含 DC）置零、正频乘以 2，用于生成解析信号频谱。
 *
 * @param d_inputSpectrum  [in/out] 大小=[nTraces * freqDomainSize]，频域复数数组
 * @param freqDomainSize   频域长度 (FFT 大小)
 * @param nTraces          道数 (每道都有 freqDomainSize 个点)
 */
__global__ void hilbertTransformKernel(
    cufftComplex *d_inputSpectrum,
    size_t freqDomainSize,
    size_t nTraces);

/**
 * @brief 从逆变换后的复数序列中提取实部，并进行 / nfft 缩放 (典型的 IFFT 归一化)。
 *
 * @param d_outReal    [out] 大小=[nfft]，存储提取的实部
 * @param d_inComplex  [in ] 大小=[nfft]，逆变换后的复数序列
 * @param nfft         时域长度 (原先的 FFT 大小)
 */
__global__ void extractReal(
    float *d_outReal,
    cufftComplex *d_inComplex,
    size_t nfft);

/**
 * @brief 在频-频域上，对 [chunkStartFreq..(chunkStartFreq+chunkFreqCount)) 这段“原始频率”做环形移频 + 高斯窗调制。
 *
 * @param d_inputSpectrum     [in ] 大小= nTraces * freqDomainLen。每道的解析频谱
 * @param d_modulatedSubChunk [out] 大小= nTraces * chunkFreqCount * freqDomainLen。输出调制结果
 * @param nTraces            道数
 * @param freqDomainLen      频域长度 (通常 = nfft)，用于环形偏移
 * @param totalFreqBins      总的频率数 (可能 = freqDomainLen/2+1)
 * @param chunkStartFreq     分块起始频率下标
 * @param chunkFreqCount     分块频率数量
 * @param scale              高斯窗宽度因子
 */
__global__ void gaussianModulateSub(
    const cufftComplex *__restrict__ d_inputSpectrum,
    cufftComplex *__restrict__ d_modulatedSubChunk,
    size_t nTraces,
    size_t freqDomainLen,
    int chunkStartFreq,
    int chunkFreqCount,
    float scale);

/**
 * @brief 对分块的时频结果 (多道) 计算相位一致性权重，并累加到全局权重矩阵中。
 *
 * @param d_subTransformAll  [in ] 大小= [nTraces, freqChunkSize, freqDomainSize]，分块 IFFT 后的复数数据
 * @param d_weightMatrix     [out] 大小= [nfreq, freqDomainSize]，用于累加相位矢量
 * @param freqDomainSize     频域长度
 * @param nTraces            道数
 * @param freqChunkStartIdx  分块起始频率下标
 * @param freqChunkSize      分块频率数
 */
__global__ void calculateWeightSub(
    const cufftComplex *__restrict__ d_subTransformAll,
    cuComplex *__restrict__ d_weightMatrix,
    size_t freqDomainSize,
    size_t nTraces,
    int freqChunkStartIdx,
    int freqChunkSize);

/**
 * @brief 对二维复数数组 [nfreq, npts] 应用相位权重矩阵，放大/抑制相位一致性。
 *
 * @param d_analysisSignal [in/out] 大小=[nfreq, npts]，原分析信号，在此处被乘以相位权重
 * @param d_weightMatrix   [in ]    大小=[nfreq, npts]，相位一致性权重
 * @param nfreq            频率维度大小
 * @param npts             第二维大小 (时域或另一个频域)
 * @param weight_order     相位加权幂次
 */
__global__ void applyWeight(
    cufftComplex *d_analysisSignal,
    cuComplex *d_weightMatrix,
    size_t nfreq,
    size_t npts,
    float weight_order);

/**
 * @brief 在二维复数数组 [nfreq, ntime] 上对“freq”维进行累加，输出一维 [ntime]。
 *
 * 用于将多个中心频率 (nfreq) 的结果合成为单一的时间序列 (ntime)。
 *
 * - 第一维 nfreq:
 *   表示频率轴 (行)。
 *
 * - 第二维 ntime:
 *   表示时间或 FFT 采样点 (列)。
 *
 * - 输入与输出形状:
 *   - @p d_input  : [nfreq, ntime]
 *   - @p d_output : [ntime]
 *
 * 线程策略:
 *   - 每个线程处理一个 @p timeIdx \in [0 .. ntime)，
 *   - 在内部 @p for (freqIdx in [0 .. nfreq)) 做累加。
 *
 * @param[in] d_input   输入 2D 复数数组 (形状 = [nfreq, ntime])。
 * @param[out] d_output 输出 1D 复数数组 (形状 = [ntime])。
 * @param[in] nfreq     第一维大小 (频率数)。
 * @param[in] ntime     第二维大小 (时间采样数)。
 */
__global__ void sumOverFreqAxisKernel(
    const cufftComplex *d_input,
    cufftComplex *d_output,
    size_t nfreq,
    size_t ntime);

/**
 * @brief 在二维复数数组 [nfreq, ntime] 上对“time”维进行累加，输出一维 [nfreq]。
 *
 * 用于将某条时间轴 (ntime) 的数据在各个采样点上叠加，保留频率轴 (nfreq)。
 *
 * - 第一维 nfreq:
 *   表示频率轴 (行)。
 *
 * - 第二维 ntime:
 *   表示时间或 FFT 采样点 (列)。
 *
 * - 输入与输出形状:
 *   - @p d_input  : [nfreq, ntime]
 *   - @p d_output : [nfreq]
 *
 * 线程策略:
 *   - 每个线程处理一个 @p freqIdx \in [0 .. nfreq)，
 *   - 在内部 @p for (timeIdx in [0 .. ntime)) 做累加。
 *
 * @param[in] d_input   输入 2D 复数数组 (形状 = [nfreq, ntime])。
 * @param[out] d_output 输出 1D 复数数组 (形状 = [nfreq])。
 * @param[in] nfreq     第一维大小 (频率数)。
 * @param[in] ntime     第二维大小 (时间采样数)。
 */
__global__ void sumOverTimeAxisKernel(
    const cufftComplex *d_input,
    cufftComplex *d_output,
    size_t nfreq,
    size_t ntime);

__global__ void linearSumTraces(float *d_sacdata, float *d_output, size_t num_trace, size_t nfft);
#endif // _CU_STRANSFORM_H_