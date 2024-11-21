#ifndef _CU_STRANSFORM_H_
#define _CU_STRANSFORM_H_
#include <cstddef>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// 将时间序列复制到复数解析序列的内存空间
__global__ void copyRealToComplex(float *realArray, cufftComplex *complexArray, size_t width, size_t height);

// 计算高斯调制矩阵
__global__ void compute_g_matrix_kernel(float *g_matrix, size_t width, size_t depth, float scale);

// 频率域希尔伯特变换
__global__ void hilbertTransformKernel(cufftComplex *d_spectrum, size_t nfft, size_t num_trace);

// 高斯窗调制
__global__ void gaussianModulate(cufftComplex *d_spectrum, cufftComplex *modulatedSpectrum, float *g_matrix,
                                 size_t num_trace, size_t nfreq, size_t nfft);

// 从各道、各调制频率，各解析信号计算频率-时间域权重矩阵
__global__ void calculateWeight(cufftComplex *trans_all, cuComplex *weight_complex, size_t nfft, size_t nfreq, size_t num_trace);

// 计算线性求和
__global__ void linearSumTraces(float *d_sacdata, float *d_output, size_t num_trace, size_t nfft);

// 将权重应用到解析信号上
__global__ void applyWeight(cufftComplex *analysisSignal, cuComplex *weight, size_t nfreq, size_t nfft, float weight_order);

// 简单求和
__global__ void sumComplexSpectrumSimple(cufftComplex *analysisSignal, cufftComplex *d_pws_spectrum, size_t nfreq, size_t nfft);

// 提取实部
__global__ void extractReal(float *d_tfpw_stack, cufftComplex *d_tfpw_stack_complex, size_t nfft);

#endif
