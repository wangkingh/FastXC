#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cuda.util.cuh"
#include "cuda.pws_util.cuh"
#include "cuda.stransform.cuh"
#include "cuda.estimate_batch.cuh"
#include "cuda.xc_dual.cuh"
#include <stdlib.h>
extern "C"
{
#include "sac.h"
#include "arguproc.h"
#include "read_segspec.h"
#include "gen_ncf_path.h"
#include "gen_pair.h"
#include "segspec.h"
#include "cal_dist.h"
#include "util.h"
}

int main(int argc, char **argv)
{
    // ======================== 解析输入参数 ===================================
    ARGUTYPE argument;
    ArgumentProcess(argc, argv, &argument);
    float cc_len = argument.cc_len;
    char *ncf_dir = argument.ncf_dir;
    size_t gpu_id = argument.gpu_id;
    int save_linear = argument.save_linear;
    int gpu_num = argument.gpu_num;
    int save_pws = argument.save_pws;
    int save_tfpws = argument.save_tfpws;
    int save_segment = argument.save_segment;
    float threshold_distance = argument.threshold_distance;
    int cpu_count = argument.cpu_count;
    CUDACHECK(cudaSetDevice(gpu_id));

    // ========================= 读取首个频谱头文件，解析一些参数 =================
    size_t pair_count = 0;
    FilePair *src_sta_pairs = NULL;
    SEGSPEC src_segspec_hd, sta_segspec_hd;
    find_matching_files(argument.src_files_list, argument.sta_files_list, &src_sta_pairs, &pair_count);
    if (pair_count == 0)
    {
        fprintf(stderr, "Error: No matching files found\n");
        return 0;
    }
    if (pair_count == 1)
    {
        save_tfpws = 0;
        save_pws = 0;
        save_linear = 1;
    }
    read_spechead(src_sta_pairs[0].source_path, &src_segspec_hd);
    read_spechead(src_sta_pairs[0].station_path, &sta_segspec_hd);
    SACHEAD ncf_hd;
    SacheadProcess(&ncf_hd, &src_segspec_hd, &sta_segspec_hd, cc_len);
    if (ncf_hd.dist > threshold_distance) // 距离大于阈值，不进行计算
    {
        return 0;
    }
    int half_npts_ncf = (int)floorf(cc_len / src_segspec_hd.dt);
    int npts_ncf = 2 * half_npts_ncf + 1;
    size_t nstep = src_segspec_hd.nstep;
    size_t nspec = src_segspec_hd.nspec;
    float dt = src_segspec_hd.dt;
    int nfft_cc = 2 * (nspec - 1);
    size_t vec_count = nstep * nspec;              // 单到源/台数据的点数
    size_t vec_size = vec_count * sizeof(complex); // 单道源/台的大小(比特)

    // ===================== 开辟 host 内存空间 ==========================
    complex *src_buffer = NULL; // 虚拟源的频谱
    complex *sta_buffer = NULL; // 虚拟接收机的频谱
    float *h_sacdata = NULL;    // 临时保留输出的互相关
    CpuMalloc((void **)&src_buffer, pair_count * vec_size);
    CpuMalloc((void **)&sta_buffer, pair_count * vec_size);
    CpuMalloc((void **)&h_sacdata, pair_count * npts_ncf * sizeof(float));
    printf("pair_count is %lu\n", pair_count);
    printf("size is %lu\n", pair_count * vec_size);
    if (!src_buffer)
    {
        fprintf(stderr, "Memory allocation failed\n");
        // 进行错误处理，例如释放已分配的资源或返回错误代码
        return -1;
    }

    if (!sta_buffer)
    {
        fprintf(stderr, "Memory allocation failed\n");
        // 进行错误处理，例如释放已分配的资源或返回错误代码
        return -1;
    }

    // ============= 初始化 互相关计算 变量 ==================
    cuComplex *d_src_buffer = NULL;                  // 1,能够储存 batch 个输入的source 虚拟源频谱
    cuComplex *d_sta_buffer = NULL;                  // 1,能够储存 batch 个输入的station 虚拟接收机频谱
    cuComplex *d_ncf_freq_buffer_multi_steps = NULL; // 1,能够储存 batch 个包含多段的的NCF结果(频率域)
    cuComplex *d_ncf_freq_buffer = NULL;             // 1/nstep,能够储存 batch 个多段叠加后的NCF结果(频率域)
    float *d_ncf_time_buffer = NULL;                 // 0.5/nstep,能够储存 batch 个输出的NCF结果(时间域),freq_buffers的反变换
    float *d_ncf_buffer_all = NULL;                  // 能够储存 pair_count 个NCF结算结果(时间域)，但是只保留中间npts_ncf个点
    int *d_sgn_vec = NULL;                           // 储存相移向量, 保证频域NCF转变到时域后0时刻在序列中心
    cufftHandle plan_inv_cc;                         // 设置从频率域到时间域的R2C FFT plan,为CrossCorrelation 而设置
    int rank = 1;
    int n[1] = {nfft_cc};
    int inembed[1] = {nfft_cc};
    int onembed[1] = {nfft_cc};
    int istride = 1;
    int idist = nfft_cc;
    int ostride = 1;
    int odist = nfft_cc;
    cufftType type = CUFFT_C2R; // 将频谱相乘之后的结果转化成时间序列
    int numType = 1;
    cufftType typeArr[1] = {type};

    // 直接计算GPU中能计算多少对台, 修改自蒋磊代码
    size_t unit_batch_ram = vec_size * (3 + 1.5 / nstep);     // 包括输入和输出的时间/频率序列
    size_t fixed_ram = pair_count * npts_ncf * sizeof(float); // 给输出数据使用
    fixed_ram += nspec * sizeof(int);                         // 给相移向量sgn_vec使用
    size_t batch = EstimateGpuBatch_CC(gpu_id, fixed_ram, unit_batch_ram, numType,
                                       rank, n, inembed, istride, idist, onembed,
                                       ostride, odist, typeArr);
    batch = batch > pair_count ? pair_count : batch;
    printf("batch size: %lu\n", batch);
    // 开辟空间
    GpuMalloc((void **)&d_src_buffer, batch * vec_size);
    GpuMalloc((void **)&d_sta_buffer, batch * vec_size);
    // GpuMalloc((void **)&d_ncf_freq_buffer_multi_steps, batch * vec_size * sizeof(cuComplex)); // 长度是输入的两倍,因为输入数据只保留了一半的频谱
    GpuMalloc((void **)&d_ncf_freq_buffer_multi_steps, batch * vec_size); // 长度是输入的两倍,因为输入数据只保留了一半的频谱
    GpuMalloc((void **)&d_ncf_freq_buffer, batch * nfft_cc * sizeof(cuComplex));
    GpuMalloc((void **)&d_ncf_time_buffer, batch * nfft_cc * sizeof(float)); //
    GpuMalloc((void **)&d_ncf_buffer_all, pair_count * npts_ncf * sizeof(float));
    CUDACHECK(cudaMemset(d_ncf_buffer_all, 0, pair_count * npts_ncf * sizeof(float))); // 初始化计算结果

    GpuCalloc((void **)&d_sgn_vec, nspec * sizeof(int));
    CufftPlanAlloc(&plan_inv_cc, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);

    // =========================== 将数据从文件列表读入到 host =================================
    // cpu_count = 1;
    ThreadPoolRead *read_pool = create_threadpool_read(cpu_count);
    parallel_read_segspec(read_pool, pair_count, src_sta_pairs, src_buffer, sta_buffer, vec_size, cpu_count);
    destroy_threadpool_read(read_pool);

    // 定义网格
    dim3 dimGrid_1D, dimBlock_1D;
    dim3 dimGrid_2D, dimBlock_2D;
    dim3 dimGrid_3D, dimBlock_3D;

    // 计算相移向量
    DimCompute1D(&dimGrid_1D, &dimBlock_1D, nspec);
    generateSignVector<<<dimGrid_1D, dimBlock_1D>>>(d_sgn_vec, nspec);
    printf("Start to calculate NCF\n");
    for (size_t pair_index = 0; pair_index < pair_count; pair_index += batch)
    {
        // 确认对应计算批次的数据的位置
        size_t start_index = pair_index;
        size_t end_index = pair_index + batch > pair_count ? pair_count : pair_index + batch;
        size_t current_batch = end_index - start_index;
        // 初始化计算数组
        CUDACHECK(cudaMemcpy(d_src_buffer, src_buffer + start_index * vec_count, current_batch * vec_size, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_sta_buffer, sta_buffer + start_index * vec_count, current_batch * vec_size, cudaMemcpyHostToDevice));
        // CUDACHECK(cudaMemset(d_ncf_freq_buffer_multi_steps, 0, current_batch * vec_size * sizeof(cuComplex)));
        CUDACHECK(cudaMemset(d_ncf_freq_buffer_multi_steps, 0, current_batch * vec_size));
        CUDACHECK(cudaMemset(d_ncf_freq_buffer, 0, current_batch * nfft_cc * sizeof(cuComplex)));
        CUDACHECK(cudaMemset(d_ncf_time_buffer, 0, current_batch * nfft_cc * sizeof(float)));
        // 以 nspec为宽，nstep*current_batch为高, 将所有频谱一次性相乘完
        DimCompute(&dimGrid_2D, &dimBlock_2D, nspec, nstep * current_batch);
        complexMul2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_src_buffer, d_sta_buffer, nspec,
                                                        d_ncf_freq_buffer_multi_steps, nspec,
                                                        nspec, nstep * current_batch);

        // 将互相关频率域每一天的计算结果进行线性叠加
        DimCompute(&dimGrid_2D, &dimBlock_2D, nspec, current_batch); // 需要修改网格大小
        for (size_t step_idx = 0; step_idx < nstep; step_idx++)
        {
            csum2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_ncf_freq_buffer, nfft_cc,
                                                      d_ncf_freq_buffer_multi_steps, nspec,
                                                      nspec, current_batch, step_idx, nstep);
        }

        DimCompute(&dimGrid_2D, &dimBlock_2D, nspec, current_batch); // 再次重新计算网格
        applyPhaseShiftKernel<<<dimGrid_2D, dimBlock_2D>>>(d_ncf_freq_buffer, d_sgn_vec, nfft_cc, nspec, current_batch);

        // ============= 将互相关频率域的计算结果转换到时间域========================
        CUFFTCHECK(cufftExecC2R(plan_inv_cc, d_ncf_freq_buffer, d_ncf_time_buffer));
        DimCompute(&dimGrid_2D, &dimBlock_2D, nfft_cc, current_batch);
        InvNormalize2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_ncf_time_buffer, nfft_cc, nfft_cc, current_batch, dt);

        // ============= 将互相关时间域的计算结果保存到 d_ncf_buffer_all ===============
        CUDACHECK(cudaMemcpy2D(d_ncf_buffer_all + start_index * npts_ncf, npts_ncf * sizeof(float),
                               d_ncf_time_buffer + (nspec - half_npts_ncf - 1), nfft_cc * sizeof(float),
                               npts_ncf * sizeof(float), current_batch, cudaMemcpyDeviceToDevice));
    }

    //   将计算结果
    // 释放计算过程的内存和显存，除了d_ncf_buffer_all
    CpuFree((void **)&src_buffer);
    CpuFree((void **)&sta_buffer);
    GpuFree((void **)&d_src_buffer);
    GpuFree((void **)&d_sta_buffer);
    GpuFree((void **)&d_ncf_freq_buffer_multi_steps);
    GpuFree((void **)&d_ncf_freq_buffer);
    GpuFree((void **)&d_ncf_time_buffer);
    GpuFree((void **)&d_sgn_vec);
    cufftDestroy(plan_inv_cc);
    // ============================= 结束互相关计算 =======================================

    // ============================= 叠加环节 ============================================
    // 相关 变量名命名的规范遵循cuda_ver tf-pws 代码
    size_t num_trace = pair_count; // 防呆，将道数设置为数据对数
    size_t nfft = npts_ncf;        // 防呆，修改变量名命名

    // ============================== 分 配 空 间 =========================================
    float *d_linear_stack = NULL; // 保存线性叠加结果
    GpuMalloc((void **)&d_linear_stack, nfft * sizeof(float));
    DimCompute1D(&dimGrid_1D, &dimBlock_1D, nfft);
    linearSumTraces<<<dimGrid_1D, dimBlock_1D>>>(d_ncf_buffer_all, d_linear_stack, num_trace, nfft);
    cudaDeviceSynchronize(); // 确保线性叠加完成

    // ====================== 生成输出文件的文件名 ===============================
    char src_file_name[MAXNAME];
    char sta_file_name[MAXNAME];

    char src_station[16], src_channel[16];
    char sta_station[16], sta_channel[16];

    char src_year[5], src_jday[4], src_hm[5];
    char sta_year[5], sta_jday[4], sta_hm[5];

    strncpy(src_file_name, basename(src_sta_pairs[0].source_path), MAXNAME);
    strncpy(sta_file_name, basename(src_sta_pairs[0].station_path), MAXNAME);

    // Split file names into individual components
    SplitFileName(src_file_name, ".", src_station, src_year, src_jday, src_hm, src_channel);
    SplitFileName(sta_file_name, ".", sta_station, sta_year, sta_jday, sta_hm, sta_channel);
    char ccf_name[MAXNAME];
    char ccf_name_segment[MAXNAME];

    snprintf(ccf_name, sizeof(ccf_name), "%s-%s.%s-%s.ncf.sac",
             src_station, sta_station, src_channel, sta_channel);

    snprintf(ccf_name_segment, sizeof(ccf_name_segment), "%s-%s.%s-%s.sac_segment",
             src_station, sta_station, src_channel, sta_channel);
    char ccf_dir[MAXLINE];
    char ccf_path[2 * MAXLINE];
    // =================== 写 出 线 性 叠 加 数 据======================
    if (save_linear == 1)
    {
        float *linear_stack = (float *)malloc(nfft * sizeof(float)); // 线性加权叠加结果
        CUDACHECK(cudaMemcpy(linear_stack, d_linear_stack, nfft * sizeof(float), cudaMemcpyDeviceToHost));
        snprintf(ccf_dir, sizeof(ccf_dir), "%s/linear/%s-%s/", ncf_dir, src_station, sta_station);
        CreateDir(ccf_dir);
        snprintf(ccf_path, 2 * MAXLINE, "%s/%s", ccf_dir, ccf_name);
        write_sac(ccf_path, ncf_hd, linear_stack); // 使用新的文件名写文件
        CpuFree((void **)&linear_stack);
    }

    // ================  写入 每一时间段计算的 NCF 结果 ============================
    if (save_segment == 1)
    {

        float *ncf_segments_buffer = (float *)malloc(num_trace * nfft * sizeof(float)); // 叠前每一个时间段的结果
        CUDACHECK(cudaMemcpy(ncf_segments_buffer, d_ncf_buffer_all, num_trace * nfft * sizeof(float), cudaMemcpyDeviceToHost));
        snprintf(ccf_dir, sizeof(ccf_dir), "%s/segments/%s-%s/", ncf_dir, src_station, sta_station);
        CreateDir(ccf_dir);
        snprintf(ccf_path, 2 * MAXLINE, "%s/%s", ccf_dir, ccf_name_segment); // 打开文件进行写入,存储每一道的数据信息
        FILE *fp = fopen(ccf_path, "wb");
        if (fp == NULL)
        {
            fprintf(stderr, "Error: Unable to open file %s for writing\n", ccf_path);
            free(ncf_segments_buffer);
            return 1;
        }

        // 写入总匹配对数
        if (fwrite(&pair_count, sizeof(pair_count), 1, fp) != 1)
        {
            fprintf(stderr, "Error: Failed to write pair count to file %s\n", ccf_path);
            fclose(fp);
            free(ncf_segments_buffer);
            return 1;
        }

        // 写入每个匹配对的时间信息
        for (int i = 0; i < pair_count; i++)
        {
            if (fwrite(&src_sta_pairs[i].time, sizeof(src_sta_pairs[i].time), 1, fp) != 1)
            {
                fprintf(stderr, "Error: Failed to write time info to file %s\n", ccf_path);
                fclose(fp);
                free(ncf_segments_buffer);
                return 1;
            }
        }

        // 写入头部信息
        if (fwrite(&ncf_hd, sizeof(ncf_hd), 1, fp) != 1)
        {
            fprintf(stderr, "Error: Failed to write header to file %s\n", ccf_path);
            fclose(fp);
            free(ncf_segments_buffer);
            return 1;
        }

        // 写入大数据块
        if (fwrite(ncf_segments_buffer, sizeof(float), num_trace * nfft, fp) != num_trace * nfft)
        {
            fprintf(stderr, "Error: Failed to write data to file %s\n", ccf_path);
            fclose(fp);
            free(ncf_segments_buffer);
            return 1;
        }
        // 清理资源
        fclose(fp);
        CpuFree((void **)&ncf_segments_buffer);
    }

    if ((save_pws == 0) && (save_tfpws == 0))
    {
        GpuFree((void **)d_ncf_buffer_all);
        GpuFree((void **)d_linear_stack);
        return 0;
    }

    // 相位加权叠加部分
    cuComplex *d_spectrum = NULL; // 输入时间序列对应的频谱,用于希尔伯特变换
    GpuMalloc((void **)&d_spectrum, num_trace * nfft * sizeof(cufftComplex));

    // =====0.5 希尔伯特变换，无论是pws还是tf-pws,都需要进行希尔伯特变换将数组变为解析信号========
    // 创建给希尔伯特变换的FFT计划,执行正变换,将每一道数据转换为频率
    int rank_hilb = 1;
    int n_hilb[1] = {(int)nfft};
    int inembed_hilb[1] = {(int)nfft};
    int onembed_hilb[1] = {(int)nfft};
    int istride_hilb = 1;
    int idist_hilb = (int)nfft;
    int ostride_hilb = 1;
    int odist_hilb = (int)nfft;

    cufftHandle plan_fwd;
    CufftPlanAlloc(&plan_fwd, rank_hilb, n_hilb, inembed_hilb, istride_hilb, idist_hilb, onembed_hilb, ostride_hilb, odist_hilb, CUFFT_R2C, num_trace);
    CUFFTCHECK(cufftExecR2C(plan_fwd, (cufftReal *)d_ncf_buffer_all, (cufftComplex *)d_spectrum));
    DimCompute2D(&dimGrid_2D, &dimBlock_2D, nfft, num_trace);
    hilbertTransformKernel<<<dimGrid_2D, dimBlock_2D>>>(d_spectrum, nfft, num_trace);

    // ======================1. PWS叠加, 使用蒋磊代码=======================================
    if (save_pws == 1)
    {
        float *d_pw_stack = NULL; // 仅保留相位加权叠加结果时用到
        // 为 PWS 计算开辟空间
        cuComplex *hilbert_complex = NULL; // 创建储存反变换后的解析信号的数组
        cuComplex *analyze_mean;           // 解析信号求和再求平均,原代码中的divide_mean
        float *abs_hilb_amp;               // analyze_mean的绝对值,作为权重调制线性叠加的结果
        float *weight;                     // 存储由多道解析信号加和得到的权重
        cufftHandle plan_inv_pws;          // C2C 用于将希尔伯特频谱转变为解析信号
        CUDACHECK(cudaMalloc((void **)&hilbert_complex, num_trace * nfft * sizeof(cuComplex)));
        CUDACHECK(cudaMalloc(&abs_hilb_amp, num_trace * nfft * sizeof(float)));
        CUDACHECK(cudaMalloc(&analyze_mean, nfft * sizeof(cufftComplex)));
        CUDACHECK(cudaMalloc(&weight, nfft * sizeof(float)));
        CUDACHECK(cudaMalloc(&d_pw_stack, nfft * sizeof(float)));
        cufftPlanMany(&plan_inv_pws, rank_hilb, n_hilb, inembed_hilb, istride_hilb, idist_hilb, onembed_hilb, ostride_hilb, odist_hilb, CUFFT_C2C, num_trace);
        DimCompute2D(&dimGrid_2D, &dimBlock_2D, nfft, num_trace);

        // C2C 反变换, 希尔伯特频谱[d_spectrum]->解析信号[hilbert_complex]
        CUFFTCHECK(cufftExecC2C(plan_inv_pws, d_spectrum, hilbert_complex, CUFFT_INVERSE));

        // 归一化解析信号到单位圆上,nfft归一化因子用来控制序列能量
        DimCompute1D(&dimGrid_1D, &dimBlock_1D, nfft * num_trace);
        cudaNormalizeComplex<<<dimGrid_1D, dimBlock_1D>>>(hilbert_complex, nfft * num_trace, nfft);

        // 计算权重并使用权重归一化
        DimCompute1D(&dimGrid_1D, &dimBlock_1D, nfft);
        cudaMean<<<dimGrid_1D, dimBlock_1D>>>(hilbert_complex, analyze_mean, num_trace, nfft);
        cudaMultiply<<<dimGrid_1D, dimBlock_1D>>>(d_linear_stack, analyze_mean, d_pw_stack, nfft);

        // 释放线性叠加的显卡空间
        GpuFree((void **)&abs_hilb_amp);
        GpuFree((void **)&hilbert_complex);
        GpuFree((void **)&analyze_mean);
        GpuFree((void **)&weight);
        cufftDestroy(plan_inv_pws);

        float *pw_stack = (float *)malloc(nfft * sizeof(float)); // 相位加权叠加结果
        CUDACHECK(cudaMemcpy(pw_stack, d_pw_stack, nfft * sizeof(float), cudaMemcpyDeviceToHost));
        snprintf(ccf_dir, sizeof(ccf_dir), "%s/pws/%s-%s/", ncf_dir, src_station, sta_station);
        CreateDir(ccf_dir);
        snprintf(ccf_path, 2 * MAXLINE, "%s/%s", ccf_dir, ccf_name);
        write_sac(ccf_path, ncf_hd, pw_stack); // 使用新的文件名写文件
        CpuFree((void **)&pw_stack);
        GpuFree((void **)&d_pw_stack);
        cudaDeviceSynchronize(); 
    }

    // ======== 2. tf-pws 时频域相位加权叠加, 修改自 Li Guoliang 的程序 ===========
    if (save_tfpws == 1)
    {
        float *d_tfpw_stack = NULL;
        // 2.0 计算一些参数
        size_t num_freq_bins = nfft / 2 + 1; // 设置调制频率数, 长度等同于Nyquist采样频率
        size_t freq_batch_size = EstimateFreqBatchSize(
            gpu_id,
            num_trace,
            nfft,
            num_freq_bins,
            gpu_num,
            0.8f);

        size_t num_freq_batches = (num_freq_bins + freq_batch_size - 1) / freq_batch_size;
        float weight_order = 1; // 设置加权阶数
        float scale = 0.1;      // 设置高斯窗的宽度, 0.1是一个比较好的值, 0.2会导致频谱变得很宽

        // ========== 2.1 分配 GPU 内存 ============
        // (A) 权重矩阵: 大小 [num_freq_bins, nfft], 复数
        cuComplex *d_tfpws_weight;
        CUDACHECK(cudaMalloc(&d_tfpws_weight, num_freq_bins * nfft * sizeof(cuComplex)));
        CUDACHECK(cudaMemset(d_tfpws_weight, 0, num_freq_bins * nfft * sizeof(cuComplex)));

        // (B) TF-PWS 最终输出(复数 & 实数)
        cufftComplex *d_tfpw_stack_complex;
        CUDACHECK(cudaMalloc((void **)&d_tfpw_stack_complex, nfft * sizeof(cufftComplex)));
        CUDACHECK(cudaMalloc((void **)&d_tfpw_stack, nfft * sizeof(float)));

        // (C) 叠后数据频谱 + 调制后的叠后数据 (时频表示)
        cufftComplex *d_stacked_spectrum;    // [nfft]  (仅1道叠后数据)
        cufftComplex *d_st_stacked_spectrum; // [nfft * num_freq_bins] (S变换结果)
        CUDACHECK(cudaMalloc((void **)&d_stacked_spectrum, nfft * sizeof(cufftComplex)));
        CUDACHECK(cudaMalloc((void **)&d_st_stacked_spectrum, num_freq_bins * nfft * sizeof(cufftComplex)));

        // ========== 2.2 创建 FFT 计划 ============
        // 用到 3 个 plan: fftPlanFwd_singleTrace, fftPlanInv_multiFreq, fftPlanInv_singleTrace, planinv_tfpws
        cufftHandle fftPlanFwd_singleTrace, fftPlanInv_multiFreq, fftPlanInv_singleTrace;

        int rank_hilb = 1;
        int n_hilb[1] = {(int)nfft};
        int inembed_hilb[1] = {(int)nfft};
        int onembed_hilb[1] = {(int)nfft};
        int istride_hilb = 1;
        int idist_hilb = (int)nfft;
        int ostride_hilb = 1;
        int odist_hilb = (int)nfft;

        // ① 叠后数据 R2C
        CUFFTCHECK(cufftPlanMany(&fftPlanFwd_singleTrace,
                                 rank_hilb, n_hilb,
                                 inembed_hilb, istride_hilb, idist_hilb,
                                 onembed_hilb, ostride_hilb, odist_hilb,
                                 CUFFT_R2C, 1));

        // ② 多频带 IFFT
        CUFFTCHECK(cufftPlanMany(&fftPlanInv_multiFreq,
                                 rank_hilb, n_hilb,
                                 inembed_hilb, istride_hilb, idist_hilb,
                                 onembed_hilb, ostride_hilb, odist_hilb,
                                 CUFFT_C2C, (int)num_freq_bins));

        // ③ 单条 IFFT（最后一步）
        CUFFTCHECK(cufftPlanMany(&fftPlanInv_singleTrace,
                                 rank_hilb, n_hilb,
                                 inembed_hilb, istride_hilb, idist_hilb,
                                 onembed_hilb, ostride_hilb, odist_hilb,
                                 CUFFT_C2C, 1));

        // ========== 2.3 对[叠后数据]做 Stockwell 变换(相当于S变换) ============
        // (A) R2C => 叠后频谱
        CUFFTCHECK(cufftExecR2C(fftPlanFwd_singleTrace, (cufftReal *)d_linear_stack, (cufftComplex *)d_stacked_spectrum));

        // (B) hilbertTransformKernel => 得到解析信号频谱 (后半部分为0)
        DimCompute2D(&dimGrid_2D, &dimBlock_2D, nfft, 1);
        hilbertTransformKernel<<<dimGrid_2D, dimBlock_2D>>>(d_stacked_spectrum, nfft, 1);
        CUDACHECK(cudaDeviceSynchronize());

        // (C) gaussianModulate => 得到叠后数据的[频率调制]  (1道 => batch=1)
        DimCompute3D(&dimGrid_3D, &dimBlock_3D, nfft, num_freq_bins, 1);
        gaussianModulateSub<<<dimGrid_3D, dimBlock_3D>>>(
            /* d_inputSpectrum       */ d_stacked_spectrum,
            /* d_modulatedSubChunk   */ d_st_stacked_spectrum,
            /* nTraces               */ 1,
            /* freqDomainLen         */ nfft,
            /* chunkStartFreq        */ 0,
            /* chunkFreqCount        */ (int)num_freq_bins,
            /* scale                 */ scale);
        CUDACHECK(cudaDeviceSynchronize());

        // (D) C2C 逆变换 => 中心频率-中心时间 域
        CUFFTCHECK(cufftExecC2C(fftPlanInv_multiFreq, d_st_stacked_spectrum, d_st_stacked_spectrum, CUFFT_INVERSE));
        CUDACHECK(cudaDeviceSynchronize());

        // 销毁 fwd 和 inv_trace 这两个用完的 plan
        cufftDestroy(fftPlanFwd_singleTrace);
        cufftDestroy(fftPlanInv_multiFreq);

        // ========== 2.4 对多道数据一次性计算权重 =============
        // (A) 我们先分配一个“子频段调制”临时数组 d_st_subbatch
        //     大小： [num_trace, freq_batch_size, nfft]
        cufftComplex *d_st_subbatch;
        CUDACHECK(cudaMalloc((void **)&d_st_subbatch, (size_t)num_trace * freq_batch_size * nfft * sizeof(cufftComplex)));
        CUDACHECK(cudaDeviceSynchronize());
        // (B) 还需要一个子批次 FFT 计划 => batch = num_trace*freq_batch_size
        cufftHandle fftPlanInv_subMultiTrace;
        CUFFTCHECK(cufftPlanMany(&fftPlanInv_subMultiTrace,
                                 rank_hilb, n_hilb,
                                 inembed_hilb, istride_hilb, idist_hilb,
                                 onembed_hilb, ostride_hilb, odist_hilb,
                                 CUFFT_C2C, (int)(num_trace * freq_batch_size)));
        CUDACHECK(cudaDeviceSynchronize());
        // (D) 频率分块循环
        for (int ichunk = 0; ichunk < num_freq_batches; ichunk++)
        {
            int f_start = ichunk * freq_batch_size;
            int sub_nfreq = ((f_start + freq_batch_size) <= (int)num_freq_bins)
                                ? freq_batch_size
                                : ((int)num_freq_bins - f_start);

            // (D1) 调用 "gaussianModulateSub<<<>>>", 仅对 [f_start..f_start+sub_nfreq) 做高斯窗调制
            //      将 d_spectrum_temp => d_st_subbatch
            //      大小: [num_trace, sub_nfreq, nfft]
            dim3 gridG, blockG;
            DimCompute3D(&gridG, &blockG, nfft, sub_nfreq, num_trace);
            gaussianModulateSub<<<gridG, blockG>>>(
                /* d_inputSpectrum       */ d_spectrum,
                /* d_modulatedSubChunk   */ d_st_subbatch,
                /* nTraces               */ num_trace,
                /* freqDomainLen         */ nfft,
                /* chunkStartFreq        */ f_start,
                /* chunkFreqCount        */ sub_nfreq,
                /* scale                 */ scale);

            // (D2) 对这块做 IFFT => fftPlanInv_subMultiTrace
            cufftExecC2C(fftPlanInv_subMultiTrace, d_st_subbatch, d_st_subbatch, CUFFT_INVERSE);
            CUDACHECK(cudaDeviceSynchronize());
            // (D3) 计算权重 => 只对子频段 [f_start..f_start+sub_nfreq) 做相位一致性统计
            //      累加到全局 d_tfpws_weight
            dim3 gridW, blockW;
            DimCompute2D(&gridW, &blockW, nfft, sub_nfreq);
            calculateWeightSub<<<gridW, blockW>>>(
                d_st_subbatch,  // [num_trace, sub_nfreq, nfft]
                d_tfpws_weight, // [num_freq_bins, nfft]
                nfft,
                num_trace,
                f_start,
                sub_nfreq);
        }
        CUDACHECK(cudaDeviceSynchronize());

        // ========== 2.5 应用权重到叠后数据 (d_st_stacked_spectrum ==========

        // (A) Kernel applyWeight: 大小 = [num_freq_bins, nfft]
        //     叠后数据只有1道 => d_st_stacked_spectrum的 shape ~ [num_freq_bins, nfft]
        DimCompute2D(&dimGrid_2D, &dimBlock_2D, nfft, num_freq_bins);

        // 对 [num_freq_bins, nfft] 的叠后数据乘以 d_tfpws_weight
        applyWeight<<<dimGrid_2D, dimBlock_2D>>>(
            d_st_stacked_spectrum, // [num_freq_bins, nfft]
            d_tfpws_weight,        // [num_freq_bins, nfft]
            num_freq_bins,
            nfft,
            weight_order);
        CUDACHECK(cudaDeviceSynchronize());
        // (B) 将加权后的叠后数据在“时间方向”叠加 => [num_freq_bins]
        //     即把 [num_freq_bins, nfft] 上每个 time slice 累加 => [num_freq_bins]
        CUDACHECK(cudaMemset(d_stacked_spectrum, 0, nfft * sizeof(cufftComplex)));
        // DimCompute1D(&dimGrid_1D, &dimBlock_1D, nfft);
        // 加和前一半的频率，只有前一半的频率有意义
        DimCompute1D(&dimGrid_1D, &dimBlock_1D, num_freq_bins);
        sumOverTimeAxisKernel<<<dimGrid_1D, dimBlock_1D>>>(
            d_st_stacked_spectrum, // d_tfAnalysis, shape=[num_freq_bins, nfft]
            d_stacked_spectrum,    // d_outSpectrum, shape=[nfft]
            num_freq_bins,
            nfft);
        CUDACHECK(cudaDeviceSynchronize());
        // (C) 逆变换 => 得到时域 TF-PWS 结果 (复数)
        CUFFTCHECK(cufftExecC2C(fftPlanInv_singleTrace, d_stacked_spectrum, d_tfpw_stack_complex, CUFFT_INVERSE));
        cufftDestroy(fftPlanInv_singleTrace);
        CUDACHECK(cudaDeviceSynchronize());
        // (D) 提取实部 => d_tfpw_stack
        DimCompute1D(&dimGrid_1D, &dimBlock_1D, nfft);
        extractReal<<<dimGrid_1D, dimBlock_1D>>>(d_tfpw_stack,
                                                 d_tfpw_stack_complex,
                                                 nfft);
        // (E) 销毁临时数组
        GpuFree((void **)&d_stacked_spectrum);
        GpuFree((void **)&d_st_stacked_spectrum);
        GpuFree((void **)&d_tfpw_stack_complex);
        GpuFree((void **)&d_tfpws_weight);

        // ========== 2.6 写结果到 SAC =============
        float *tfpw_stack = (float *)malloc(nfft * sizeof(float));
        CUDACHECK(cudaMemcpy(tfpw_stack, d_tfpw_stack,
                             nfft * sizeof(float), cudaMemcpyDeviceToHost));
        snprintf(ccf_dir, sizeof(ccf_dir), "%s/tfpws/%s-%s/", ncf_dir, src_station, sta_station);
        CreateDir(ccf_dir);
        snprintf(ccf_path, 2 * MAXLINE, "%s/%s", ccf_dir, ccf_name);
        CreateDir(ccf_dir);
        write_sac(ccf_path, ncf_hd, tfpw_stack);

        free(tfpw_stack);
        GpuFree((void **)&d_tfpw_stack);
    }

    // 销毁输出数组
    GpuFree((void **)&d_linear_stack);
    GpuFree((void **)&d_ncf_buffer_all);
    return 0;
}