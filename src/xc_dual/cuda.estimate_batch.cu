#include "cuda.estimate_batch.cuh"

// ===================== 小工具函数:测量某个plan的工作区 ===================
/**
 * @brief 测量给定FFT配置的工作区大小(仅单精度).
 *
 * @param rank        维度数(你的程序里多是1)
 * @param n           数组大小, n[0] = npts_ncf
 * @param inembed     一般 = n
 * @param istride     1
 * @param idist       npts_ncf
 * @param onembed     一般 = n
 * @param ostride     1
 * @param odist       npts_ncf
 * @param type        cufftType (CUFFT_R2C or CUFFT_C2C)
 * @param batch       批大小
 * @return            该 plan 所需的工作区字节数
 *
 * 说明: 这里做 "plan" 的时候, 我们用 cufftSetAutoAllocation(plan,0),
 *       让它只测量不实际分配.
 */
size_t MeasurePlanWorkspace(int rank, int *n,
                            int *inembed, int istride, int idist,
                            int *onembed, int ostride, int odist,
                            cufftType type,
                            int batch)
{
    cufftHandle plan;
    cufftCreate(&plan);

    // 不使用 cuFFT 的自动分配，这样我们能获取workSize
    cufftSetAutoAllocation(plan, 0);

    size_t workSize = 0;
    // 如果用 cufftMakePlanMany, 它会返回 plan中 "workSize"
    cufftResult ret = cufftMakePlanMany(plan,
                                        rank, n,
                                        inembed, istride, idist,
                                        onembed, ostride, odist,
                                        type,
                                        batch,
                                        &workSize);
    if (ret != CUFFT_SUCCESS)
    {
        fprintf(stderr, "[MeasurePlanWorkspace] cufftMakePlanMany failed with code=%d\n", ret);
        exit(1);
    }

    cufftDestroy(plan);
    return workSize;
}



/**
 * @brief 根据 TF-PWS 代码中需要分配的数据和 plan, 实际测量plan工作区来估算可用 freq_chunk_size
 *        并使用二分搜索，得到能放下的最大分块大小。
 *
 * @param gpu_id         GPU编号
 * @param num_trace      道数
 * @param npts_ncf       每道点数 (nfft)
 * @param nfreq          总频率点数
 * @param gpu_task_num   同一台GPU上并行多少个任务(不同进程/线程)
 * @param safety_factor  留一定余量(0.8表示留20%)
 * @return               计算出的 freq_chunk_size
 */
size_t EstimateFreqBatchSize(
    int gpu_id,
    size_t num_trace,
    size_t npts_ncf,
    size_t nfreq,
    size_t gpu_task_num,
    float safety_factor = 0.8f)
{
    // 1) 查询可用显存
    size_t free_bytes = QueryAvailGpuRam(gpu_id);
    size_t usable_mem = static_cast<size_t>(free_bytes * safety_factor) / gpu_task_num;

    // 2) 整理 TF-PWS 所需的 plan 参数(大部分和你的代码一致)
    int rank_hilb = 1;
    int n_hilb[1] = {static_cast<int>(npts_ncf)};
    int inembed[1] = {static_cast<int>(npts_ncf)};
    int onembed[1] = {static_cast<int>(npts_ncf)};
    int istride = 1, ostride = 1;
    int idist = static_cast<int>(npts_ncf), odist = static_cast<int>(npts_ncf);

    // === 2.1 先测量 "固定" plan 的工作区 ===
    // planfwd_trace (batch=1, R2C)
    size_t planfwd_trace_ws = MeasurePlanWorkspace(rank_hilb, n_hilb,
                                                   inembed, istride, idist,
                                                   onembed, ostride, odist,
                                                   CUFFT_R2C,
                                                   1);

    // planinv_trace (batch=nfreq, C2C)
    size_t planinv_trace_ws = MeasurePlanWorkspace(rank_hilb, n_hilb,
                                                   inembed, istride, idist,
                                                   onembed, ostride, odist,
                                                   CUFFT_C2C,
                                                   static_cast<int>(nfreq));

    // planinv_trace_one (batch=1, C2C)
    size_t planinv_trace_one_ws = MeasurePlanWorkspace(rank_hilb, n_hilb,
                                                       inembed, istride, idist,
                                                       onembed, ostride, odist,
                                                       CUFFT_C2C,
                                                       1);

    // 把这三者相加就是“固定 plan 工作区”
    size_t fixed_plan_workspace = planfwd_trace_ws + planinv_trace_ws + planinv_trace_one_ws;

    // === 2.2 计算固定数据分配(fixed_alloc): 不随 freq_chunk_size 变化 ===
    // tfpws_weight: [nfreq, npts_ncf], cuComplex
    // d_tfpw_stack_complex: [npts_ncf], cuComplex
    // d_tfpw_stack: [npts_ncf], float
    // d_spectrum_trace: [npts_ncf], cuComplex
    // modulatedAnalysis_trace: [nfreq, npts_ncf], cuComplex
    size_t fixed_alloc = 0;
    fixed_alloc += nfreq * npts_ncf * sizeof(cufftComplex); // tfpws_weight
    fixed_alloc += npts_ncf * sizeof(cufftComplex);         // d_tfpw_stack_complex
    fixed_alloc += npts_ncf * sizeof(float);                // d_tfpw_stack
    fixed_alloc += npts_ncf * sizeof(cufftComplex);         // d_spectrum_trace
    fixed_alloc += nfreq * npts_ncf * sizeof(cufftComplex); // modulatedAnalysis_trace

    // === 2.3 定义一个lambda: 给定 test_fs 时，测量 planinv_tfpws_sub + 计算总需求 ===
    auto calcNeededMem = [&](size_t test_fs) -> size_t
    {
        // 先测量 planinv_tfpws_sub (batch = num_trace * test_fs, C2C)
        size_t planinv_tfpws_sub_ws = MeasurePlanWorkspace(
            rank_hilb, n_hilb,
            inembed, istride, idist,
            onembed, ostride, odist,
            CUFFT_C2C,
            static_cast<int>(num_trace * test_fs));

        // 分块分配: modulatedAnalysis_sub [num_trace, test_fs, npts_ncf], cuComplex
        size_t chunk_alloc = num_trace * test_fs * npts_ncf * sizeof(cufftComplex);

        // 计划工作区汇总
        size_t total_plan_ws = fixed_plan_workspace + planinv_tfpws_sub_ws;

        // 数据分配汇总
        size_t total_data_alloc = fixed_alloc + chunk_alloc;

        // 这里可以再留一点点“临时 overhead”，哪怕已测了 plan workspace，
        // 以防零碎开销(比如 kernel 调用、对齐、栈空间等)：
        float overhead_factor = 1.05f; // 留5% ~ 10%等
        size_t needed = static_cast<size_t>((total_plan_ws + total_data_alloc) * overhead_factor);

        return needed;
    };

    // === 2.4 二分搜索 [1 .. nfreq] 找最大 test_fs ===
    size_t left = 1, right = nfreq;
    size_t best_fs = 0;

    while (left <= right)
    {
        size_t mid = (left + right) / 2;
        size_t needed = calcNeededMem(mid);

        if (needed <= usable_mem)
        {
            best_fs = mid;  // 能放下 => 记录
            left = mid + 1; // 继续尝试更大
        }
        else
        {
            if (mid == 0)
                break;
            right = mid - 1;
        }
    }

    if (best_fs == 0)
    {
        fprintf(stderr, "[EstimateFreqBatchSize] Even sub_nfreq=1 is too large for available GPU mem.\n");
        exit(1);
    }

    return best_fs;
}