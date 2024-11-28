#include "cuda.xc_dual.cuh"
#include "cuda.util.cuh"
#include "segspec.h"
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <linux/limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define min(x, y) ((x) < (y) ? (x) : (y))

extern "C"
{
#include "sac.h"
#include "arguproc.h"
#include "read_segspec.h"
#include "read_spec_lst.h"
#include "par_read_spec.h"
#include "par_write_sac.h"
#include "par_filter_nodes.h"
#include "gen_ccfpath.h"
#include "util.h"
}

int main(int argc, char **argv)
{

  ARGUTYPE argument;
  ArgumentProcess(argc, argv, &argument);

  float cclength = argument.cclength;
  char *ncf_dir = argument.ncf_dir;
  size_t gpu_id = argument.gpu_id;
  size_t gpu_task_num = argument.gpu_task_num;
  size_t cpu_count = argument.cpu_count;
  float max_distance = argument.max_distance;

  CUDACHECK(cudaSetDevice(gpu_id));

  FilePaths *pSrcPaths = read_spec_lst(argument.src_lst_path); // 生成虚拟源的路径列表
  FilePaths *pStaPaths = read_spec_lst(argument.sta_lst_path); // 生成虚拟接收机路径列表

  SEGSPEC sample_spechead; // 读取第一个虚拟源文件, 获得一些参数
  read_spechead(pSrcPaths->paths[0], &sample_spechead);
  int nspec = sample_spechead.nspec;
  int nstep = sample_spechead.nstep;
  float delta = sample_spechead.dt;
  int nfft = 2 * (nspec - 1);
  int nhalfcc = (int)floorf(cclength / delta);
  int ncc = 2 * nhalfcc + 1;

  // 估算主机和计算设备存放数据矩阵的能力
  size_t max_sta_num = EstimateGpuBatch(gpu_id, nspec, nstep, gpu_task_num);
  printf("src count is %d, sta count is %d\n", pSrcPaths->count, pStaPaths->count);
  max_sta_num = min(max_sta_num, pSrcPaths->count);
  max_sta_num = min(max_sta_num, pStaPaths->count);
  max_sta_num /= 2;
  max_sta_num = max_sta_num > 0 ? max_sta_num : 1;
  size_t pair_batch = max_sta_num * max_sta_num;

  // 管理 每一批数据处理哪些 源-台 对
  int src_group_count = (pSrcPaths->count + max_sta_num - 1) / max_sta_num;
  int sta_group_count = (pStaPaths->count + max_sta_num - 1) / max_sta_num;
  int num_managers = src_group_count * sta_group_count;
  PAIRLIST_MANAGER *managers = NULL;

  int MULTI_ARRAY_FLAG = 0;
  CpuMalloc((void **)&managers, num_managers * sizeof(PAIRLIST_MANAGER)); // 存储"列表"对的列表

  size_t SINGLE_ARRAY_FLAG = strcmp(argument.src_lst_path, argument.sta_lst_path);
  int manager_idx = 0;
  for (int src_group_idx = 0; src_group_idx < src_group_count; src_group_idx++)
  {
    int sta_group_idx_start = 0;
    if (SINGLE_ARRAY_FLAG == 0)
    {
      sta_group_idx_start = src_group_idx;
    }
    for (int sta_group_idx = sta_group_idx_start; sta_group_idx < sta_group_count; sta_group_idx++)
    {
      int src_start = src_group_idx * max_sta_num;
      int src_end = (src_start + max_sta_num > pSrcPaths->count) ? pSrcPaths->count : src_start + max_sta_num;
      int sta_start = sta_group_idx * max_sta_num;
      int sta_end = (sta_start + max_sta_num > pStaPaths->count) ? pStaPaths->count : sta_start + max_sta_num;
      // 初始化pairlist_manager
      PAIRLIST_MANAGER *manager = &managers[manager_idx];
      manager->src_start_idx = src_start; // 这个参数用来控制读取源文件从哪里开始
      manager->src_end_idx = src_end;     // 这个参数用来控制读取源文件到哪里结束
      manager->sta_start_idx = sta_start; // 这个参数用来控制读取台文件从哪里开始
      manager->sta_end_idx = sta_end;     // 这个参数用来控制读取台文件到哪里结束

      CpuMalloc((void **)&(manager->src_idx_list), pair_batch * sizeof(size_t));
      CpuMalloc((void **)&(manager->sta_idx_list), pair_batch * sizeof(size_t));

      // 初始化单一阵列标志
      if (strcmp(argument.src_lst_path, argument.sta_lst_path) == 0 &&
          src_start == sta_start && src_end == sta_end)
      {
        manager->single_array_flag = 1; // 判断为单一台阵
      }
      else
      {
        manager->single_array_flag = 0;
        MULTI_ARRAY_FLAG = 1; // 只要存在一个双台阵，双台阵标志设置为1
      }

      size_t index = 0;
      for (size_t src_idx = src_start; src_idx < src_end; src_idx++)
      {
        size_t sta_start_tmp = manager->single_array_flag ? src_idx : sta_start;
        for (size_t sta_idx = sta_start_tmp; sta_idx < sta_end; sta_idx++)
        {
          manager->src_idx_list[index] = src_idx - src_start;
          manager->sta_idx_list[index] = sta_idx - sta_start;
          index++;
        }
      }

      manager->node_count = index;
      size_t node_count = manager->node_count;
      // 为结构体中的每个指针类型成员分配内存
      CpuMalloc((void **)&(manager->stla_list), node_count * sizeof(float));
      CpuMalloc((void **)&(manager->stlo_list), node_count * sizeof(float));
      CpuMalloc((void **)&(manager->evla_list), node_count * sizeof(float));
      CpuMalloc((void **)&(manager->evlo_list), node_count * sizeof(float));
      CpuMalloc((void **)&(manager->Gcarc_list), node_count * sizeof(float));
      CpuMalloc((void **)&(manager->Az_list), node_count * sizeof(float));
      CpuMalloc((void **)&(manager->Baz_list), node_count * sizeof(float));
      CpuMalloc((void **)&(manager->Dist_list), node_count * sizeof(float));
      CpuMalloc((void **)&(manager->ok_flag_list), node_count * sizeof(int));

      // 初始化 OK_lists
      for (size_t i = 0; i < node_count; i++)
      {
        manager->ok_flag_list[i] = 0;
      }
      manager_idx++;
    }
  }
  num_managers = manager_idx;
  ThreadPoolFilter *filterpool = create_threadpool_filter_nodes(cpu_count);
  for (int idx = 0; idx < num_managers; idx++)
  {
    FilterNodeParallel(&managers[idx], pSrcPaths, pStaPaths, filterpool, max_distance);
  }
  destroy_threadpool_filter_nodes(filterpool);

  for (int idx = 0; idx < num_managers; idx++)
  {
    CompressManager(&managers[idx]);
  }

  // 为数据分配空间
  size_t vec_cnt = nstep * nspec;              // 一个频谱文件中的点数
  size_t vec_size = vec_cnt * sizeof(complex); // 一个频谱文件的数据块的大小

  complex *h_src_buffer = NULL;   // 输入的虚拟源的频谱
  cuComplex *d_src_buffer = NULL; // 输入虚拟源数据
  CpuMalloc((void **)&h_src_buffer, max_sta_num * vec_size);
  GpuMalloc((void **)&d_src_buffer, max_sta_num * vec_size);

  complex *h_sta_buffer = NULL;   // 输入的虚拟接收机的频谱
  cuComplex *d_sta_buffer = NULL; // 输出虚拟台数据
  if (MULTI_ARRAY_FLAG == 1)      // 一旦有多台阵情形,为sta_buffer分配空间
  {
    CpuMalloc((void **)&h_sta_buffer, max_sta_num * vec_size);
    GpuMalloc((void **)&d_sta_buffer, max_sta_num * vec_size);
  }

  float *h_ncf_time = NULL; // 写出互相关函数缓冲区
  CpuMalloc((void **)&h_ncf_time, pair_batch * ncc * sizeof(float));

  // 为计算设备端的数据分配空间
  dim3 dimGrid_1D, dimBlock_1D;
  dim3 dimGrid_2D, dimBlock_2D;
  cufftHandle plan; // 创建从互相关频谱到互相关事件序列的IFFT计划
  int rank = 1;
  int n[1] = {nfft};
  int inembed[1] = {nfft};
  int onembed[1] = {nfft};
  int istride = 1;
  int idist = nfft;
  int ostride = 1;
  int odist = nfft;
  cufftType type = CUFFT_C2R;
  CufftPlanAlloc(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, pair_batch); // 为FFT计划分配空间

  size_t *d_src_idx_list = NULL;  // 输入台站对节点的源在批数据中的索引
  size_t *d_sta_idx_list = NULL;  // 输入台站对节点的台在批数据中的索引
  cuComplex *d_ncf_buffer = NULL; // 输出的NCF频谱, 规模为paircnt*nstep*nfft
  cuComplex *d_ncf_stack = NULL;  // 输出的NCF频谱, 按nstep叠加, 规模为paircnt*nfft
  int *d_sgn_vec = NULL;          // 储存相移向量, 保证频域NCF转变到时域后0时刻在序列中心
  float *d_ncf_time = NULL;       // IFFT 之后的NCF序列, 长度为nfft

  GpuMalloc((void **)&d_src_idx_list, pair_batch * sizeof(size_t));
  GpuMalloc((void **)&d_sta_idx_list, pair_batch * sizeof(size_t));
  GpuMalloc((void **)&d_ncf_buffer, pair_batch * vec_size);
  GpuMalloc((void **)&d_ncf_stack, pair_batch * nfft * sizeof(cuComplex));
  GpuMalloc((void **)&d_ncf_time, pair_batch * nfft * sizeof(float));

  // 计算相移向量
  GpuCalloc((void **)&d_sgn_vec, nspec * sizeof(int));
  DimCompute1D(&dimGrid_1D, &dimBlock_1D, nspec);
  generateSignVector<<<dimGrid_1D, dimBlock_1D>>>(d_sgn_vec, nspec);

  ThreadPoolRead *read_pool = create_threadpool_read(cpu_count);
  ThreadWritePool *write_pool = create_threadwrite_pool(cpu_count);
  size_t src_start_flag = 0;
  size_t src_end_flag = 0;
  size_t sta_start_flag = 0;
  size_t sta_end_flag = 0;
  for (size_t manager_idx = 0; manager_idx < num_managers; manager_idx++)
  {
    PAIRLIST_MANAGER *manager = &managers[manager_idx];
    size_t src_start = manager->src_start_idx;
    size_t src_end = manager->src_end_idx;
    size_t sta_start = manager->sta_start_idx;
    size_t sta_end = manager->sta_end_idx;
    size_t src_trace_num = src_end - src_start;
    size_t sta_trace_num = sta_end - sta_start;
    size_t node_count = manager->node_count; // 每个节点存储一对 源-台 索引

    memset(h_ncf_time, 0, pair_batch * ncc * sizeof(float));                      // 清空主机端输出数据
    CUDACHECK(cudaMemset(d_ncf_stack, 0, pair_batch * nfft * sizeof(cuComplex))); // 清空设备端输出数据
    CUDACHECK(cudaMemset(d_ncf_time, 0, pair_batch * nfft * sizeof(float)));      // 清空设备端输出数据
    CUDACHECK(cudaMemcpy(d_src_idx_list, manager->src_idx_list, node_count * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_sta_idx_list, manager->sta_idx_list, node_count * sizeof(size_t), cudaMemcpyHostToDevice));
    DimCompute(&dimGrid_2D, &dimBlock_2D, vec_cnt, node_count);

    // 只在数据变化时读取源和台站数据
    if (src_start != src_start_flag || src_end != src_end_flag)
    {
      ReadSpecArrayParallel(pSrcPaths, h_src_buffer, src_start, src_end, vec_cnt, read_pool);
      CUDACHECK(cudaMemcpy(d_src_buffer, h_src_buffer, src_trace_num * vec_size, cudaMemcpyHostToDevice));
      src_start_flag = src_start;
      src_end_flag = src_end;
    }
    if (manager->single_array_flag == 0) // 双台阵情形
    {
      if (sta_start != sta_start_flag || sta_end != sta_end_flag)
      {
        ReadSpecArrayParallel(pStaPaths, h_sta_buffer, sta_start, sta_end, vec_cnt, read_pool);
        CUDACHECK(cudaMemcpy(d_sta_buffer, h_sta_buffer, sta_trace_num * vec_size, cudaMemcpyHostToDevice));
        sta_start_flag = sta_start;
        sta_end_flag = sta_end;
      }
      cmultiply2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_src_buffer, d_src_idx_list,
                                                     d_sta_buffer, d_sta_idx_list,
                                                     d_ncf_buffer, node_count, vec_cnt);
    }
    else // 单台阵情形，只需读取源数据
    {
      cmultiply2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_src_buffer, d_src_idx_list,
                                                     d_src_buffer, d_sta_idx_list,
                                                     d_ncf_buffer, node_count, vec_cnt);
    }

    // // 累加各个 step 的计算结果
    DimCompute(&dimGrid_2D, &dimBlock_2D, nspec, node_count); // 需要修改网格大小
    for (size_t step_idx = 0; step_idx < nstep; step_idx++)
    {
      csum2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_ncf_stack, nfft, d_ncf_buffer, nspec,
                                                nspec, node_count, step_idx, nstep);
    }
    DimCompute(&dimGrid_2D, &dimBlock_2D, nspec, node_count); // 再次重新计算网格
    applyPhaseShiftKernel<<<dimGrid_2D, dimBlock_2D>>>(d_ncf_stack, d_sgn_vec, nfft, nspec, node_count);

    cufftExecC2R(plan, (cufftComplex *)d_ncf_stack, (cufftReal *)d_ncf_time);
    DimCompute(&dimGrid_2D, &dimBlock_2D, nfft, node_count);
    InvNormalize2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_ncf_time, nfft, nfft, node_count, delta);

    CUDACHECK(cudaMemcpy2D(h_ncf_time, ncc * sizeof(float),
                           d_ncf_time + (nspec - nhalfcc - 1), nfft * sizeof(float),
                           ncc * sizeof(float), node_count, cudaMemcpyDeviceToHost));

    /* Write out ncf */
    write_pairs_parallel(manager, pSrcPaths, pStaPaths,
                         h_ncf_time, delta, ncc, cclength,
                         ncf_dir, write_pool);
  }
  destroy_threadpool_read(read_pool);
  destroy_threadwrite_pool(write_pool);
  /* Free cpu memory */
  for (int manager_idx = 0; manager_idx < num_managers; manager_idx++)
  {
    free(managers[manager_idx].src_idx_list);
    free(managers[manager_idx].sta_idx_list);
    free(managers[manager_idx].stla_list);
    free(managers[manager_idx].stlo_list);
    free(managers[manager_idx].evla_list);
    free(managers[manager_idx].evlo_list);
    free(managers[manager_idx].Gcarc_list);
    free(managers[manager_idx].Az_list);
    free(managers[manager_idx].Baz_list);
    free(managers[manager_idx].Dist_list);
    free(managers[manager_idx].ok_flag_list);
  }
  free(managers);
  CpuFree((void **)&h_ncf_time);
  CpuFree((void **)&h_src_buffer);
  CpuFree((void **)&h_sta_buffer);
  freeFilePaths(pSrcPaths);
  freeFilePaths(pStaPaths);

  // Free gpu memory
  GpuFree((void **)&d_src_buffer);
  GpuFree((void **)&d_sta_buffer);
  GpuFree((void **)&d_src_idx_list);
  GpuFree((void **)&d_sta_idx_list);
  GpuFree((void **)&d_ncf_buffer);
  GpuFree((void **)&d_ncf_stack);
  GpuFree((void **)&d_sgn_vec);
  GpuFree((void **)&d_ncf_time);
  CUFFTCHECK(cufftDestroy(plan));

  return 0;
}
