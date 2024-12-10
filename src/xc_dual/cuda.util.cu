#include "cuda.util.cuh"
#include <unistd.h>

const float RAMUPPERBOUND = 0.9;

// DimCompute: BLOCKX = 32, BLOCKY = 32
void DimCompute(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
{
  pdimblk->x = BLOCKX;
  pdimblk->y = BLOCKY;

  // for debug, trying to limit dimgrd
  pdimgrd->x = (width + BLOCKX - 1) / BLOCKX;
  pdimgrd->y = (height + BLOCKY - 1) / BLOCKY;
}

size_t QueryAvailGpuRam(size_t deviceID)
{
  size_t freeram, totalram;
  cudaSetDevice(deviceID);
  CUDACHECK(cudaMemGetInfo(&freeram, &totalram));
  freeram *= RAMUPPERBOUND;

  const size_t gigabytes = 1L << 30;
  printf("Avail gpu ram: %.3f GB\n", freeram * 1.0 / gigabytes);
  return freeram;
}

void DimCompute1D(dim3 *pdimgrd, dim3 *pdimblk, size_t width)
{
  // 设置线程块大小，假设 BLOCKX1D 是预定义的每个线程块的线程数
  pdimblk->x = BLOCKX1D;

  // 计算所需的网格大小，确保能够覆盖所有的元素
  pdimgrd->x = (width + BLOCKX1D - 1) / BLOCKX1D;
}

void DimCompute2D(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
{
  pdimblk->x = BLOCKX2D;
  pdimblk->y = BLOCKY2D;

  pdimgrd->x = (width + BLOCKX2D - 1) / BLOCKX2D;
  pdimgrd->y = (height + BLOCKY2D - 1) / BLOCKY2D;
}

void DimCompute3D(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height, size_t depth)
{
  // 设置每个维度的块大小
  pdimblk->x = BLOCKX3D;
  pdimblk->y = BLOCKY3D;
  pdimblk->z = BLOCKZ3D;

  // 计算每个维度所需的网格大小
  pdimgrd->x = (width + BLOCKX3D - 1) / BLOCKX3D;  // 计算X维度的网格数
  pdimgrd->y = (height + BLOCKY3D - 1) / BLOCKY3D; // 计算Y维度的网格数
  pdimgrd->z = (depth + BLOCKZ3D - 1) / BLOCKZ3D;  // 计算Z维度的网格数
}

void GpuFree(void **pptr)
{
  if (*pptr != NULL)
  {
    cudaFree(*pptr);
    *pptr = NULL;
  }
}

size_t EstimateGpuBatch_CC(size_t gpu_id, size_t fiexed_ram, size_t unitram,
                           int numType, int rank, int *n, int *inembed,
                           int istride, int idist, int *onembed, int ostride,
                           int odist, cufftType *typeArr)
{
  size_t d_batch = 0;
  size_t availram = QueryAvailGpuRam(gpu_id);
  size_t reqram = fiexed_ram;
  if (reqram > availram)
  {
    fprintf(stderr, "Not enough gpu ram required:%lu, gpu remain ram: %lu\n",
            reqram, availram);
    exit(1);
  }
  size_t step = 360; // 没有特殊情况下，步长为360，因为我喜欢这个数字
  size_t last_valid_batch = 0;
  while (reqram < availram)
  {
    d_batch += step;
    size_t tmp_reqram = reqram;
    for (int i = 0; i < numType; i++)
    {
      size_t tmpram = 0;
      cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride,
                        odist, typeArr[i], d_batch, &tmpram);
      tmp_reqram += tmpram;
    }
    tmp_reqram += d_batch * unitram;
    if (tmp_reqram > availram)
    {
      d_batch -= step; // 回退到上一个有效的批次
      if (step == 1)
      {
        break; // 如果步长已经是1，进一步减少会变成0，因此在此退出
      }
      step /= 2; // 减小步长
    }
    else
    {
      last_valid_batch = d_batch;
      reqram = tmp_reqram;
      step = (step == 0) ? 1 : step * 2; // 指数增长步长
    }
  }
  return last_valid_batch;
}

size_t EstimateGpuBatch_TFPWS(size_t gpu_id, int nfft, int nfreq)
{
  size_t d_batch = 0;
  size_t availram = QueryAvailGpuRam(gpu_id);
  size_t reqram = 0;

  int rank_hilb = 1;
  int n_hilb[1] = {(int)nfft};
  int inembed_hilb[1] = {(int)nfft};
  int onembed_hilb[1] = {(int)nfft};
  int istride_hilb = 1;
  int idist_hilb = (int)nfft;
  int ostride_hilb = 1;
  int odist_hilb = (int)nfft;

  size_t unitram = (nfreq + 1) * nfft * sizeof(cuComplex); // 调制后的数据和临时频谱

  size_t tmpram = 0;
  cufftEstimateMany(rank_hilb, n_hilb, inembed_hilb, istride_hilb, idist_hilb, onembed_hilb, ostride_hilb,
                    odist_hilb, CUFFT_R2C, 1, &tmpram); // FFT Memory [叠后数据]频域希尔伯特变换
  reqram += tmpram;

  tmpram = 0;
  cufftEstimateMany(rank_hilb, n_hilb, inembed_hilb, istride_hilb, idist_hilb, onembed_hilb, ostride_hilb,
                    odist_hilb, CUFFT_C2C, 1, &tmpram); // 将数据从[叠后]频谱变换为解析信号
  reqram += tmpram;
  
  tmpram = 0;
  cufftEstimateMany(rank_hilb, n_hilb, inembed_hilb, istride_hilb, idist_hilb, onembed_hilb, ostride_hilb,
                    odist_hilb, CUFFT_C2C, nfreq, &tmpram); // [叠后]调制频谱变换为[叠后]调制解析信号
  reqram += tmpram;

  if (reqram > availram)
  {
    fprintf(stderr, "Not enough gpu ram required:%lu, gpu remain ram: %lu\n",
            reqram, availram);
    exit(1);
  }
  size_t step = 360; // 没有特殊情况下，步长为360，因为我喜欢这个数字
  size_t last_valid_batch = 0;

  while (reqram < availram)
  {
    d_batch += step;
    size_t tmp_reqram = reqram;

    tmpram = 0;
    cufftEstimateMany(rank_hilb, n_hilb, inembed_hilb, istride_hilb, idist_hilb, onembed_hilb, ostride_hilb,
                      odist_hilb, CUFFT_C2C, d_batch * nfreq, &tmpram); // FFT Memory 从频谱变换到解析信号
    tmp_reqram += tmpram;

    tmp_reqram += d_batch * unitram;
    if (tmp_reqram > availram)
    {
      d_batch -= step; // 回退到上一个有效的批次
      if (step == 1)
      {
        break; // 如果步长已经是1，进一步减少会变成0，因此在此退出
      }
      step /= 2; // 减小步长
    }
    else
    {
      last_valid_batch = d_batch;
      reqram = tmp_reqram;
      step = (step == 0) ? 1 : step * 2; // 指数增长步长
    }
  }
  if (last_valid_batch == 0)
  {
    fprintf(stderr, "Not enough gpu ram required:%lu, gpu remain ram: %lu\n",
            reqram, availram);
    exit(1);
  }
  return last_valid_batch;
}

void CufftPlanAlloc(cufftHandle *pHandle, int rank, int *n, int *inembed,
                    int istride, int idist, int *onembed, int ostride,
                    int odist, cufftType type, int batch)
{
  // create cufft plan
  CUFFTCHECK(cufftPlanMany(pHandle, rank, n, inembed, istride, idist, onembed,
                           ostride, odist, type, batch));
}

void GpuMalloc(void **pptr, size_t sz) { CUDACHECK(cudaMalloc(pptr, sz)); }

void GpuCalloc(void **pptr, size_t sz)
{
  CUDACHECK(cudaMalloc(pptr, sz));

  CUDACHECK(cudaMemset(*pptr, 0, sz));
}