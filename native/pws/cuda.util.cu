#include "cuda.util.cuh"

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

size_t EstimateCufftWorkspace1D(size_t nsamples, size_t batch, cufftType type)
{
  int rank = 1;
  int n[1] = {(int)nsamples};
  int embed[1] = {(int)nsamples};
  int stride = 1;
  int dist = (int)nsamples;
  size_t workspace = 0;

  CUFFTCHECK(cufftEstimateMany(rank, n,
                               embed, stride, dist,
                               embed, stride, dist,
                               type, (int)batch,
                               &workspace));
  return workspace;
}

size_t QueryGpuFreeBytes(int gpu_id)
{
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDACHECK(cudaSetDevice(gpu_id));
  CUDACHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return free_bytes;
}

void CudaCheckLastKernel(const char *name)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG_ERROR("kernel_launch_failed",
              "kernel=\"%s\" error=\"%s\"",
              name,
              cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void GpuFree(void **pptr)
{
  if (*pptr != NULL)
  {
    cudaFree(*pptr);
    *pptr = NULL;
  }
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
