#include "cuda.util.cuh"

void DimCompute1D(dim3 *pdimgrd, dim3 *pdimblk, size_t width)
{
  // 设置线程块大小，假设 BLOCKX1D 是预定义的每个线程块的线程数
  pdimblk->x = BLOCKX1D;

  // 计算所需的网格大小，确保能够覆盖所有的元素
  pdimgrd->x = (width + BLOCKX1D - 1) / BLOCKX1D;
}

void DimCompute(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
{
  pdimblk->x = BLOCKX;
  pdimblk->y = BLOCKY;

  pdimgrd->x = (width + BLOCKX - 1) / BLOCKX;
  pdimgrd->y = (height + BLOCKY - 1) / BLOCKY;
}

void GpuFree(void **pptr)
{
  CUDACHECK(cudaFree(*pptr));
  *pptr = NULL;
}
