#include "cuda.util.cuh"
#include <unistd.h>

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

void CufftPlanAlloc(cufftHandle *pHandle, int rank, int *n, int *inembed,
                    int istride, int idist, int *onembed, int ostride,
                    int odist, cufftType type, int batch)
{
  // create cufft plan
  CUFFTCHECK(cufftPlanMany(pHandle, rank, n, inembed, istride, idist, onembed,
                           ostride, odist, type, batch));
}

int CufftPlanQueryWorkSize(int rank, int *n, int *inembed,
                           int istride, int idist, int *onembed,
                           int ostride, int odist, cufftType type,
                           int batch, size_t *work_size)
{
  cufftHandle handle;
  cufftResult_t status = cufftCreate(&handle);
  if (status != CUFFT_SUCCESS)
    return (int)status;

  status = cufftSetAutoAllocation(handle, 0);
  if (status != CUFFT_SUCCESS)
  {
    cufftDestroy(handle);
    return (int)status;
  }

  size_t required = 0;
  status = cufftMakePlanMany(handle, rank, n, inembed, istride, idist, onembed,
                             ostride, odist, type, batch, &required);
  cufftDestroy(handle);
  if (status != CUFFT_SUCCESS)
    return (int)status;

  *work_size = required;
  return 0;
}

void CufftPlanAllocManual(cufftHandle *pHandle, int rank, int *n, int *inembed,
                          int istride, int idist, int *onembed, int ostride,
                          int odist, cufftType type, int batch,
                          void *work_area, size_t work_area_bytes,
                          size_t *required_work_size)
{
  size_t required = 0;
  CUFFTCHECK(cufftCreate(pHandle));
  CUFFTCHECK(cufftSetAutoAllocation(*pHandle, 0));
  CUFFTCHECK(cufftMakePlanMany(*pHandle, rank, n, inembed, istride, idist,
                               onembed, ostride, odist, type, batch,
                               &required));
  if (required > work_area_bytes)
  {
    LOG_ERROR("cufft_workspace_too_small",
              "required_mib=%.3f available_mib=%.3f",
              required / (1024.0 * 1024.0),
              work_area_bytes / (1024.0 * 1024.0));
    exit(EXIT_FAILURE);
  }
  if (required > 0)
    CUFFTCHECK(cufftSetWorkArea(*pHandle, work_area));
  if (required_work_size)
    *required_work_size = required;
}

void GpuMalloc(void **pptr, size_t sz) { CUDACHECK(cudaMalloc(pptr, sz)); }

void GpuCalloc(void **pptr, size_t sz)
{
  CUDACHECK(cudaMalloc(pptr, sz));

  CUDACHECK(cudaMemset(*pptr, 0, sz));
}
