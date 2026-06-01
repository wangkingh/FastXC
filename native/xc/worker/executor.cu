#include "executor.hpp"

#include "cuda.util.cuh"
#include "cuda.kernels.cuh"

void finalize_xc_batch(GpuBuffers *buf,
                       const RuntimeShape *shape,
                       size_t task_count)
{
  dim3 grid, block;
  DimCompute(&grid, &block, (size_t)shape->nspec, task_count);
  applyPhaseShiftKernel<<<grid, block>>>(buf->d_stack,
                                         buf->d_sign,
                                         (size_t)shape->nspec,
                                         (size_t)shape->nspec,
                                         task_count);
  CUDACHECK(cudaGetLastError());

  if (task_count < buf->pair_capacity)
  {
    const size_t used = task_count * (size_t)shape->nspec;
    const size_t tail = (buf->pair_capacity - task_count) * (size_t)shape->nspec;
    CUDACHECK(cudaMemset(buf->d_stack + used, 0, tail * sizeof(cuComplex)));
  }

  CUFFTCHECK(cufftExecC2R(buf->plan,
                          (cufftComplex *)buf->d_stack,
                          (cufftReal *)buf->d_time));

  DimCompute(&grid, &block, (size_t)shape->nfft, task_count);
  InvNormalize2DKernel<<<grid, block>>>(buf->d_time,
                                        (size_t)shape->nfft,
                                        (size_t)shape->nfft,
                                        task_count,
                                        shape->dt);
  CUDACHECK(cudaGetLastError());

  CUDACHECK(cudaMemcpy2D(buf->h_cc,
                         (size_t)shape->cc_size * sizeof(float),
                         buf->d_time + (shape->nspec - shape->half_cc - 1),
                         (size_t)shape->nfft * sizeof(float),
                         (size_t)shape->cc_size * sizeof(float),
                         task_count,
                         cudaMemcpyDeviceToHost));
}
