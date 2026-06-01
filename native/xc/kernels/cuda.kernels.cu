#include "cuda.kernels.cuh"

__global__ void generateSignVector(int *sgn_vec, size_t width)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < width)
  {
    if (col == 0)
    {
      sgn_vec[col] = 0;
    }
    sgn_vec[col] = (col % 2 == 0) ? 1 : -1;
  }
}

__global__ void accumulateStepXc2DKernel(const cuComplex *__restrict__ d_spec_buffer,
                                         const size_t *__restrict__ src_idx_list,
                                         const size_t *__restrict__ rec_idx_list,
                                         cuComplex *__restrict__ d_stack,
                                         size_t nspec, float scale, size_t pair_count)
{
  size_t freq = blockIdx.x * blockDim.x + threadIdx.x;
  size_t pair = blockIdx.y * blockDim.y + threadIdx.y;
  if (freq >= nspec || pair >= pair_count)
    return;

  size_t src_idx = src_idx_list[pair] * nspec + freq;
  size_t rec_idx = rec_idx_list[pair] * nspec + freq;
  cuComplex src = d_spec_buffer[src_idx];
  cuComplex rec = d_spec_buffer[rec_idx];
  cuComplex src_conj = make_cuComplex(src.x, -src.y);
  cuComplex prod = cuCmulf(src_conj, rec);
  size_t dst_idx = pair * nspec + freq;
  d_stack[dst_idx].x += prod.x * scale;
  d_stack[dst_idx].y += prod.y * scale;
}

__global__ void applyPhaseShiftKernel(cuComplex *ncf_vec, int *sgn_vec, size_t spitch, size_t width, size_t height)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width && row < height)
  {
    size_t idx = row * spitch + col;
    int sign = sgn_vec[col]; // 使用一维向量
    ncf_vec[idx].x *= sign;
    ncf_vec[idx].y *= sign;
  }
}

__global__ void InvNormalize2DKernel(float *d_segdata, size_t pitch,
                                     size_t width, size_t height, float dt)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  double weight = 1.0 / (width * dt);
  if (row < height && col < width)
  {
    size_t idx = row * pitch + col;
    d_segdata[idx] *= weight;
  }
}
