#include "cuda.xc_dual.cuh"

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

__global__ void cmultiply2DKernel(cuComplex *d_src_buffer, size_t *src_idx_list,
                                  cuComplex *d_sta_buffer, size_t *sta_idx_list,
                                  cuComplex *d_ncf_buffer, size_t height, size_t width)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    size_t ncf_idx = row * width + col;
    size_t srcidx = src_idx_list[row] * width + col;
    size_t staidx = sta_idx_list[row] * width + col;

    cuComplex sta = d_sta_buffer[staidx];
    cuComplex src_conj = make_cuComplex(d_src_buffer[srcidx].x, -d_src_buffer[srcidx].y);

    cuComplex mul_result = cuCmulf(src_conj, sta);
    d_ncf_buffer[ncf_idx].x = mul_result.x;
    d_ncf_buffer[ncf_idx].y = mul_result.y;
  }
}

__global__ void csum2DKernel(cuComplex *d_total_spectrum, int dpitch,
                             cuComplex *d_segment_spectrum, int spitch,
                             size_t width, size_t height, int step_idx, int nstep)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int sidx = row * spitch * nstep + step_idx * spitch + col;
    int didx = row * dpitch + col;
    cuComplex temp = d_segment_spectrum[sidx];
    temp.x /= nstep; // divide the real part by nstep
    temp.y /= nstep; // divide the imaginary part by nstep

    d_total_spectrum[didx] = cuCaddf(d_total_spectrum[didx], temp);
  }
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

// sum2dKernel is used to sum the 2D array of float, not used in the current version
__global__ void sum2DKernel(float *d_finalccvec, int dpitch, float *d_segncfvec,
                            int spitch, size_t width, size_t height,
                            int nstep)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int sidx = row * spitch + col;
    int didx = row * dpitch + col;
    d_finalccvec[didx] += (d_segncfvec[sidx] / nstep);
  }
}

__global__ void csum2DKernel(cuComplex *d_total_spectrum, int dpitch,
                             cuComplex *d_segment_spectrum, int spitch,
                             size_t width, size_t height, int nstep)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int sidx = row * spitch + col;
    int didx = row * dpitch + col;
    cuComplex temp = d_segment_spectrum[sidx];
    temp.x /= nstep; // divide the real part by nstep
    temp.y /= nstep; // divide the imaginary part by nstep

    d_total_spectrum[didx] = cuCaddf(d_total_spectrum[didx], temp);
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
