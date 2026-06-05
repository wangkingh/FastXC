#include "real_matrix.cuh"

#include <math.h>

/* Time-domain callers currently pass compact rows, so pitch == width.
 * Keep both arguments explicit because this file follows the generic 2-D
 * matrix convention also used by padded FFT/spectrum buffers, where pitch can
 * be larger than the active width.
 */
__global__ void abs2DKernel(float *d_data, size_t pitch, size_t width,
                            size_t height)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    size_t idx = row * pitch + col;
    d_data[idx] = fabs(d_data[idx]);
  }
}

__global__ void clampmin2DKernel(float *d_data, size_t pitch, size_t width,
                                 size_t height, float minval)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int idx = row * pitch + col;
    if (!isfinite(d_data[idx]) || d_data[idx] < minval)
    {
      d_data[idx] = minval;
    }
  }
}

__global__ void isnan2DKernel(float *d_data, size_t pitch, size_t width,
                              size_t height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width && row < height)
  {
    int idx = row * pitch + col;
    if (isnan(d_data[idx]) || isinf(d_data[idx]))
    {
      d_data[idx] = 0;
    }
  }
}

__global__ void div2DKernel(float *d_data, size_t dpitch, float *d_divisor,
                            size_t spitch, size_t width, size_t height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int sidx = row * spitch + col;
    int didx = row * dpitch + col;
    float divisor = d_divisor[sidx];
    if (!isfinite(divisor) || divisor < MINVAL)
    {
      d_data[didx] = 0.0f;
    }
    else
    {
      d_data[didx] /= divisor;
    }
  }
}

__global__ void sum2DKernel(float *d_data_out, size_t dpitch, float *d_data_in,
                            size_t spitch, size_t width, size_t height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int didx = row * dpitch + col;
    int sidx = row * spitch + col;
    d_data_out[didx] = d_data_out[didx] + d_data_in[sidx];
  }
}

__global__ void expandSharedWeight2DKernel(float *d_weight_full, size_t full_pitch,
                                           const float *d_weight_shared, size_t shared_pitch,
                                           size_t width, size_t height, int num_ch)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    size_t shared_row = row / (size_t)num_ch;
    d_weight_full[row * full_pitch + col] = d_weight_shared[shared_row * shared_pitch + col];
  }
}

__global__ void cutmax2DKernel(float *d_data, size_t pitch, size_t width,
                               size_t height, float maxval)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int idx = row * pitch + col;

    float val = d_data[idx];

    if (val > maxval)
    {
      d_data[idx] = maxval;
    }
    else if (val < -1 * maxval)
    {
      d_data[idx] = -1 * maxval;
    }
  }
}

__global__ void onebit2DKernel(float *d_data, size_t pitch, size_t width,
                               size_t height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int idx = row * pitch + col;
    d_data[idx] = (d_data[idx] > 0.0f)   ? 1.0f
                  : (d_data[idx] < 0.0f) ? -1.0f
                                         : 0.0f;
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

__global__ void smoothRowsRollingKernel(float *d_out, int dpitch, const float *d_tmp,
                                        int spitch, int width, int height, int winsize)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= height || width <= 0 || winsize <= 0)
  {
    return;
  }

  const float *src = d_tmp + (size_t)row * (size_t)spitch;
  float *dst = d_out + (size_t)row * (size_t)dpitch;
  int start = -(winsize / 2);
  int end = start + winsize - 1;
  int first_end = (end < width) ? end : width - 1;
  double val = 0.0;
  double weight = 1.0 / (double)winsize;

  for (int i = 0; i <= first_end; i++)
  {
    val += src[i];
  }

  for (int col = 0; col < width; col++)
  {
    dst[col] = (float)(weight * val);
    if (start >= 0 && start < width)
    {
      val -= src[start];
    }
    start++;
    end++;
    if (end >= 0 && end < width)
    {
      val += src[end];
    }
  }
}
