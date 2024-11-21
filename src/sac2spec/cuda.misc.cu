#include "cuda.misc.cuh"
#include <cstddef>

/* Define Kernel Fucntions */
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
    if (d_data[idx] < minval)
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

__global__ void cisnan2DKernel(cuComplex *d_data, size_t pitch, size_t width,
                               size_t height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int idx = row * pitch + col;
    if (isnan(d_data[idx].x) || isinf(d_data[idx].x) || isnan(d_data[idx].y) ||
        isinf(d_data[idx].y))
    {
      d_data[idx].x = 0;
      d_data[idx].y = 0;
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
    d_data[didx] /= d_divisor[sidx];
  }
}

__global__ void cdiv2DKernel(cuComplex *d_data, size_t dpitch, float *d_divisor,
                             size_t spitch, size_t width, size_t height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int didx = row * dpitch + col;
    int sidx = row * spitch + col;
    d_data[didx].x /= d_divisor[sidx];
    d_data[didx].y /= d_divisor[sidx];
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
    d_data_out[didx] = d_data_out[sidx] + d_data_in[sidx];
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

__global__ void amp2DKernel(float *d_amp, size_t dpitch,
                            cuComplex *d_data, size_t spitch,
                            size_t width, size_t height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    if (col == 0)
    {
      d_amp[row * dpitch] = fabs(cuCrealf(d_data[row * spitch]));
      d_amp[row * dpitch + width] = fabs(cuCimagf(d_data[row * spitch]));
    }
    cuComplex c = d_data[row * spitch + col];
    d_amp[row * dpitch + col] = cuCabsf(c);
  }
}

__global__ void filterKernel(cuComplex *d_spectrum, cuComplex *d_response, size_t pitch, size_t width, size_t height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int idx = row * pitch + col;

    // filter the first time
    d_spectrum[idx] = cuCmulf(d_spectrum[idx], d_response[col]);

    // doing conjugate, to reverse the time direction
    d_spectrum[idx] = cuConjf(d_spectrum[idx]);

    // filter the second time
    d_spectrum[idx] = cuCmulf(d_spectrum[idx], d_response[col]);

    // doing conjugate, to reverse the time direction again
    d_spectrum[idx] = cuConjf(d_spectrum[idx]);
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

__global__ void FwdNormalize2DKernel(cuComplex *d_segspec, size_t pitch,
                                     size_t width, size_t height, float dt)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  double weight = dt;
  if (row < height && col < width)
  {
    size_t idx = row * pitch + col;
    d_segspec[idx].x *= weight;
    d_segspec[idx].y *= weight;
  }
}

// IFFT Nomalize add by wangjx@2023-05-21
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

__global__ void smooth2DKernel(float *d_out, int dpitch, float *d_tmp,
                               int spitch, int width, int height, int winsize)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  double weight = 1.0 / winsize;
  if (col < width && row < height)
  {
    int idx = row * dpitch + col;
    double val = 0;
    int nstart = col - winsize / 2;

    for (int i = 0; i < winsize; i++)
    {
      if (nstart + i >= 0 && nstart + i < width)
      {
        val += d_tmp[row * spitch + nstart + i];
      }
    }

    float smooth_val = weight * val;
    d_out[idx] = smooth_val;
  }
}
