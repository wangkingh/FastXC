#include "complex_matrix.cuh"

#include <math.h>

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

__global__ void cdiv2DKernel(cuComplex *d_data, size_t dpitch, float *d_divisor,
                             size_t spitch, size_t width, size_t height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int didx = row * dpitch + col;
    int sidx = row * spitch + col;
    float divisor = d_divisor[sidx];
    if (!isfinite(divisor) || divisor < MINVAL)
    {
      d_data[didx].x = 0.0f;
      d_data[didx].y = 0.0f;
    }
    else
    {
      d_data[didx].x /= divisor;
      d_data[didx].y /= divisor;
    }
  }
}

__global__ void spectralOnebit2DKernel(cuComplex *d_data, size_t pitch, size_t width,
                                       size_t height, float minval)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    size_t idx = row * pitch + col;
    float real = d_data[idx].x;
    float imag = d_data[idx].y;
    float amp = hypotf(real, imag);

    if (!isfinite(amp) || amp <= minval)
    {
      d_data[idx].x = 0.0f;
      d_data[idx].y = 0.0f;
    }
    else
    {
      float inv_amp = 1.0f / amp;
      d_data[idx].x = real * inv_amp;
      d_data[idx].y = imag * inv_amp;
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

__global__ void filterKernel(cuComplex *d_spectrum, const float *d_response, size_t pitch, size_t width, size_t height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int idx = row * pitch + col;
    float response_power = d_response[col];
    d_spectrum[idx].x *= response_power;
    d_spectrum[idx].y *= response_power;
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
