#include "cuda.rdcrtr.cuh"

__global__ void sumSingleBlock2DKernel(double *d_sum, int dpitch,
                                       const float *d_data, int spitch,
                                       int width, int height)
{
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t tx = threadIdx.x;
  size_t ty = threadIdx.y;

  double sum = 0;
  size_t i;
  for (i = tx; i < width && row < height; i += blockDim.x)
  {
    sum += d_data[row * spitch + i];
  }

  extern __shared__ double partial[];
  if (row < height && tx < blockDim.x)
  {
    partial[ty * blockDim.x + tx] = sum;
  }
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride >= 1; stride /= 2)
  {
    if (row < height && tx < stride)
    {
      partial[ty * blockDim.x + tx] += partial[ty * blockDim.x + tx + stride];
    }
    __syncthreads();
  }

  __syncthreads();

  if (tx == 0 && row < height)
  {
    d_sum[row * dpitch] = partial[ty * blockDim.x];
  }
}

__global__ void isumSingleBlock2DKernel(double *d_sum, int dpitch,
                                        const float *d_data, int spitch,
                                        int width, int height)
{
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t tx = threadIdx.x;
  size_t ty = threadIdx.y;

  double sum = 0;
  size_t i;
  for (i = tx; i < width && row < height; i += blockDim.x)
  {
    sum += d_data[row * spitch + i] * i;
  }

  extern __shared__ double partial[];
  if (row < height && tx < blockDim.x)
  {
    partial[ty * blockDim.x + tx] = sum;
  }
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride >= 1; stride /= 2)
  {
    if (row < height && tx < stride)
    {
      partial[ty * blockDim.x + tx] += partial[ty * blockDim.x + tx + stride];
    }
    __syncthreads();
  }

  __syncthreads();

  if (tx == 0 && row < height)
  {
    d_sum[row * dpitch] = partial[ty * blockDim.x];
  }
}

__global__ void rdc2DKernel(float *d_data, int pitch, int width, int height,
                            double *d_sum)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = row * pitch + col;
  if (col < width && row < height)
  {
    d_data[idx] -= d_sum[row] / width;
  }
}

__global__ void rtr2DKernel(float *d_data, int pitch, int width, int height,
                            double *d_sum, double *d_isum)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    /* SUM(i*x) */
    double y1 = d_isum[row];
    /* SUM(x) */
    double y2 = d_sum[row];

    double a, b, a11, a12, a22;
    a12 = 0.5 * width * (width - 1);
    a11 = a12 * (2 * width - 1) / 3.;
    a22 = width;

    b = a11 * a22 - a12 * a12;
    a = (a22 * y1 - a12 * y2) / b;
    b = (a11 * y2 - a12 * y1) / b;

    int idx = row * pitch + col;
    d_data[idx] -= a * col + b;
  }
}