#include "cuda.taper.cuh"

// The width of taper is nfft+1
__global__ void specTaper2DKernel(cuComplex *d_segspec, size_t pitch,
                                  size_t width, size_t height, int np,
                                  int idx1, int idx2, int idx3, int idx4)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  // Ensure that the index does not exceed th Nyquist frequency
  idx4 = idx4 < width ? idx4 : width;

  double dom, factor;
  int ntrans, j;

  if (col < width && row < height)
  {
    size_t idx = row * pitch + col;

    // we zero DC and nyquist freq up to f1
    if (col < idx1)
    {
      d_segspec[idx].x = 0.0;
      d_segspec[idx].y = 0.0;
    }
    // left low freq
    else if (col >= idx1 && col < idx2)
    {
      ntrans = idx2 - idx1;
      dom = M_PI / ntrans;

      factor = 1.0;
      for (j = 0; j < np; j++)
      {
        factor = factor * (1.0 - cos(dom * (col - idx1))) / 2.0;
      }
      d_segspec[idx].x = d_segspec[idx].x * factor;
      d_segspec[idx].y = d_segspec[idx].y * factor;
    }

    // idx2 to idx3 is flat

    // right high freq
    else if (col >= idx3 && col < idx4)
    {
      ntrans = idx4 - idx3;
      dom = M_PI / ntrans;

      factor = 1.0;
      for (j = 0; j < np; j++)
      {
        factor = factor * (1.0 + cos(dom * (col - idx3))) / 2.0;
      }
      d_segspec[idx].x = d_segspec[idx].x * factor;
      d_segspec[idx].y = d_segspec[idx].y * factor;
    }
    // higher freq are zero
    else if (col >= idx4 && col < width)
    {
      d_segspec[idx].x = 0.0;
      d_segspec[idx].y = 0.0;
    }
  }
}

__global__ void timetaper2DKernel(float *d_data, int pitch, int width, int height, int taper_size)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    size_t idx = row * pitch + col;

    // Apply the taper to the beginning and end of the signal.
    float factor = 1.0;
    if (col < taper_size)
    {
      factor = 0.5 * (1.0 - cos(M_PI * col / taper_size));
    }
    else if (col >= width - taper_size)
    {
      factor = 0.5 * (1.0 - cos(M_PI * (width - col) / taper_size));
    }
    else
    {
      factor = 1.0;
    }
    // Apply the taper to the signal.
    d_data[idx] *= factor;
  }
}
