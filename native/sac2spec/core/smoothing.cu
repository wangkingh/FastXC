#include "smoothing.cuh"

#include "../kernels/real_matrix.cuh"

void launch_smooth_rows(float *d_out, int dpitch, const float *d_in,
                        int spitch, int width, int height, int winsize)
{
  if (width <= 0 || height <= 0 || winsize <= 0)
  {
    return;
  }

  /* FastXC smoothing windows are normally large, so always use the rolling
   * row kernel instead of keeping a small-window elementwise branch.
   */
  const int block_size = 256;
  dim3 block(block_size);
  dim3 grid((height + block_size - 1) / block_size);
  smoothRowsRollingKernel<<<grid, block>>>(d_out, dpitch, d_in, spitch,
                                           width, height, winsize);
}
