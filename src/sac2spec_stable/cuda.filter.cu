#include "cuda.filter.cuh"
#include <cstddef>

__global__ void filterTKernel(float *d_sac, float *d_filtered, double *a, double *b, float *sac_hist, float *filtered_hist, size_t width, size_t height)
{
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int len_a = 5;
    int len_b = 5;
    int i = 0;
    if (row < height)
    {
        int offset_a = row * len_a;
        int offset_b = row * len_b;
        for (size_t col = 0; col < width; col++)
        {
            int idx = row * width + col;
            sac_hist[offset_a + 0] = d_sac[idx];

            // calculate the output point
            d_filtered[idx] = float(b[0] * sac_hist[offset_a + 0]);
            for (i = 1; i < 5; i++) // 1,2,3,4
            {
                d_filtered[idx] += float(b[i] * sac_hist[offset_a + i] - a[i] * filtered_hist[offset_b + i - 1]);
            }
            // update the history
            for (i = 4; i > 0; i--) // 4,3,2,1
            {
                sac_hist[offset_a + i] = sac_hist[offset_a + i - 1];
            }
            for (i = 4; i > 1; i--) // 4,3,2
            {
                filtered_hist[offset_b + i - 1] = filtered_hist[offset_b + i - 2];
            }
            filtered_hist[offset_b + 0] = d_filtered[idx];
        }
    }
}

__global__ void reverseKernel(const float *d_input, float *d_output, size_t width, size_t height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int index_in = row * width + col;
        int index_out = row * width + (width - 1 - col);
        d_output[index_out] = d_input[index_in];
    }
}
