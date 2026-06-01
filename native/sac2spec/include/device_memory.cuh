#ifndef _DEVICE_MEMORY_CUH
#define _DEVICE_MEMORY_CUH

#include "cuda.util.cuh"

/* Estimate the largest per-worker frame batch that fits in GPU memory.
 * The caller converts frames to file groups with nstep_valid. */
size_t EstimateGpuFrameBatch(size_t gpu_id, int nseg,
                             int filter_nfft, int output_nfft,
                             int num_ch, int filter_count,
                             size_t wh_flag, size_t runabs_flag,
                             size_t runabs_mf_flag,
                             double gpu_ram_limit_mib);

/* Allocate GPU arrays and cuFFT plans for one worker. */
void AllocateGpuMemory(int batch, int nseg, int filter_nfft, int output_nfft,
                       int num_ch, int do_runabs, int do_runabs_mf,
                       int wh_flag,
                       float **d_sacdata, cuComplex **d_spectrum,
                       float **d_sacdata_2x, cuComplex **d_spectrum_2x,
                       cuComplex **d_base_spectrum_2x,
                       float **d_filtered_sacdata,
                       float **d_total_sacdata,
                       float **d_filter_responses, float **d_tmp,
                       float **d_weight, float **d_tmp_weight,
                       int filter_count, double **d_sum, double **d_isum,
                       void **d_cufft_work,
                       cufftHandle *planfwd, cufftHandle *planinv,
                       cufftHandle *planfwd_filter,
                       cufftHandle *planinv_filter,
                       cufftHandle *planfwd_output);

#endif
