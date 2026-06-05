#ifndef _NORMALIZATION_CUH
#define _NORMALIZATION_CUH

#include <cstddef>
#include <cuComplex.h>
#include <cufft.h>

int time_onebit(float *d_data,
                                size_t pitch,
                                size_t width,
                                size_t height);

int runabs(float *d_data,
                           float *d_tmp,
                           float *d_weight,
                           float *d_tmp_weight,
                           float freq_low,
                           float delta,
                           int frame_count,
                           int num_ch,
                           int pitch,
                           float maxval);

int runabs_mf(float *d_sacdata,
                              float *d_filtered_sacdata,
                              float *d_total_sacdata,
                              float *d_padded_sacdata,
                              cuComplex *d_padded_spectrum,
                              cuComplex *d_base_padded_spectrum,
                              float *d_responses,
                              float *d_tmp,
                              float *d_weight,
                              float *d_tmp_weight,
                              float *freq_lows,
                              int filter_count,
                              float delta,
                              int frame_count,
                              int num_ch,
                              float maxval,
                              int segment_pts,
                              int filter_nfft,
                              int plan_batch,
                              cufftHandle *planinv_filter,
                              cufftHandle *planfwd_filter);

#endif
