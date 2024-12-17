#ifndef _CU_PRE_PROCESS_H_
#define _CU_PRE_PROCESS_H_
#include "cuda.util.cuh"
#include "cuda.misc.cuh"
#include "cuda.rdcrtr.cuh"
#include "cuda.filter.cuh"
#include "cuda.taper.cuh"

extern "C"
{
#include "design_filter_response.h"
}

void preprocess(float *d_sacdata, double *d_sum, double *d_isum, int pitch, size_t proccnt, int taper_percentage);

void mycustom_filter(float *d_sacdata, float *d_filtered, float *d_filtered_tmp,
                     double *d_a, double *d_b, float *d_sac_hist, float *d_filtered_hist,
                     size_t width, size_t height);

void runabs_mf(float *d_sacdata, float *d_sac_sum,
               float *d_filtered, float *d_filtered_tmp,
               double *d_a, double *d_b, float *d_sac_hist, float *d_filtered_hist,
               float *d_tmp, float *d_weight, float *d_tmp_weight, ButterworthFilter *filters,
               int filterCount, float delta, int proc_batch, int num_ch, int pitch, float maxval);

void freqWhiten(cuComplex *d_spectrum,
                float *d_weight, float *d_tmp_weight, float *d_tmp,
                int num_ch, int pitch, int proc_batch,
                float delta, int idx1, int idx2, int idx3, int idx4);

void runabs(float *d_sacdata, float *d_tmp, float *d_weight, float *d_tmp_weight,
            float freq_lows_limit, float delta, int proc_batch, int num_ch, int pitch, float maxval);

#endif