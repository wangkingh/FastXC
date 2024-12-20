#ifndef _CU_PRE_PROCESS_H_
#define _CU_PRE_PROCESS_H_
#include "cuda.util.cuh"
#include "cuda.misc.cuh"
#include "cuda.rdcrtr.cuh"
#include "cuda.taper.cuh"

void preprocess(float *d_sacdata, double *d_sum, double *d_isum, int pitch, size_t proccnt, float freq_low, float delta);

void runabs_mf(float *d_sacdata, float *d_filtered_sacdata, float *d_total_sacdata,
               float *d_sacdata_2x, cuComplex *d_spectrum_2x,
               cuComplex *d_responses, float *d_tmp,
               float *d_weight, float *d_tmp_weight,
               cufftHandle *planinv, float *freq_lows,
               int filterCount, float delta, int proc_batch, int num_ch, float maxval,
               int nseg_1x, int nseg_2x, cufftHandle *planinv_2x, cufftHandle *planfwd_2x);

void freqWhiten(cuComplex *d_spectrum,
                float *d_weight, float *d_tmp_weight, float *d_tmp,
                int num_ch, int pitch, int proc_batch,
                float delta, int idx1, int idx2, int idx3, int idx4);

void runabs(float *d_sacdata, float *d_tmp, float *d_weight, float *d_tmp_weight,
            float freq_lows_limit, float delta, int proc_batch, int num_ch, int pitch, float maxval);

#endif