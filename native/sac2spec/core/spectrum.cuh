#ifndef _SPECTRUM_CUH
#define _SPECTRUM_CUH

#include <cstddef>
#include <cuComplex.h>

int fft_forward_normalize(cuComplex *d_data,
                          size_t pitch,
                          size_t width,
                          size_t height,
                          float delta);

int fft_inverse_normalize(float *d_data,
                          size_t pitch,
                          size_t width,
                          size_t height,
                          float delta);

int complex_sanitize(cuComplex *d_data,
                     size_t pitch,
                     size_t width,
                     size_t height);

int apply_filter_response(cuComplex *d_spectrum,
                          const float *d_response,
                          size_t pitch,
                          size_t width,
                          size_t height);

int phase_only(cuComplex *d_data,
               size_t pitch,
               size_t width,
               size_t height,
               float minval);

int freq_whiten(cuComplex *d_spectrum,
                float *d_weight,
                float *d_tmp_weight,
                float *d_tmp,
                int num_ch,
                int pitch,
                int frame_count,
                float delta,
                int idx1,
                int idx2,
                int idx3,
                int idx4);

#endif
