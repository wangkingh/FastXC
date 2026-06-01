#ifndef _PREPROCESS_CUH
#define _PREPROCESS_CUH

#include <cstddef>

int preprocess(float *d_sacdata, double *d_sum, double *d_isum,
                               int pitch, size_t row_count,
                               float freq_low, float delta);

#endif
