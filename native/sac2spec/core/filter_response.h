#ifndef _FILTER_RESPONSE_H
#define _FILTER_RESPONSE_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ButterworthFilter
{
    float freq_low;
    float freq_high;
    double b[5];
    double a[5];
} ButterworthFilter;

typedef struct FilterResp
{
    float freq_low;
    float *response;
} FilterResp;

ButterworthFilter *readButterworthFilters(const char *filepath, int *filterCount);

FilterResp *processButterworthFilters(ButterworthFilter *filters, int filterCount,
                                      float df, int nfft);

int estimateButterworthFilterPadding(const ButterworthFilter *filters,
                                     int filterCount, int maxPadding);

#endif
