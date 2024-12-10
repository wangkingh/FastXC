#ifndef _DESIGN_RESPONSE_H_
#define _DESIGN_RESPONSE_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "complex.h"

typedef struct ButterworthFilter
{
    float freq_low;
    float freq_high;
    double b[5]; /* Numerator coefficients */
    double a[5]; /* Denominator coefficients */
} ButterworthFilter;

typedef struct FilterResp
{
    float freq_low;    // lower frequency
    complex *response; // filter response
} FilterResp;

int parseCoefficientsLine(char *line, double *coefficients);

ButterworthFilter *readButterworthFilters(const char *filepath, int *filterCount);

void printButterworthFilters(const ButterworthFilter *filters, int filterCount);

void calFilterResp(double *b, double *a, int nseg, complex *response);

FilterResp *processButterworthFilters(ButterworthFilter *filters, int filterCount, float df_1x, int nseg);

#endif