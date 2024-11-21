#include "design_filter_response.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// parse single filter line
int parseCoefficientsLine(char *line, double *coefficients)
{
    if (sscanf(line, "%lf %lf %lf %lf %lf",
               &coefficients[0], &coefficients[1], &coefficients[2],
               &coefficients[3], &coefficients[4]) != 5)
    {
        fprintf(stderr, "Error: Invalid coefficients line format\n");
        return -1;
    }
    return 0;
}

// read butterworth filter file
ButterworthFilter *readButterworthFilters(const char *filepath, int *filterCount)
{
    FILE *file = fopen(filepath, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error: Cannot open filter file %s\n", filepath);
        return NULL;
    }

    char line[1024];
    int count = 0, state = 0;
    ButterworthFilter *filters = NULL;
    ButterworthFilter tempFilter;

    while (fgets(line, sizeof(line), file))
    {
        // Find the first line of a filter
        if (line[0] == '#')
        {
            if (sscanf(line, "# %f/%f", &tempFilter.freq_low, &tempFilter.freq_high) != 2)
            {
                fprintf(stderr, "Error: Invalid frequency band format\n");
                continue;
            }
            state = 1; // the next line is the b line of a filter
            continue;
        }

        // parsing b line, the numerator coefficients
        if (state == 1)
        {
            if (parseCoefficientsLine(line, tempFilter.b) != 0)
            {
                continue;
            }
            state = 2; // the next line is the a line of a filter
            continue;
        }

        // parsing a line, the denominator coefficients
        if (state == 2)
        {
            if (parseCoefficientsLine(line, tempFilter.a) != 0)
            {
                state = 0; // reset the state if the line is invalid
                continue;
            }

            ButterworthFilter *newFilters = (ButterworthFilter *)realloc(filters, (count + 1) * sizeof(ButterworthFilter));
            if (!newFilters)
            {
                perror("Error reallocating memory");
                free(filters);
                fclose(file);
                return NULL;
            }
            filters = newFilters;
            filters[count] = tempFilter;
            count++;
            state = 0; // reset the state
        }
    }
    *filterCount = count;
    fclose(file);
    return filters;
}

// debug function to print butterworth filters
void printButterworthFilters(const ButterworthFilter *filters, int filterCount)
{
    int i;
    if (filters == NULL)
    {
        printf("No filters to display.\n");
        return;
    }

    for (i = 0; i < filterCount; ++i)
    {
        const ButterworthFilter *filter = &filters[i];
        printf("Filter %d:\n", i + 1);
        printf("    Frequency Band: %.3f - %.3f\n", filter->freq_low, filter->freq_high);
        printf("    b Coefficients: %.5e %.5e %.5e %.5e %.5e\n",
               filter->b[0], filter->b[1], filter->b[2], filter->b[3], filter->b[4]);
        printf("    a Coefficients: %.5e %.5e %.5e %.5e %.5e\n",
               filter->a[0], filter->a[1], filter->a[2], filter->a[3], filter->a[4]);
        printf("\n");
    }
}

void calFilterResp(double *b, double *a, int nseg, complex *response)
{
    int i, j;
    for (i = 0; i < nseg / 2 + 1; i++)
    {
        float normalized_freq = (float)i / (float)(nseg / 2 + 1);

        complex B = {0, 0};
        complex A = {0, 0};

        // Calculate B(f) and A(f), using DTFT
        for (j = 0; j < 5; j++)
        {
            float angle = -1 * M_PI * normalized_freq * j;
            complex exp_val = {cos(angle), sin(angle)}; // e^(-j * 2 * pi * f * n)
            B.x += b[j] * exp_val.x;
            B.y += b[j] * exp_val.y;
            A.x += a[j] * exp_val.x;
            A.y += a[j] * exp_val.y;
        }

        // calculate magnitude and phase
        float magnitude = sqrt(B.x * B.x + B.y * B.y) / sqrt(A.x * A.x + A.y * A.y);
        float phase = atan2(B.y, B.x) - atan2(A.y, A.x);

        // calculate real and imaginary part of response
        response[i].x = magnitude * cos(phase);
        response[i].y = magnitude * sin(phase);
    }
}

FilterResp *processButterworthFilters(ButterworthFilter *filters, int filterCount, float df_1x, int nseg)
{
    // allocate a pointer for all filter responses
    FilterResp *responses = (FilterResp *)malloc(filterCount * sizeof(FilterResp));
    if (responses == NULL)
    {
        fprintf(stderr, "Memory allocation failed for filter responses.\n");
        return NULL;
    }
    int i, j;
    for (i = 0; i < filterCount; i++)
    {
        // allocate memory for current filter response
        responses[i].freq_low = filters[i].freq_low; // set lower frequency limit
        responses[i].response = (complex *)malloc(nseg * sizeof(complex));

        if (responses[i].response == NULL)
        {
            fprintf(stderr, "Memory allocation failed for filter response.\n");
            for (j = 0; j < i; j++)
            {
                free(responses[j].response);
            }
            free(responses);
            return NULL;
        }

        if (responses[i].response != NULL)
        {
            memset(responses[i].response, 0, nseg * sizeof(complex));
        }

        // calculate filter response
        calFilterResp(filters[i].b, filters[i].a, nseg, responses[i].response);
    }
    return responses;
}