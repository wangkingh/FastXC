#include "filter_response.h"

#include "logger.h"

#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int estimateSinglePassImpulseLength(const ButterworthFilter *filter, int maxSamples)
{
    const double energyTol = 1.0e-6;
    double *response = (double *)calloc((size_t)maxSamples, sizeof(double));
    if (response == NULL)
    {
        return maxSamples;
    }

    double xhist[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double yhist[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double totalEnergy = 0.0;
    double a0 = filter->a[0];
    if (fabs(a0) < 1.0e-30)
    {
        free(response);
        return maxSamples;
    }

    for (int n = 0; n < maxSamples; n++)
    {
        for (int j = 4; j > 0; j--)
        {
            xhist[j] = xhist[j - 1];
            yhist[j] = yhist[j - 1];
        }
        xhist[0] = (n == 0) ? 1.0 : 0.0;

        double y = 0.0;
        for (int j = 0; j < 5; j++)
        {
            y += filter->b[j] * xhist[j];
        }
        for (int j = 1; j < 5; j++)
        {
            y -= filter->a[j] * yhist[j];
        }
        y /= a0;
        yhist[0] = y;

        response[n] = y;
        totalEnergy += y * y;
    }

    if (totalEnergy <= 0.0)
    {
        free(response);
        return 1;
    }

    double tailEnergy = 0.0;
    int impulseLength = 1;
    for (int n = maxSamples - 1; n >= 0; n--)
    {
        tailEnergy += response[n] * response[n];
        if (tailEnergy > totalEnergy * energyTol)
        {
            impulseLength = n + 1;
            break;
        }
    }

    free(response);
    return impulseLength;
}

int estimateButterworthFilterPadding(const ButterworthFilter *filters, int filterCount, int maxPadding)
{
    if (filters == NULL || filterCount <= 0 || maxPadding <= 0)
    {
        return maxPadding;
    }

    int maxImpulseLength = 1;
    for (int i = 0; i < filterCount; i++)
    {
        int impulseLength = estimateSinglePassImpulseLength(&filters[i], maxPadding);
        if (impulseLength > maxImpulseLength)
        {
            maxImpulseLength = impulseLength;
        }
    }

    int padding = 2 * maxImpulseLength;
    if (padding < 1)
    {
        padding = 1;
    }
    if (padding > maxPadding)
    {
        padding = maxPadding;
    }
    return padding;
}

static int parseCoefficientsLine(char *line, double *coefficients)
{
    if (sscanf(line, "%lf %lf %lf %lf %lf",
               &coefficients[0], &coefficients[1], &coefficients[2],
               &coefficients[3], &coefficients[4]) != 5)
    {
        LOG_ERROR("filter_coefficients_malformed", "line=\"%s\"", line);
        return -1;
    }
    return 0;
}

ButterworthFilter *readButterworthFilters(const char *filepath, int *filterCount)
{
    FILE *file = fopen(filepath, "r");
    if (file == NULL)
    {
        LOG_ERROR("open_filter_file_failed", "path=\"%s\"", filepath);
        return NULL;
    }

    char line[1024];
    int count = 0;
    int state = 0;
    ButterworthFilter *filters = NULL;
    ButterworthFilter tempFilter;
    memset(&tempFilter, 0, sizeof(tempFilter));

    while (fgets(line, sizeof(line), file))
    {
        if (line[0] == '#')
        {
            if (sscanf(line, "# %f/%f", &tempFilter.freq_low, &tempFilter.freq_high) != 2)
            {
                LOG_WARN("filter_band_malformed", "path=\"%s\" line=\"%s\"", filepath, line);
                continue;
            }
            state = 1;
            continue;
        }

        if (state == 1)
        {
            if (parseCoefficientsLine(line, tempFilter.b) != 0)
            {
                continue;
            }
            state = 2;
            continue;
        }

        if (state == 2)
        {
            if (parseCoefficientsLine(line, tempFilter.a) != 0)
            {
                state = 0;
                continue;
            }

            ButterworthFilter *newFilters = (ButterworthFilter *)realloc(filters, (size_t)(count + 1) * sizeof(ButterworthFilter));
            if (newFilters == NULL)
            {
                LOG_ERROR("alloc_failed", "target=butterworth_filters count=%d", count + 1);
                free(filters);
                fclose(file);
                return NULL;
            }
            filters = newFilters;
            filters[count] = tempFilter;
            count++;
            memset(&tempFilter, 0, sizeof(tempFilter));
            state = 0;
        }
    }

    *filterCount = count;
    fclose(file);
    return filters;
}

static void calFilterPowerResp(const double *b, const double *a, int nseg_2x, float *response)
{
    if (b == NULL || a == NULL || response == NULL || nseg_2x <= 0)
    {
        return;
    }

    int half = nseg_2x / 2;
    if (half <= 0)
    {
        return;
    }

    for (int i = 0; i < half + 1; i++)
    {
        double normalized_freq = (double)i / (double)half;
        double Bx = 0.0;
        double By = 0.0;
        double Ax = 0.0;
        double Ay = 0.0;

        for (int j = 0; j < 5; j++)
        {
            double angle = -1.0 * M_PI * normalized_freq * (double)j;
            double c = cos(angle);
            double s = sin(angle);

            Bx += b[j] * c;
            By += b[j] * s;
            Ax += a[j] * c;
            Ay += a[j] * s;
        }

        double num_power = Bx * Bx + By * By;
        double den_power = Ax * Ax + Ay * Ay;
        double response_power = (den_power <= DBL_MIN) ? 0.0 : (num_power / den_power);
        if (!isfinite(response_power))
        {
            response_power = 0.0;
        }
        else if (response_power > FLT_MAX)
        {
            response_power = FLT_MAX;
        }

        response[i] = (float)response_power;
    }
}

FilterResp *processButterworthFilters(ButterworthFilter *filters, int filterCount, float df_2x, int nseg_2x)
{
    (void)df_2x;
    if (filters == NULL || filterCount <= 0 || nseg_2x <= 0)
    {
        LOG_ERROR("filter_response_invalid_input", "filter_count=%d nfft=%d", filterCount, nseg_2x);
        return NULL;
    }

    FilterResp *responses = (FilterResp *)malloc((size_t)filterCount * sizeof(FilterResp));
    if (responses == NULL)
    {
        LOG_ERROR("alloc_failed", "target=filter_responses count=%d", filterCount);
        return NULL;
    }

    for (int i = 0; i < filterCount; i++)
    {
        responses[i].freq_low = filters[i].freq_low;
        responses[i].response = (float *)calloc((size_t)nseg_2x, sizeof(float));
        if (responses[i].response == NULL)
        {
            LOG_ERROR("alloc_failed", "target=filter_response index=%d nfft=%d", i, nseg_2x);
            for (int j = 0; j < i; j++)
            {
                free(responses[j].response);
            }
            free(responses);
            return NULL;
        }

        calFilterPowerResp(filters[i].b, filters[i].a, nseg_2x, responses[i].response);
    }

    return responses;
}
