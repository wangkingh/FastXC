#include "fft_length.h"
#include "logger.h"

#include <errno.h>
#include <limits.h>
#include <math.h>

static int seconds_to_nearest_samples(float seconds, float delta)
{
    if (seconds <= 0.0f || delta <= 0.0f)
        return 0;

    double samples = (double)seconds / (double)delta;
    if (!isfinite(samples) || samples > (double)INT_MAX)
        return INT_MAX;

    return (int)llround(samples);
}

static int is_smooth_2357(int value)
{
    int remaining = value;
    int primeFactors[] = {2, 3, 5, 7};
    int i;

    if (remaining < 1)
        return 0;

    for (i = 0; i < 4; i++)
    {
        while (remaining > 1 && remaining % primeFactors[i] == 0)
        {
            remaining /= primeFactors[i];
        }
    }
    return remaining == 1;
}

int next_smooth_2357_length(int min_length)
{
    int length = min_length;
    if (length < 1)
        length = 1;

    while (1)
    {
        if (is_smooth_2357(length))
        {
            break;
        }
        length++;
    }
    return length;
}

int next_even_smooth_2357_length(int min_length)
{
    int length = next_smooth_2357_length(min_length);
    while (length % 2 != 0)
    {
        length = next_smooth_2357_length(length + 1);
    }
    return length;
}

static int previous_smooth_2357_length(int max_length, int *error)
{
    if (max_length <= 1)
    {
        *error = EINVAL;
        return 0;
    }

    int length = max_length - 1;
    while (length > 0)
    {
        if (is_smooth_2357(length))
        {
            *error = 0;
            return length;
        }
        length--;
    }

    *error = EINVAL;
    return 0;
}

int segment_length_from_seconds(float seconds, int max_samples, float delta)
{
    int requested_samples = seconds_to_nearest_samples(seconds, delta);
    int segment_samples = next_smooth_2357_length(requested_samples);

    if (segment_samples > max_samples)
    {
        int err;
        segment_samples = previous_smooth_2357_length(max_samples, &err);
        if (err)
        {
            LOG_ERROR("smooth_2357_length_failed", "input_npts=%d", max_samples);
        }
        else
        {
            LOG_WARN("segment_length_clamped", "requested_pts=%d npts=%d effective_pts=%d",
                     requested_samples, max_samples, segment_samples);
        }
    }
    return segment_samples;
}
