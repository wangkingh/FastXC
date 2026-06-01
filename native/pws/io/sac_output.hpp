#ifndef SAC_OUTPUT_HPP
#define SAC_OUTPUT_HPP

#include <cstddef>
#include <cstring>

extern "C"
{
#include "sac.h"
}

static inline void set_sac_string(char *dst, std::size_t width, const char *src)
{
    std::memset(dst, ' ', width);
    std::size_t len = std::strlen(src);
    if (len > width)
        len = width;
    std::memcpy(dst, src, len);
}

static inline void configure_output_header(SACHEAD *hd,
                                           unsigned nsamples,
                                           unsigned num_segments,
                                           unsigned ngroups,
                                           std::size_t group_sz)
{
    hd->npts = (int)nsamples;
    hd->iftype = ITIME;
    hd->leven = TRUE;

    if (hd->delta != -12345.0f)
        hd->e = hd->b + (float)(nsamples - 1) * hd->delta;

    hd->user0 = (float)num_segments;
    hd->user1 = (float)ngroups;
    hd->user2 = (float)group_sz;
    set_sac_string(hd->kuser0, sizeof(hd->kuser0), "NTRACE");
    set_sac_string(hd->kuser1, sizeof(hd->kuser1), "NGROUP");
    set_sac_string(hd->kuser2, sizeof(hd->kuser2), "SUBSTK");

    hd->nzyear = 2010;
    hd->nzjday = 214;
    hd->nzhour = 0;
    hd->nzmin = 0;
    hd->nzsec = 0;
    hd->nzmsec = 0;
}

static inline void update_output_stats(SACHEAD *hd,
                                       const float *data,
                                       unsigned nsamples)
{
    if (nsamples == 0)
        return;

    float minv = data[0];
    float maxv = data[0];
    double sum = 0.0;

    for (unsigned i = 0; i < nsamples; ++i)
    {
        float v = data[i];
        if (v < minv)
            minv = v;
        if (v > maxv)
            maxv = v;
        sum += v;
    }

    hd->depmin = minv;
    hd->depmax = maxv;
    hd->depmen = (float)(sum / (double)nsamples);
}

#endif
