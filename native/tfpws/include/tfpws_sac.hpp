#ifndef TFPWS_SAC_HPP
#define TFPWS_SAC_HPP

extern "C"
{
#include "sac.h"
}

static inline void configure_tfpws_output_header(SACHEAD *hd,
                                                 unsigned nsamples,
                                                 unsigned num_segments)
{
    hd->npts = (int)nsamples;
    hd->iftype = ITIME;
    hd->user0 = (float)num_segments;
    hd->nzyear = 2010;
    hd->nzjday = 214;
    hd->nzhour = 0;
    hd->nzmin = 0;
    hd->nzmsec = 0;
}

#endif
