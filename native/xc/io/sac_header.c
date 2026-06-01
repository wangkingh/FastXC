#include "sac_header.h"

void SacheadProcess(SACHEAD *ncfhd,
                    float stla, float stlo, float evla, float evlo,
                    float Gcarc, float Az, float Baz, float Dist,
                    float delta, int ncc, float cclength,
                    const XcTimeData *time_info)
{
    *ncfhd = sac_null;

    ncfhd->stla = stla;
    ncfhd->stlo = stlo;
    ncfhd->evla = evla;
    ncfhd->evlo = evlo;

    ncfhd->gcarc = Gcarc;
    ncfhd->az = Az;
    ncfhd->baz = Baz;
    ncfhd->dist = Dist;

    ncfhd->iftype = 1;
    ncfhd->leven = 1;
    ncfhd->delta = delta;
    ncfhd->npts = ncc;
    ncfhd->b = -1.0f * cclength;
    ncfhd->e = cclength;
    ncfhd->unused27 = 1;
    ncfhd->o = 0.0f;

    if (time_info)
    {
        ncfhd->nzyear = time_info->year;
        ncfhd->nzjday = time_info->day_of_year;
        ncfhd->nzhour = time_info->hour;
        ncfhd->nzmin = time_info->minute;
        ncfhd->nzsec = 0;
        ncfhd->nzmsec = 0;
    }
}
