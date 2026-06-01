#ifndef SAC_HEADER_H
#define SAC_HEADER_H

#include "include/sac.h"

typedef struct XcTimeData
{
    int year;
    int day_of_year;
    int hour;
    int minute;
} XcTimeData;

#ifdef __cplusplus
extern "C"
{
#endif

void SacheadProcess(SACHEAD *ncfhd,
                    float stla, float stlo, float evla, float evlo,
                    float Gcarc, float Az, float Baz, float Dist,
                    float delta, int ncc, float cclength,
                    const XcTimeData *time_info);

#ifdef __cplusplus
}
#endif

#endif
