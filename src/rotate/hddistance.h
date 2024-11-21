#ifndef _HDDISTANCE_H_
#define _HDDISTANCE_H_
#ifndef M_PI
#define M_PI 3.1415926535897932
#endif
extern void hd_distaz(float evlo, float evla, float stlo, float stla,
                      float *gcarc, float *az, float *baz, float *distkm);

#endif
