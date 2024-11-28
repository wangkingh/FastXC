#ifndef _CAL_DIST_H_
#define _CAL_DIST_H_

#include<stdio.h>
#include<math.h>
#include<float.h>
#ifndef M_PI
#define M_PI 3.1415926535897
#endif

void distkm_az_baz_Rudoe(double evlo,double evla,double stlo,double stla,double *gcarc,double *az,double* baz,double *distkm);




#endif