/* Header for segment-by-segment spectra used by cross-correlation. */

#ifndef _SEGSPEC_H
#define _SEGSPEC_H

typedef struct segspec_s
{
  float stla;
  float stlo;
  int nstep;
  int nspec; /* use fftr() number of complex eg 2*nspec float */
  float df;
  float dt;
  int gnsl_id;
} SEGSPEC;

#endif
