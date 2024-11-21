/* Header for the segment by segment spectrum for noise cross correlation
 * History:
 * 1. init by wangwt to speed up CC on huge dataset.
 *
 * last update wangjx@20230504
 * */

#ifndef _SEGSPEC_H
#define _SEGSPEC_H

typedef struct segspec_s
{
  float stla;
  float stlo;
  /* segment info */
  int nstep;

  /* FFT info  */
  int nspec; /* use fftr() number of complex eg 2*nspec float */
  float df;
  float dt;

} SEGSPEC;

#endif
