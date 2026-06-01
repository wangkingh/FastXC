/* Legacy segment-spectrum header for the pre-XCache XC pipeline.
 *
 * Current native/xc consumes XCache `.xcspec` shards and does not use this
 * structure in the active compute path.  The header is kept only to document
 * the older on-disk format and to make historical files easier to inspect.
 *
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
  int gnsl_id;

} SEGSPEC;

#endif
