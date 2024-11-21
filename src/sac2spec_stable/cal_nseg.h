#ifndef _CAL_NSEG_H_
#define _CAL_NSEG_H_
#include <errno.h>
#include <stdio.h>

// Find the number of segment using npts and seglen, delta
int cal_nseg(int seglen, int npts, float delta);

#endif