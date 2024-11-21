#ifndef _ROTATE_WRITE_H_
#define _ROTATE_WRITE_H_

#include <stdio.h>
#include <math.h>
#include "sac.h"

void generate_rotate_matrix(double azi, double baz, double **matrix);

void rotate(float **rtz_data, float **enz_data, double **rotate_matrix, int npts);

#endif