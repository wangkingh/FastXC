#ifndef _MY_WRITE_H
#define _MY_WRITE_H

#include "include/sac.h"
#include "include/write_mode.h"

int my_write_sac(const char *name, SACHEAD hd, const float *ar, int write_mode);

#endif
