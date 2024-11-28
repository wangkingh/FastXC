#ifndef _UTIL_H
#define _UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <linux/limits.h>
#include "sac.h"

// sharedItem
typedef struct
{
  pthread_mutex_t mtx;
  int valid; /* -1: default; 1: ready to file; 2: finish to file */
  char fname[PATH_MAX];
  SACHEAD *phead;
  float *pdata;
} SHAREDITEM;

size_t QueryAvailCpuRam();
size_t EstimateCpuBatch(size_t fixedRam, size_t unitRam);
void CpuMalloc(void **pptr, size_t sz);
void CpuCalloc(void **pptr, size_t sz);
void CpuFree(void **pptr);

double getElapsedTime(struct timespec start, struct timespec end);
int write_multiple_sac(const char *filename, SHAREDITEM *pItem, int paircnt);

#endif
