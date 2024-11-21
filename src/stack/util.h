#ifndef _UTIL_H
#define _UTIL_H

#include <stdio.h>

size_t QueryAvailCpuRam();
size_t EstimateCpuBatch(size_t batch, size_t unitram);
void CpuMalloc(void **pptr, size_t sz);
void CpuCalloc(void **pptr, size_t sz);
void CpuFree(void **pptr);

#endif
