#ifndef _UTIL_H
#define _UTIL_H

#include <stdio.h>
#include <stdlib.h>

size_t QueryAvailCpuRam();
size_t EstimateCpuBatch(size_t fixedRam, size_t unitRam);
void CpuMalloc(void **pptr, size_t sz);
void CpuCalloc(void **pptr, size_t sz);
void CpuFree(void **pptr);
char* my_strdup(const char* s);
#endif
