#ifndef _UTIL_H
#define _UTIL_H

#include <stdio.h>
#include <sys/sysinfo.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

size_t QueryAvailCpuRam();
size_t EstimateCpuBatch(size_t unitRam, int thread_num);
void CpuMalloc(void **, size_t);
void CpuFree(void **);
char* my_strdup(const char* s);
#endif