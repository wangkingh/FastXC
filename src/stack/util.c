#include "util.h"
#include <string.h>
#include <stdlib.h>
#include <sys/sysinfo.h>

const float SHRINKRATIO = 2;
const float RAMUPPERBOUND = 0.8;

size_t QueryAvailCpuRam()
{
    const size_t LINEMAX = 256;
    const size_t KILOBYTES = 1L<<10;
    const size_t GIGABYTES = 1L<<30;

    struct sysinfo sinfo;
    char buffer[LINEMAX];

    FILE *fid = fopen("/proc/meminfo", "r");

    size_t availram = 0;    

    while (fgets(buffer, LINEMAX, fid) != NULL)
    {
        if (strstr(buffer, "MemAvailable") != NULL)
        {
            sscanf(buffer, "MemAvailable: %lu kB", &availram);
            availram *= KILOBYTES; /* kB -> B */
            availram *= RAMUPPERBOUND;
        }
    }
    fclose(fid);

    /* In Linux sysinfo's free ram is far smaller than available ram
     * Use this in condition that cannot find Memavailble in /proc/meminfo
     */
    if (availram == 0)
    {
        int err = sysinfo(&sinfo);
        if (err != 0)
        {
            perror("Get sys info\n");
            exit(-1);
        }        
        availram = sinfo.freeram;
    }

    printf("Avail cpu ram: %.3f GB\n", availram*1.0/GIGABYTES);

    return availram;
}

void CpuMalloc(void **pptr, size_t sz)
{
    if ((*pptr = malloc(sz)) == NULL)
    {
        perror("Malloc cpu memory");
        exit(-1);
    }
}

void CpuCalloc(void **pptr, size_t sz)
{
    if ((*pptr = malloc(sz)) == NULL)
    {
        perror("Calloc cpu memory\n");
        exit(-1);
    }
    memset(*pptr, 0, sz);
}

void CpuFree(void **pptr)
{
    free(*pptr);
    *pptr = NULL;
}

size_t EstimateCpuBatch(size_t batch, size_t unitram)
{
    size_t availram = QueryAvailCpuRam();

    size_t reqram = batch * unitram;
    while (reqram >= availram)
    {
        batch = (batch + SHRINKRATIO - 1) / SHRINKRATIO;
        reqram = batch * unitram;
    }

    return batch;
}
