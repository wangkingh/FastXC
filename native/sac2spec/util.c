#include "util.h"
#include "logger.h"

#include <sys/sysinfo.h>

static const float CPU_RAM_UPPER_BOUND = 0.8f;

size_t QueryAvailCpuRam()
{
    const size_t LINEMAX = 256;
    const size_t KILOBYTES = 1L << 10;
    // const size_t GIGABYTES = 1L << 30;
    struct sysinfo sinfo;
    char buffer[LINEMAX];

    FILE *fid = fopen("/proc/meminfo", "r");

    size_t availram = 0;

    if (fid != NULL)
    {
        while (fgets(buffer, LINEMAX, fid) != NULL)
        {
            if (strstr(buffer, "MemAvailable") != NULL)
            {
                sscanf(buffer, "MemAvailable: %lu kB", &availram);
                availram *= KILOBYTES; /* kB -> B */
                availram *= CPU_RAM_UPPER_BOUND;
            }
        }
        fclose(fid);
    }

    /* In Linux, sysinfo's free RAM is usually smaller than available RAM.
     * Use it only when MemAvailable cannot be read from /proc/meminfo.
     */
    if (availram == 0)
    {
        int err = sysinfo(&sinfo);
        if (err != 0)
        {
            LOG_ERROR("cpu_memory_query_failed", "api=sysinfo");
            exit(-1);
        }
        availram = sinfo.freeram;
    }

    return availram;
}

size_t EstimateCpuBatch(size_t unitRam, int thread_num)
{
    // Query available CPU RAM
    size_t availableRam = QueryAvailCpuRam() / thread_num;
    const size_t GIGABYTES = 1L << 30;
    LOG_INFO("cpu_memory_available", "available_gb=%.3f",
             availableRam * 1.0 / GIGABYTES);
    // Initialize batch count and required RAM
    size_t batch = 0;
    size_t requiredRam = 0;

    // Keep increasing the batch count until required RAM exceeds available RAM
    while (requiredRam < availableRam)
    {
        // Increment the batch count
        batch++;

        // Update the required RAM based on the new batch count
        requiredRam = batch * unitRam;
    }

    // Decrease the batch count by 1 since the last increment caused required RAM
    // to exceed available RAM
    batch--;
    batch = batch > _RISTRICT_MAX_CPU_BATCH ? _RISTRICT_MAX_CPU_BATCH : batch;
    // Return the estimated batch count
    return batch;
}

void CpuMalloc(void **pptr, size_t sz)
{
    if ((*pptr = malloc(sz)) == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "bytes=%zu", sz);
        exit(-1);
    }
}

void CpuFree(void **pptr)
{
    free(*pptr);
    *pptr = NULL;
}

// in case some old compiler cannot deal with strdup
char *my_strdup(const char *s)
{
    if (s == NULL)
    {
        return NULL;
    }
    char *new_str = (char *)malloc(strlen(s) + 1);
    if (new_str == NULL)
    {
        return NULL;
    }
    char *p = new_str;
    while (*s)
    {
        *p++ = *s++;
    }
    *p = '\0';
    return new_str;
}
