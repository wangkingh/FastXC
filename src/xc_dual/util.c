#include "util.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <pthread.h>
#include <linux/limits.h>

#include "segspec.h"
#include "sac.h"
#include "arguproc.h"
#include "read_segspec.h"
#include "gen_ncf_path.h"
#include "util.h"

const float SHRINKRATIO = 2;
const float RAMUPPERBOUND = 0.8;

size_t QueryAvailCpuRam()
{
  const size_t LINEMAX = 256;
  const size_t KILOBYTES = 1L << 10;
  const size_t GIGABYTES = 1L << 30;

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

  // availram = availram / 4; // set by wangjx 2023.06.07
  availram = availram;
  printf("[INFO]: Avail cpu ram: %.3f GB\n", availram * 1.0 / GIGABYTES);

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

// Estimate the number of batches that can be allocated in CPU RAM
size_t EstimateCpuBatch(size_t fixedRam, size_t unitRam)
{
  // Query available CPU RAM
  size_t availableRam = QueryAvailCpuRam();

  printf("[INFO]: available CPU RAM: %.3f GB\n", (float)availableRam / (1024 * 1024 * 1024));

  if (availableRam < fixedRam)
  {
    fprintf(stderr, "Error: Available RAM is less than fixed RAM (%lu GB).\n",
            fixedRam >> 30);
    exit(EXIT_FAILURE);
  }

  // Initialize batch count and required RAM
  size_t batchCount = 0;
  size_t requiredRam = 0;

  // Keep increasing the batch count until required RAM exceeds available RAM
  while (requiredRam < availableRam)
  {
    // Increment the batch count
    batchCount++;

    // Update the required RAM based on the new batch count
    requiredRam = fixedRam + batchCount * unitRam;
  }

  // Decrease the batch count by 1 since the last increment caused required RAM
  // to exceed available RAM
  if (batchCount > 1)
  {
    batchCount--;
  }
  else
  {
    fprintf(stderr,
            "Error: Not enough available RAM to allocate a single batch.\n");
    exit(EXIT_FAILURE);
  }

  // Return the estimated batch count
  return batchCount;
}

double getElapsedTime(struct timespec start, struct timespec end)
{
  return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// write all ncf.sac into just one file
int write_multiple_sac(const char *filename, SHAREDITEM *pItem, int paircnt)
{
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL)
  {
    fprintf(stderr, "Error opening file %s for writing ncf.sac!\n", filename);
    return -1;
  }

  for (int i = 0; i < paircnt; i++)
  {
    SHAREDITEM *ptr = pItem + i;
    // printf("ncf_filename: %s\n", ptr->fname);

    // 写入文件名
    // size_t fname_len = strlen(ptr->fname);
    if (fwrite(ptr->fname, PATH_MAX, 1, fp) != 1)
    {
      fprintf(stderr, "Error writing file name for item %d.\n", i);
      fclose(fp);
      return -1;
    }
    // 写入SAC头部信息
    if (fwrite(ptr->phead, sizeof(SACHEAD), 1, fp) != 1)
    {
      fprintf(stderr, "Error writing SAC header for item %d.\n", i);
      fclose(fp);
      return -1;
    }

    // 写入SAC数据
    int data_size = ptr->phead->npts * sizeof(float); // 计算数据大小
    if (fwrite(ptr->pdata, data_size, 1, fp) != 1)
    {
      fprintf(stderr, "Error writing SAC data for item %d.\n", i);
      fclose(fp);
      return -1;
    }
  }

  // 使用ftell获取文件大小
  long filesize = ftell(fp);
  if (filesize == -1)
  {
    fprintf(stderr, "Error determining file size.\n");
    fclose(fp);
    return -1;
  }
  else
  {
    double filesizeGB = filesize / (double)(1 << 30); // 转换为GB
    printf("[INFO]: File size: %.3f GB.\n", filesizeGB);
  }

  fclose(fp);
  return 0;
}
