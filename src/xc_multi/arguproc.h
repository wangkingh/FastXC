#ifndef __CU_ARG_PROC_H
#define __CU_ARG_PROC_H

#define MAX_GPU_COUNT 100

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct ARGUTYPE
{

  char *src_lst_path;  /* input list file of -A */
  char *sta_lst_path;  /* input list file of -B */
  char *ncf_dir;       /* output dir for CC vector */
  float cclength;      /* half length of output NCF */
  float max_distance;  /* higher distance threshold for calaulating CCF*/
  size_t gpu_id;       /* GPU ID */
  size_t gpu_task_num; /*number of tasks deploy on a single GPU*/
  size_t cpu_count;    /*number of CPUs will be used in this threads*/
} ARGUTYPE;

void usage();
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);

#endif