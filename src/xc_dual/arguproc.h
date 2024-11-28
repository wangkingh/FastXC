#ifndef __CU_ARG_PROC_H
#define __CU_ARG_PROC_H

#define MAX_GPU_COUNT 100

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

typedef struct ARGUTYPE
{

  char *src_files_list; // input list file of -A and -B
  char *sta_files_list;
  char *ncf_dir;            // output dir for CC vector
  float cc_len;             // half length of output NCF
  int gpu_id;               // GPU ID
  int save_linear;          // 是否线性叠加结果
  int save_pws;             // 保留相位加权叠加结果
  int save_tfpws;           // 保留时频域相位加权叠加结果
  int save_segment;         // 是否保存每一段NCF(叠前)
  int cpu_count;            // cpu并行数
  int gpu_num;              // 同时开启的gpu数量
  float threshold_distance; // 阈值距离
} ARGUTYPE;

void usage();
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg);
#endif