#ifndef _ARG_PROC_H
#define _ARG_PROC_H

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct ARGUTYPE
{
  char *sourcepack_list;   /* text file: one sourcepack_index.tsv path per line */
  char *output_sourcepack; /* output directory for TF-PWS SourcePack */
  char *gpu_list;          /* comma-separated GPU worker list; duplicate IDs are allowed */
  char *gpu_ram_limit_mib_list;
  int band_limited;
  double band_fmin;
  double band_fmax;
  double band_taper_hz;
  int sub_stack_size;
  char *progress_file;
} ARGUTYPE;

void usage(void);
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg);

#endif
