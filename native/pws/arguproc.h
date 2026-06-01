#ifndef _ARG_PROC_H
#define _ARG_PROC_H

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct PwsSourcePackArgs
{
  char *index_list_path;       /* --index-list */
  char *output_dir;            /* --output-dir */
  char *gpu_worker_list;       /* --gpu-workers / -G */
  char *gpu_memory_mib_list;   /* --gpu-memory-mib / -M */
  int substack_size;           /* --substack-size / -B */
  size_t staged_group_limit;   /* --staged-group-limit / -H */
  char *progress_path;         /* --progress sidecar TSV */
} PwsSourcePackArgs;

void usage(void);
void ParsePwsSourcePackArgs(int argc, char **argv, PwsSourcePackArgs *args);

#endif /* _ARG_PROC_H */
