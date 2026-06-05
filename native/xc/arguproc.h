#ifndef __ARG_PROC_H
#define __ARG_PROC_H

#include <stddef.h>

#define MAX_GPU_WORKERS 32

typedef struct ARGUTYPE
{
  char *input_path;            /* -I: stepack directory or TSV */
  char *ncf_dir;               /* -O output root */
  float cclength;              /* -C seconds */

  char *allowed_paths_file; /* -P <allowed_paths.tsv> */

  char *gpu_id_text; /* -G, comma-separated virtual worker placement. */
  size_t gpu_ids[MAX_GPU_WORKERS];
  size_t gpu_count;
  size_t gpu_id;    /* first GPU */

  char *gpu_mem_limit_text; /* -M, comma-separated per-worker VRAM caps in MiB */
  double gpu_mem_limit_mib[MAX_GPU_WORKERS];
  size_t gpu_mem_limit_count;
  int gpu_mem_limit_set;
  size_t lazy_write_depth; /* -J pending write batches per GPU worker */
  char *progress_file;     /* --progress sidecar TSV */
} ARGUTYPE;

void usage(void);
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg);

#endif
