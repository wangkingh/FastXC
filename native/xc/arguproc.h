#ifndef __ARG_PROC_H
#define __ARG_PROC_H

#include <stddef.h>

#define MAX_GPU_WORKERS 32

typedef struct ARGUTYPE
{
  char *timestamp_index_path;  /* -I: xcspec_index.tsv */
  char *single_timestamp_path; /* --timestamp: one .xcspec shard */
  char *ncf_dir;               /* -O output root */
  float cclength;              /* -C seconds */

  char *allowed_paths_file; /* -P <allowed_paths.tsv> */

  char *gpu_id_text; /* -G, comma-separated virtual worker placement. */
  size_t gpu_ids[MAX_GPU_WORKERS];
  size_t gpu_count;
  size_t gpu_id;    /* first GPU */
  size_t cpu_count; /* -T writer threads for direct-output modes */

  char *gpu_mem_limit_text; /* -M, comma-separated per-worker VRAM caps in MiB */
  double gpu_mem_limit_mib[MAX_GPU_WORKERS];
  size_t gpu_mem_limit_count;
  int gpu_mem_limit_set;
  size_t lazy_write_depth; /* -J pending write batches per GPU worker */
  char *progress_file;     /* --progress sidecar TSV */

  int write_mode; /* --write-mode append, aggregate, or pack */
} ARGUTYPE;

void usage(void);
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg);

#endif
