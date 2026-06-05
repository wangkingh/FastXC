#include "arguproc.h"
#include "logger.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static int is_option(const char *text)
{
  return text && text[0] == '-' && text[1] != '\0';
}

static char *need_value(int argc, char **argv, int *idx, const char *opt)
{
  if (*idx + 1 >= argc || is_option(argv[*idx + 1]))
  {
    LOG_ERROR("argument_value_missing", "option=\"%s\"", opt);
    usage();
    exit(1);
  }
  (*idx)++;
  return argv[*idx];
}

static void parse_gpu_ids(ARGUTYPE *parg)
{
  char *copy = NULL;
  char *token = NULL;
  size_t count = 0;

  if (!parg->gpu_id_text || parg->gpu_id_text[0] == '\0')
    parg->gpu_id_text = (char *)"0";

  copy = (char *)malloc(strlen(parg->gpu_id_text) + 1);
  if (!copy)
  {
    LOG_ERROR("gpu_list_parse_oom", "option=\"-G\"");
    exit(1);
  }
  strcpy(copy, parg->gpu_id_text);

  token = copy;
  while (token)
  {
    char *next = strchr(token, ',');
    char *endptr = NULL;
    long parsed = 0;
    if (next)
      *next = '\0';
    if (token[0] == '\0')
    {
      LOG_ERROR("gpu_list_entry_empty", "gpu_list=\"%s\"", parg->gpu_id_text);
      free(copy);
      exit(1);
    }
    if (count >= MAX_GPU_WORKERS)
    {
      LOG_ERROR("gpu_worker_count_exceeded", "max=%d", MAX_GPU_WORKERS);
      free(copy);
      exit(1);
    }
    errno = 0;
    parsed = strtol(token, &endptr, 10);
    if (errno != 0 || endptr == token || *endptr != '\0' ||
        parsed < 0 || parsed > INT_MAX)
    {
      LOG_ERROR("gpu_list_entry_invalid", "entry=\"%s\"", token);
      free(copy);
      exit(1);
    }
    parg->gpu_ids[count++] = (size_t)parsed;
    token = next ? next + 1 : NULL;
  }

  if (count == 0)
  {
    LOG_ERROR("gpu_list_empty", "option=\"-G\"");
    free(copy);
    exit(1);
  }

  parg->gpu_count = count;
  parg->gpu_id = parg->gpu_ids[0];
  free(copy);
}

static int parse_double_token(const char *text, double *out)
{
  char *endptr = NULL;
  double parsed = 0.0;
  errno = 0;
  parsed = strtod(text, &endptr);
  if (errno != 0 || endptr == text || *endptr != '\0')
    return 0;
  *out = parsed;
  return 1;
}

static void parse_gpu_memory_limits(ARGUTYPE *parg)
{
  char *copy = NULL;
  char *token = NULL;
  size_t count = 0;

  if (!parg->gpu_mem_limit_text)
  {
    parg->gpu_mem_limit_count = parg->gpu_count;
    for (size_t i = 0; i < parg->gpu_count; ++i)
      parg->gpu_mem_limit_mib[i] = 0.0;
    return;
  }

  copy = (char *)malloc(strlen(parg->gpu_mem_limit_text) + 1);
  if (!copy)
  {
    LOG_ERROR("gpu_memory_list_parse_oom", "option=\"-M\"");
    exit(1);
  }
  strcpy(copy, parg->gpu_mem_limit_text);

  token = copy;
  while (token)
  {
    char *next = strchr(token, ',');
    double value = 0.0;
    if (next)
      *next = '\0';
    if (token[0] == '\0')
    {
      LOG_ERROR("gpu_memory_list_entry_empty", "memory_list=\"%s\"",
                parg->gpu_mem_limit_text);
      free(copy);
      exit(1);
    }
    if (count >= MAX_GPU_WORKERS)
    {
      LOG_ERROR("gpu_memory_entry_count_exceeded", "max=%d", MAX_GPU_WORKERS);
      free(copy);
      exit(1);
    }
    if (!parse_double_token(token, &value) || value < 0.0)
    {
      LOG_ERROR("gpu_memory_entry_invalid", "entry=\"%s\"", token);
      free(copy);
      exit(1);
    }
    parg->gpu_mem_limit_mib[count++] = value;
    token = next ? next + 1 : NULL;
  }

  free(copy);
  if (count != parg->gpu_count)
  {
    LOG_ERROR("gpu_memory_entry_count_mismatch",
              "memory_entries=%zu gpu_workers=%zu", count, parg->gpu_count);
    exit(1);
  }
  parg->gpu_mem_limit_count = count;
  parg->gpu_mem_limit_set = 1;
}

void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg)
{
  memset(parg, 0, sizeof(*parg));
  parg->gpu_id_text = (char *)"0";
  parg->gpu_mem_limit_text = NULL;
  parg->gpu_mem_limit_count = 0;
  parg->gpu_mem_limit_set = 0;
  parg->lazy_write_depth = 1;
  parg->progress_file = NULL;

  if (argc <= 1)
  {
    usage();
    exit(1);
  }

  int debug_requested = 0;
  for (int i = 1; i < argc; ++i)
  {
    const char *opt = argv[i];
    if (strcmp(opt, "-h") == 0 || strcmp(opt, "--help") == 0)
    {
      usage();
      exit(0);
    }
    else if (strcmp(opt, "-I") == 0)
      parg->input_path = need_value(argc, argv, &i, opt);
    else if (strcmp(opt, "-O") == 0)
      parg->ncf_dir = need_value(argc, argv, &i, opt);
    else if (strcmp(opt, "-C") == 0)
      parg->cclength = (float)atof(need_value(argc, argv, &i, opt));
    else if (strcmp(opt, "-G") == 0)
      parg->gpu_id_text = need_value(argc, argv, &i, opt);
    else if (strcmp(opt, "-M") == 0)
      parg->gpu_mem_limit_text = need_value(argc, argv, &i, opt);
    else if (strcmp(opt, "-P") == 0)
      parg->allowed_paths_file = need_value(argc, argv, &i, opt);
    else if (strcmp(opt, "-J") == 0)
      parg->lazy_write_depth = (size_t)atoi(need_value(argc, argv, &i, opt));
    else if (strcmp(opt, "--progress") == 0 || strcmp(opt, "--progress-file") == 0)
      parg->progress_file = need_value(argc, argv, &i, opt);
    else if (strcmp(opt, "--debug") == 0)
      debug_requested = 1;
    else
    {
      LOG_ERROR("argument_unknown", "option=\"%s\"", opt);
      usage();
      exit(1);
    }
  }

  parse_gpu_ids(parg);
  parse_gpu_memory_limits(parg);

  if (!parg->ncf_dir || parg->cclength <= 0.0f || !parg->allowed_paths_file)
  {
    LOG_ERROR("required_argument_missing", "required=\"-O,-C,-P\"");
    usage();
    exit(1);
  }

  if (!parg->input_path)
  {
    LOG_ERROR("timestamp_input_missing", "required=\"-I\"");
    usage();
    exit(1);
  }

  if (debug_requested)
    parg->lazy_write_depth = 0;
}

void usage(void)
{
  fprintf(stderr,
          "\nUsage:\n"
          "  xc_fast -I <stepack_dir|stepack.tsv> -P allowed_paths.tsv -O out -C sec [options]\n\n"
          "Input:\n"
          "  -I <path>            Stepack directory or stepack TSV sidecar\n"
          "  -P <file>            Canonical allowed path table\n"
          "  -O <dir>             Output root\n"
          "  -C <sec>             Half CC length\n\n"
          "Runtime:\n"
          "  -G <ids>             GPU worker list, e.g. 0 or 0,1 or 0,0\n"
          "  -M <MiB,...>         Per-worker VRAM caps matching -G; 0 means auto\n"
          "                       Auto budget = 0.90*free/workers_on_same_gpu\n"
          "                       Host RAM auto budget = 0.80*MemAvailable/workers\n"
          "  -J <num>             Lazy write queue depth per worker (default 1, 0 sync)\n"
          "  --debug              Disable lazy write for easier native debugging\n"
          "  --progress <file>    Write progress sidecar TSV\n\n"
          "Logging:\n"
          "  FASTXC_LOG_LEVEL=ERROR|WARN|INFO|DEBUG (default INFO)\n"
          "  XC_LOG_LEVEL is honored when FASTXC_LOG_LEVEL is unset\n"
          "  SAC2SPEC_LOG_LEVEL remains accepted as a legacy fallback\n\n"
          "Output:\n"
          "  Writes <output>/xcpack/*.xcpack + *.tsv sidecars.\n\n");
}
