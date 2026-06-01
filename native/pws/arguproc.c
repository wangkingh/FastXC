#include "arguproc.h"

#include "logger.h"

void ParsePwsSourcePackArgs(int argc, char **argv, PwsSourcePackArgs *args)
{
  args->index_list_path = NULL;
  args->output_dir = NULL;
  args->gpu_worker_list = (char *)"0";
  args->gpu_memory_mib_list = NULL;
  args->substack_size = 1;
  args->staged_group_limit = 200000;
  args->progress_path = NULL;

  if (argc <= 1)
  {
    usage();
    exit(EXIT_FAILURE);
  }

  int c;
  int option_index = 0;
  static struct option long_options[] = {
      {"progress", required_argument, 0, 1000},
      {"progress-file", required_argument, 0, 1000},
      {"index-list", required_argument, 0, 1001},
      {"sourcepack-list", required_argument, 0, 1001},
      {"output-dir", required_argument, 0, 1002},
      {"output-sourcepack", required_argument, 0, 1002},
      {"gpu-workers", required_argument, 0, 'G'},
      {"gpu-memory-mib", required_argument, 0, 'M'},
      {"substack-size", required_argument, 0, 'B'},
      {"staged-group-limit", required_argument, 0, 'H'},
      {"host-group-limit", required_argument, 0, 'H'},
      {0, 0, 0, 0},
  };

  while ((c = getopt_long(argc, argv, "G:B:H:M:", long_options, &option_index)) != -1)
  {
    switch (c)
    {
    case 'G':
      args->gpu_worker_list = optarg;
      if (strlen(args->gpu_worker_list) == 0)
      {
        LOG_ERROR("invalid_gpu_workers", "message=\"--gpu-workers/-G must not be empty\"");
        exit(EXIT_FAILURE);
      }
      break;
    case 'B':
      args->substack_size = atoi(optarg);
      if (args->substack_size < 2)
        args->substack_size = 1;
      break;
    case 'H':
      args->staged_group_limit = (size_t)strtoull(optarg, NULL, 10);
      if (args->staged_group_limit < 1)
      {
        LOG_ERROR("invalid_staged_group_limit", "value=%zu", args->staged_group_limit);
        exit(EXIT_FAILURE);
      }
      break;
    case 'M':
      args->gpu_memory_mib_list = optarg;
      if (strlen(args->gpu_memory_mib_list) == 0)
      {
        LOG_ERROR("invalid_gpu_memory_mib", "message=\"--gpu-memory-mib/-M must not be empty\"");
        exit(EXIT_FAILURE);
      }
      break;
    case 1000:
      args->progress_path = optarg;
      break;
    case 1001:
      args->index_list_path = optarg;
      break;
    case 1002:
      args->output_dir = optarg;
      break;
    default:
      LOG_ERROR("unknown_option", "option=-%c", optopt);
      usage();
      exit(EXIT_FAILURE);
    }
  }

  if (!args->index_list_path || !args->output_dir)
  {
    LOG_ERROR("missing_sourcepack_io",
              "index_list=\"%s\" output_dir=\"%s\"",
              args->index_list_path ? args->index_list_path : "",
              args->output_dir ? args->output_dir : "");
    usage();
    exit(EXIT_FAILURE);
  }

  LOG_DEBUG("pws_args_parsed",
            "index_list=\"%s\" output_dir=\"%s\" gpu_workers=\"%s\" gpu_memory_mib=\"%s\" substack_size=%d staged_group_limit=%zu progress=\"%s\"",
            args->index_list_path ? args->index_list_path : "",
            args->output_dir ? args->output_dir : "",
            args->gpu_worker_list ? args->gpu_worker_list : "",
            args->gpu_memory_mib_list ? args->gpu_memory_mib_list : "",
            args->substack_size,
            args->staged_group_limit,
            args->progress_path ? args->progress_path : "");
}

void usage(void)
{
  fprintf(stderr,
          "\nUsage:\n"
          "  ncf_pws --index-list <file> --output-dir <dir> [options]\n\n"
          "Required arguments:\n"
          "  --index-list <file>  Text file with one sourcepack_index.tsv path per line\n"
          "  --output-dir <dir>   Output directory for pws pack shards and sourcepack_index.tsv\n\n"
          "Optional arguments:\n"
          "  -G, --gpu-workers <list>      Comma-separated GPU worker list; duplicate IDs are allowed (default: 0)\n"
          "  -M, --gpu-memory-mib <list>   Comma-separated per-worker MiB caps; 0 means auto\n"
          "                                If provided, length must match --gpu-workers\n"
          "                                Auto budgets use 0.90 * free GPU memory / same-GPU worker count\n"
          "  -B, --substack-size <int>     Average every <int> traces before PWS\n"
          "                                Values < 2 disable sub-stack averaging (default: disabled)\n"
          "  -H, --staged-group-limit <int> Max staged host groups across workers (default: 200000)\n\n"
          "  --progress <file>             Write progress sidecar TSV\n\n"
          "Logging:\n"
          "  FASTXC_LOG_LEVEL=ERROR|WARN|INFO|DEBUG (default INFO)\n"
          "  PWS_LOG_LEVEL is honored when FASTXC_LOG_LEVEL is unset\n\n"
          "Version: PWS SourcePack-only mode\n");
}
