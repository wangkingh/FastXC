#include "arguproc.h"
#include "logger.h"

void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg)
{
  parg->sourcepack_list = NULL;
  parg->output_sourcepack = NULL;
  parg->gpu_list = (char *)"0";
  parg->gpu_ram_limit_mib_list = NULL;
  parg->band_limited = 0;
  parg->band_fmin = 0.0;
  parg->band_fmax = 0.0;
  parg->band_taper_hz = -1.0;
  parg->sub_stack_size = 1;
  parg->progress_file = NULL;

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
      {"input-bigsac", required_argument, 0, 1001},
      {"output-sac", required_argument, 0, 1002},
      {"sourcepack-list", required_argument, 0, 1003},
      {"output-sourcepack", required_argument, 0, 1004},
      {0, 0, 0, 0},
  };
  while ((c = getopt_long(argc, argv, "I:O:G:S:B:M:F:T:", long_options, &option_index)) != -1)
  {
    switch (c)
    {
    case 'I':
      parg->sourcepack_list = optarg;
      break;

    case 'O':
      parg->output_sourcepack = optarg;
      break;

    case 'G':
      parg->gpu_list = optarg;
      if (strlen(parg->gpu_list) == 0)
      {
        LOG_ERROR("invalid_gpu_list", "message=\"-G must not be empty\"");
        exit(EXIT_FAILURE);
      }
      break;

    case 'S':
      if (strlen(optarg) != 3 || strspn(optarg, "01") != 3)
      {
        LOG_ERROR("invalid_stack_selector",
                  "value=\"%s\" expected=\"3-digit binary string, e.g. 001\"",
                  optarg);
        exit(EXIT_FAILURE);
      }
      if (optarg[2] != '1')
      {
        LOG_ERROR("invalid_stack_selector",
                  "value=\"%s\" message=\"TF-PWS bit must be 1\"",
                  optarg);
        exit(EXIT_FAILURE);
      }
      if (optarg[0] == '1' || optarg[1] == '1')
      {
        LOG_WARN("ignored_stack_selector_bits",
                 "value=\"%s\" message=\"linear/PWS -S bits are ignored by TF-PWS\"",
                 optarg);
      }
      break;

    case 'M':
      parg->gpu_ram_limit_mib_list = optarg;
      break;

    case 'F':
      if (sscanf(optarg, "%lf/%lf", &parg->band_fmin, &parg->band_fmax) != 2 ||
          parg->band_fmin < 0.0 ||
          parg->band_fmax <= parg->band_fmin)
      {
        LOG_ERROR("invalid_band_argument",
                  "value=\"%s\" expected=\"fmin/fmax in Hz, e.g. 0.02/0.20\"",
                  optarg);
        exit(EXIT_FAILURE);
      }
      parg->band_limited = 1;
      break;

    case 'T':
      parg->band_taper_hz = atof(optarg);
      if (parg->band_taper_hz < 0.0)
      {
        LOG_ERROR("invalid_band_taper",
                  "value=\"%s\"",
                  optarg);
        exit(EXIT_FAILURE);
      }
      break;

    case 'B':
      parg->sub_stack_size = atoi(optarg);
      if (parg->sub_stack_size < 2)
      {
        parg->sub_stack_size = 1;
      }
      break;

    case 1000:
      parg->progress_file = optarg;
      break;
    case 1001:
    case 1002:
      LOG_ERROR("legacy_io_removed",
                "message=\"--input-bigsac/--output-sac were removed; use -I/--sourcepack-list and -O/--output-sourcepack\"");
      exit(EXIT_FAILURE);
    case 1003:
      parg->sourcepack_list = optarg;
      break;
    case 1004:
      parg->output_sourcepack = optarg;
      break;

    default:
      LOG_ERROR("unknown_option", "option=-%c", optopt);
      usage();
      exit(EXIT_FAILURE);
    }
  }

  int has_sourcepack = parg->sourcepack_list && parg->output_sourcepack;
  if (!has_sourcepack)
  {
    LOG_ERROR("missing_sourcepack_io",
              "message=\"use -I/-O or --sourcepack-list/--output-sourcepack\"");
    usage();
    exit(EXIT_FAILURE);
  }
  if ((parg->sourcepack_list && !parg->output_sourcepack) ||
      (!parg->sourcepack_list && parg->output_sourcepack))
  {
    LOG_ERROR("partial_sourcepack_io",
              "sourcepack_list=%s output_sourcepack=%s",
              parg->sourcepack_list ? "set" : "missing",
              parg->output_sourcepack ? "set" : "missing");
    exit(EXIT_FAILURE);
  }
}

void usage(void)
{
  fprintf(stderr,
          "\nUsage:\n"
          "  ncf_tfpws -I <sourcepack_inputs.txt> -O <output_dir> [options]\n\n"
          "Required arguments:\n"
          "  -I <file>   Text file with one sourcepack_index.tsv path per line\n"
          "              Alias: --sourcepack-list <file>\n"
          "  -O <dir>    Output directory for TF-PWS SourcePack\n"
          "              Alias: --output-sourcepack <dir>\n\n"
          "Optional arguments:\n"
          "  -G <list>   Comma-separated GPU worker IDs; repeats share one physical GPU\n"
          "              e.g. 0 or 0,1,2,3 or 0,0\n"
          "  -S <bin>    Deprecated compatibility flag; TF-PWS bit must be 1\n"
          "              e.g. 001 (linear/PWS bits are ignored)\n"
          "  -M <list>   Optional comma-separated MiB caps, one per -G worker\n"
          "              Applies to GPU arrays, shared cuFFT workspace, and host staging\n"
          "              0 means auto; auto is 0.90*free divided by workers on that GPU\n"
          "  -F <lo/hi>  Only apply TF-PWS weighting inside this Hz band\n"
          "  -T <Hz>     Raised-cosine taper width inside -F band\n"
          "              Default: 5%% of the requested band width; use 0 for hard edges\n"
          "  -B <int>    Pre-stack every <int> traces before TF-PWS\n"
          "              Set < 2 to disable pre-stack grouping (default: disabled)\n\n"
          "  --progress <file> Write progress sidecar TSV\n\n"
          "Logging:\n"
          "  FASTXC_LOG_LEVEL=ERROR|WARN|INFO|DEBUG (default INFO)\n"
          "  TFPWS_LOG_LEVEL is honored when FASTXC_LOG_LEVEL is unset\n"
          "  SAC2SPEC_LOG_LEVEL remains accepted as a legacy fallback\n\n"
          "Version: tfpws SourcePack-only mode 2026-05-30\n");
}
