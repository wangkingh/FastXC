#include "arguproc.h"
#include <errno.h>

static char *requireOptionValue(int argc, char **argv, int *idx, char opt)
{
    if (*idx + 1 >= argc)
    {
        fprintf(stderr, "Error: -%c requires a value\n", opt);
        usage();
        exit(1);
    }
    *idx += 1;
    return argv[*idx];
}

static char *requireLongOptionValue(int argc, char **argv, int *idx, const char *opt)
{
    if (*idx + 1 >= argc)
    {
        fprintf(stderr, "Error: %s requires a value\n", opt);
        usage();
        exit(1);
    }
    *idx += 1;
    return argv[*idx];
}

static int parseIntToken(const char *value, int *out);

static void parseSkipSteps(const char *value, ARGUTYPE *pargument)
{
    char buffer[MAXLINE];
    char *token;

    strncpy(buffer, value, MAXLINE - 1);
    buffer[MAXLINE - 1] = '\0';

    token = strtok(buffer, ",");
    pargument->skip_step_count = 0;
    while (token != NULL)
    {
        int val = 0;
        if (!parseIntToken(token, &val))
        {
            fprintf(stderr, "Error: Invalid skip step %s\n", token);
            exit(1);
        }
        if (val == -1)
        {
            break;
        }
        if (val < 0)
        {
            fprintf(stderr, "Error: Skip steps must be >= 0, or -1 as terminator\n");
            exit(1);
        }
        if (pargument->skip_step_count >= MAX_SKIP_STEPS_SIZE)
        {
            fprintf(stderr, "Error: Too many skip steps; max is %d\n", MAX_SKIP_STEPS_SIZE);
            exit(1);
        }
        pargument->skip_steps[pargument->skip_step_count++] = val;
        token = strtok(NULL, ",");
    }
}

static int parseFloatToken(const char *value, float *out)
{
    char *endptr = NULL;
    errno = 0;
    float parsed = strtof(value, &endptr);
    if (errno != 0 || endptr == value || *endptr != '\0')
    {
        return 0;
    }
    *out = parsed;
    return 1;
}

static int parseDoubleToken(const char *value, double *out)
{
    char *endptr = NULL;
    errno = 0;
    double parsed = strtod(value, &endptr);
    if (errno != 0 || endptr == value || *endptr != '\0')
    {
        return 0;
    }
    *out = parsed;
    return 1;
}

static int parseIntToken(const char *value, int *out)
{
    char *endptr = NULL;
    errno = 0;
    long parsed = strtol(value, &endptr, 10);
    if (errno != 0 || endptr == value || *endptr != '\0' ||
        parsed < -2147483647L || parsed > 2147483647L)
    {
        return 0;
    }
    *out = (int)parsed;
    return 1;
}

static void parseWindowSpec(const char *value, ARGUTYPE *pargument)
{
    char buffer[MAXLINE];
    char *sac_len;
    char *win_len;
    char *shift_len;
    char *xcorr_lag;
    char *extra;

    strncpy(buffer, value, MAXLINE - 1);
    buffer[MAXLINE - 1] = '\0';

    sac_len = strtok(buffer, "/");
    win_len = strtok(NULL, "/");
    shift_len = strtok(NULL, "/");
    xcorr_lag = strtok(NULL, "/");
    extra = strtok(NULL, "/");

    if (sac_len == NULL || win_len == NULL || shift_len == NULL || xcorr_lag == NULL || extra != NULL ||
        !parseFloatToken(sac_len, &pargument->sac_len_sec) ||
        !parseFloatToken(win_len, &pargument->seglen) ||
        !parseFloatToken(shift_len, &pargument->segshift) ||
        !parseFloatToken(xcorr_lag, &pargument->xcorr_lag_sec))
    {
        fprintf(stderr, "Error: -L expects sac_len/win_len/shift_len/xcorr_lag, e.g. 86400/7200/7200/100\n");
        exit(1);
    }
    pargument->sac_len_sec_set = 1;
    pargument->xcorr_lag_sec_set = 1;
}

static void parseWhitenSpec(const char *value, ARGUTYPE *pargument)
{
    char buffer[MAXLINE];
    char *whiten_type;
    char *phase_only;
    char *extra;

    strncpy(buffer, value, MAXLINE - 1);
    buffer[MAXLINE - 1] = '\0';

    whiten_type = strtok(buffer, "/");
    phase_only = strtok(NULL, "/");
    extra = strtok(NULL, "/");

    if (whiten_type == NULL || phase_only == NULL || extra != NULL ||
        !parseIntToken(whiten_type, &pargument->whitenType) ||
        !parseIntToken(phase_only, &pargument->outputPhaseOnly))
    {
        fprintf(stderr, "Error: -W expects whiten_type/output_phase_only, e.g. 2/1\n");
        exit(1);
    }
}

static void parseGpuList(const char *value, ARGUTYPE *pargument)
{
    char buffer[MAXLINE];
    char *token;

    strncpy(buffer, value, MAXLINE - 1);
    buffer[MAXLINE - 1] = '\0';

    token = strtok(buffer, ",");
    pargument->gpu_worker_count = 0;
    while (token != NULL && pargument->gpu_worker_count < MAX_GPU_DEVICES)
    {
        char *endptr = NULL;
        errno = 0;
        long gpu_id_long = strtol(token, &endptr, 10);
        if (errno != 0 || endptr == token || *endptr != '\0' ||
            gpu_id_long < 0 || gpu_id_long > 2147483647L)
        {
            fprintf(stderr, "Error: Invalid GPU id %s\n", token);
            exit(1);
        }
        int gpu_id = (int)gpu_id_long;
        pargument->gpu_ids[pargument->gpu_worker_count++] = gpu_id;
        token = strtok(NULL, ",");
    }

    if (token != NULL)
    {
        fprintf(stderr, "Error: Too many GPU ids; max is %d\n", MAX_GPU_DEVICES);
        exit(1);
    }
    if (pargument->gpu_worker_count < 1)
    {
        fprintf(stderr, "Error: -G requires at least one GPU id\n");
        exit(1);
    }
}

static void parseGpuMemoryList(const char *value, ARGUTYPE *pargument)
{
    char buffer[MAXLINE];
    char *token;

    strncpy(buffer, value, MAXLINE - 1);
    buffer[MAXLINE - 1] = '\0';

    token = strtok(buffer, ",");
    pargument->gpu_ram_limit_count = 0;
    while (token != NULL && pargument->gpu_ram_limit_count < MAX_GPU_DEVICES)
    {
        double limit_mib = 0.0;
        if (!parseDoubleToken(token, &limit_mib) || limit_mib < 0.0)
        {
            fprintf(stderr, "Error: Invalid GPU memory limit %s\n", token);
            exit(1);
        }
        pargument->gpu_ram_limits_mib[pargument->gpu_ram_limit_count++] = limit_mib;
        token = strtok(NULL, ",");
    }

    if (token != NULL)
    {
        fprintf(stderr, "Error: Too many GPU memory limits; max is %d\n", MAX_GPU_DEVICES);
        exit(1);
    }
    if (pargument->gpu_ram_limit_count < 1)
    {
        fprintf(stderr, "Error: -M requires at least one memory limit\n");
        exit(1);
    }
}

void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument)
{
    if (argc < 2)
    {
        usage();
        exit(1);
    }
    int argi;

    memset(pargument, 0, sizeof(ARGUTYPE));

    // Set default values for optional arguments
    pargument->whitenType = WHITEN_NONE;
    pargument->outputPhaseOnly = OUTPUT_KEEP_AMPLITUDE;
    pargument->normalizeType = NORMALIZE_NONE;
    pargument->skip_step_count = 0;
    pargument->thread_num = 1;
    pargument->segshift = 0.0f;
    pargument->sac_len_sec = 0.0f;
    pargument->sac_len_sec_set = 0;
    pargument->xcorr_lag_sec = 0.0f;
    pargument->xcorr_lag_sec_set = 0;
    pargument->num_ch = 3;
    pargument->gpu_worker_count = 1;
    pargument->gpu_ids[0] = 0;
    pargument->lazy_async = 1;
    pargument->gpu_ram_limit_count = 0;
    int debug_requested = 0;

    // Parse simple short options without a platform-specific parser.
    for (argi = 1; argi < argc; argi++)
    {
        char *value;
        char opt;
        if (strcmp(argv[argi], "-h") == 0 || strcmp(argv[argi], "--help") == 0)
        {
            usage();
            exit(0);
        }

        if (strncmp(argv[argi], "--", 2) == 0)
        {
            const char *opt = argv[argi];
            if (strcmp(opt, "--normalize") == 0 ||
                     strcmp(opt, "--normalize-type") == 0)
            {
                pargument->normalizeType = atoi(requireLongOptionValue(argc, argv, &argi, opt));
            }
            else if (strcmp(opt, "--debug") == 0)
            {
                debug_requested = 1;
            }
            else
            {
                fprintf(stderr, "Error: Unknown option %s\n", opt);
                usage();
                exit(1);
            }
            continue;
        }

        if (argv[argi][0] != '-' || argv[argi][1] == '\0' || argv[argi][2] != '\0')
        {
            fprintf(stderr, "Error: Invalid option %s\n", argv[argi]);
            usage();
            exit(1);
        }

        opt = argv[argi][1];
        switch (opt)
        {
        case 'I':
            value = requireOptionValue(argc, argv, &argi, opt);
            pargument->input_list = value;
            break;
        case 'O':
            value = requireOptionValue(argc, argv, &argi, opt);
            pargument->output_root = value;
            break;
        case 'C':
            value = requireOptionValue(argc, argv, &argi, opt);
            pargument->num_ch = atoi(value);
            break;
        case 'B':
            value = requireOptionValue(argc, argv, &argi, opt);
            pargument->filter_file = value;
            break;
        case 'L':
            value = requireOptionValue(argc, argv, &argi, opt);
            parseWindowSpec(value, pargument);
            break;

        case 'G':
            value = requireOptionValue(argc, argv, &argi, opt);
            parseGpuList(value, pargument);
            break;
        case 'W':
            value = requireOptionValue(argc, argv, &argi, opt);
            parseWhitenSpec(value, pargument);
            break;
        case 'N':
            value = requireOptionValue(argc, argv, &argi, opt);
            pargument->normalizeType = atoi(value);
            break;
        case 'Q':
            value = requireOptionValue(argc, argv, &argi, opt);
            parseSkipSteps(value, pargument);
            break;
        case 'T':
            value = requireOptionValue(argc, argv, &argi, opt);
            pargument->thread_num = atoi(value);
            break;
        case 'M':
            value = requireOptionValue(argc, argv, &argi, opt);
            parseGpuMemoryList(value, pargument);
            break;
        default:
            fprintf(stderr, "Error: Unknown option -%c\n", opt);
            usage();
            exit(1);
        }
    }

    if (debug_requested)
    {
        pargument->lazy_async = 0;
    }

    if (pargument->input_list == NULL ||
        pargument->output_root == NULL ||
        pargument->filter_file == NULL ||
        !pargument->sac_len_sec_set ||
        pargument->seglen <= 0.0f ||
        !pargument->xcorr_lag_sec_set)
    {
        if (!pargument->sac_len_sec_set || !pargument->xcorr_lag_sec_set)
        {
            fprintf(stderr, "Error: -L sac_len/win_len/shift_len/xcorr_lag is required\n");
        }
        usage();
        exit(1);
    }

    if (pargument->segshift <= 0.0f)
    {
        fprintf(stderr, "Error: -L shift_len must be > 0\n");
        exit(1);
    }

    if (pargument->sac_len_sec <= 0.0f)
    {
        fprintf(stderr, "Error: -L sac_len must be > 0\n");
        exit(1);
    }

    if (pargument->num_ch < 1)
    {
        fprintf(stderr, "Error: -C must be >= 1\n");
        exit(1);
    }

    if (pargument->thread_num < 1)
    {
        fprintf(stderr, "Error: -T must be >= 1\n");
        exit(1);
    }

    if (pargument->gpu_ram_limit_count > 0 &&
        pargument->gpu_ram_limit_count != pargument->gpu_worker_count)
    {
        fprintf(stderr,
                "Error: -M count must match -G count; got %d memory limits for %d virtual GPU workers\n",
                pargument->gpu_ram_limit_count, pargument->gpu_worker_count);
        exit(1);
    }

    if (pargument->xcorr_lag_sec_set && pargument->xcorr_lag_sec < 0.0f)
    {
        fprintf(stderr, "Error: -L xcorr_lag must be >= 0\n");
        exit(1);
    }

    if (pargument->whitenType < WHITEN_NONE ||
        pargument->whitenType > WHITEN_BEFORE_AND_AFTER)
    {
        fprintf(stderr, "Error: -W whiten_type must be 0, 1, 2, or 3\n");
        exit(1);
    }

    if (pargument->outputPhaseOnly != OUTPUT_KEEP_AMPLITUDE &&
        pargument->outputPhaseOnly != OUTPUT_PHASE_ONLY)
    {
        fprintf(stderr, "Error: -W output_phase_only must be 0 or 1\n");
        exit(1);
    }
}

/* display usage */
void usage(void)
{
    fprintf(stderr,
        "Usage:  sac2spec  -I <input_list> -O <output_root> -L <file/window/shift/ncf_lag> [options]\n"
        "\n"
        "Required arguments\n"
        "  -I FILE            Input SAC index TSV\n"
        "  -O DIR             Output workspace; creates DIR/stepack and DIR/progress\n"
        "  -L A/B/C/D         SAC file length, segment window, segment shift, and NCF lag in seconds\n"
        "                     SAC data is truncated or zero-padded to A using the checked SAC delta\n"
        "  -B FILE            Butterworth filter-coefficient file; first band is the broad band\n"
        "\n"
        "Generated files\n"
        "                     Progress is written to <output_root>/progress/sac2spec_progress.tsv\n"
        "                     Writes one pitched stepack file and TSV sidecar per worker batch\n"
        "\n"
        "Processing options\n"
        "  -C INT             Number of channels per NSL/timestamp group (default: 3)\n"
        "  -G LIST            Comma-separated physical GPU ids for virtual workers, e.g. 0,1,2 or 0,0\n"
        "  -T INT             Total CPU I/O threads across GPU workers     (default: 1)\n"
        "  -M LIST            Optional comma-separated MiB caps matching -G; omitted or 0 means auto\n"
        "  --debug            Disable lazy async for easier native debugging\n"
        "  -W A/B             Whitening strategy and output phase-only flag (default: 0/0)\n"
        "                       0  none\n"
        "                       1  before time-domain normalisation\n"
        "                       2  after  time-domain normalisation\n"
        "                       3  both before and after normalisation\n"
        "                     Output mode B:\n"
        "                       0  keep output spectral amplitudes\n"
        "                       1  write phase-only output spectrum\n"
        "  -N INT             Time-domain normalisation type (default: 0)\n"
        "  --normalize INT    Alias for normalisation type\n"
        "                       0  none\n"
        "                       1  multi-frequency run-abs\n"
        "                       2  one-bit\n"
        "                       3  run-abs\n"
        "  -Q LIST            Comma-separated list of *segment indices* to skip, e.g. \"3,7,9\";\n"
        "                     terminate with -1. Empty list -> process all segments.\n"
        "  -h                 Show this help and exit\n"
        "\n"
        "Example\n"
        "  sac2spec -I sac_index.tsv -O output -L 86400/7200/7200/100 -C 3 \\\n"
        "          -W 1/0 -N 1 -B filter.txt -G 0,1,2,3 -T 8 -M 8192,8192,8192,8192 -Q 5,10,-1\n"
        "  sac2spec -I sac_index.tsv -O output -L 86400/7200/7200/100 -C 3 \\\n"
        "          -W 1/0 -N 1 -B filter.txt -G 0,0 -T 2 -M 4096,4096\n"
        "\n"
        "Stepack path: <output_root>/stepack/w<worker_id>.b<batch_seq>.stepack and matching .tsv\n"
        "Stepack file: batch header + NSLC table + pitched [step][batch_nslc][freq] complex payload\n");
}
