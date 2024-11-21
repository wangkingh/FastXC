#include "arguproc.h"

void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument)
{
    if (argc < 2)
    {
        usage();
        exit(1);
    }
    int opt;

    // Set default values for optional arguments
    pargument->whitenType = 0;
    pargument->normalizeType = 0;
    pargument->skip_step_count = 0;
    pargument->thread_num = 1;

    // Parse command line arguments using getopt
    while ((opt = getopt(argc, argv, "I:O:C:B:L:G:F:W:N:Q:T:U:")) != -1)
    {
        switch (opt)
        {
        case 'I':
            pargument->sac_lst = optarg;
            break;
        case 'O':
            pargument->spec_lst = optarg;
            break;
        case 'C':
            pargument->num_ch = atoi(optarg);
            break;
        case 'B':
            pargument->filter_file = optarg;
            break;
        case 'L':
            pargument->seglen = atof(optarg);
            break;
        case 'G':
            pargument->gpu_id = atoi(optarg);
            break;
        case 'F':
        {
            float freq_low, freq_high;
            if (sscanf(optarg, "%f/%f", &freq_low, &freq_high) != 2)
            {
                fprintf(stderr, "Error: Invalid frequency band format\n");
                exit(1);
            }

            // ensure freq_low_limit < freq_high_limit
            if (freq_low >= freq_high)
            {
                fprintf(stderr, "Error: Invalid frequency band range\n");
                exit(1);
            }
            pargument->freq_low = freq_low;
            pargument->freq_high = freq_high;
            break;
        }
        case 'W':
            pargument->whitenType = atoi(optarg);
            break;
        case 'N':
            pargument->normalizeType = atoi(optarg);
            break;
        case 'Q':
        {
            char *token = strtok(optarg, "/");
            pargument->skip_step_count = 0;
            while (token != NULL && pargument->skip_step_count + 1 < MAX_SKIP_STEPS_SIZE)
            {
                int val = atoi(token);
                if (val == -1)
                {
                    break; // stop parsing when find -1
                }
                pargument->skip_steps[pargument->skip_step_count++] = val;
                token = strtok(NULL, "/");
            }
            break;
        }
        case 'U':
            pargument->gpu_num = atoi(optarg);
        case 'T':
            pargument->thread_num = atoi(optarg);
            break;
        default: // '?' or ':' for unrecognized options or missing option arguments
            usage();
            exit(-1);
        }
    }
}

/* display usage */
void usage()
{
    fprintf(stderr, "Usage:\n"
                    "sac2spec_c1   -I filelist -O outputlist -L winlength [-G gpu_no] \n"
                    "              -W whitenType -N normalizeType -F f1/f2 -B filterfile \n"
                    "               [-T num_of_gpu_para] [-Q skip step]\n"
                    " Options:\n"
                    "   -I   input list of sac files of multi/single components.\n"
                    "		 Number of lines represent different component of same date/station should be continuous.\n"
                    "		 of different components with same staion name and date.\n"
                    "    	 Also, the input KCMPNM should be set right.The input files must have the same\n"
                    "     	 byte-order, little-endian is suggested, and the same NPTS\n"
                    "   -O   output spec list of segment spectrum files. Corresponding to -I option\n"
                    "   -L   length of segment window in seconds. Usually I use 7200s 2 hour\n"
                    "   -G   Index of Gpu device \n"
                    "   -F   FREQ_BANDS Frequency bands for spectral whitenning (in Hz) \n"
                    "        using the format f_low_limit/f_high_limit \n"
                    "   -W   Whiten type.\n"
                    "		   0: donot do any whitenning or normalizaion\n"
                    "		   1: do bandwhitening before time domain normalization\n"
                    "		   2: do bandwhitening after time domain normalization\n"
                    "         Set as 3: do bandwhitenning before and after time domain normalization\n"
                    "   -N    Normalization type\n"
                    "          0: no normalization.\n"
                    "          1: runabs normalization.\n"
                    "          2: one-bit normalization.\n"
                    "   -B  Butterworth filter file. The filter file should be generated \n"
                    "   -Q   skip segment step. Default is 1, which means we do not skip any step\n"
                    "   -T  [optional >=0] GPU thread number. Default is 1\n"
                    "   Last updated by wangjx@20240108\n");
}