#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "arguproc.h"

int getopt();
extern char *optarg;
extern int optind;

void parse_arguments(int argc, char *argv[], CommandLineOptions *opts)
{
    int lag;
    while ((lag = getopt(argc, argv, "I:O:")) != -1)
    {
        switch (lag)
        {
        case 'I':
            strcpy(opts->infn, optarg);
            break;
        case 'O':
            strcpy(opts->outdir, optarg);
            break;
        default:
            fprintf(stderr, "Unsupported option\n");
            exit(1);
        }
    }
}
