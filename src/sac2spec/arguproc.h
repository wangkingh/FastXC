#ifndef _ARGU_PROC_H
#define _ARGU_PROC_H
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include "config.h"

typedef struct ARGUTYPE
{
    char *sac_lst;
    char *spec_lst;
    char *filter_file;
    float seglen;
    float freq_low;
    float freq_high;
    int num_ch;
    int gpu_id;
    int whitenType;
    int normalizeType;
    int skip_steps[MAX_SKIP_STEPS_SIZE];
    int skip_step_count;
    int thread_num;
    int gpu_num;
} ARGUTYPE;

/* Parsing the input argument */
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);
void usage();
#endif