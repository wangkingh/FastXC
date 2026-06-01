#ifndef _ARGU_PROC_H
#define _ARGU_PROC_H
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

typedef struct ARGUTYPE
{
    char *input_list;
    char *output_root;
    char *filter_file;
    float sac_len_sec;
    float seglen;
    float segshift;
    float xcorr_lag_sec;
    int num_ch;
    int gpu_ids[MAX_GPU_DEVICES];
    int gpu_worker_count;
    int whitenType;
    int outputPhaseOnly;
    int normalizeType;
    int sac_len_sec_set;
    int xcorr_lag_sec_set;
    int skip_steps[MAX_SKIP_STEPS_SIZE];
    int skip_step_count;
    int thread_num;
    int lazy_async;
    double gpu_ram_limits_mib[MAX_GPU_DEVICES];
    int gpu_ram_limit_count;
} ARGUTYPE;

/* Parsing the input argument */
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);
void usage();
#endif
