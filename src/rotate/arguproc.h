#ifndef _ARGUPROC_H_
#define _ARGUPROC_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

typedef struct ARGUTYPE
{
    /* data */
    char *ee_in;
    char *en_in;
    char *ez_in;
    char *ne_in;
    char *nn_in;
    char *nz_in;
    char *ze_in;
    char *zn_in;
    char *zz_in;
    char *rr_out;
    char *rt_out;
    char *rz_out;
    char *tr_out;
    char *tt_out;
    char *tz_out;
    char *zr_out;
    char *zt_out;
    char *zz_out;
} ARGUTYPE;
char* my_strdup(const char* s);
void Argumentprocess(int argc, char **argv, ARGUTYPE *parg);
void usage();
#endif