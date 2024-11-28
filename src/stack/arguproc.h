#ifndef _ARGUPROC_H_
#define _ARGUPROC_H_

#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
typedef struct ARGUTYPE
{
    /* used by -L */
    char *sac_lst;
    char *out_sac;
    /* used by -A */
    /* take the normlized version as default. by wangwt@20130927 */
    int normalize;
} ARGUTYPE;

void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);
void usage();
#endif