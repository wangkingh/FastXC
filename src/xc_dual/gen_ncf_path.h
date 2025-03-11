#ifndef _GEN_NCF_PATH_H
#define _GEN_NCF_PATH_H
#define MAXLINE 8192
#define MAXPATH 8192
#define MAXNAME 255
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>
#include "sac.h"
#include "segspec.h"
#include "cal_dist.h"

char *my_strdup(const char *s);

char *my_strtok(char *str, const char *delim, char **saveptr);

void CreateDir(char *sPathName);

void SplitFileName(const char *fname, const char *delimiter, char *stastr,
                   char *yearstr, char *jdaystr, char *hmstr, char *chnstr);

void SacheadProcess(SACHEAD *ncfhd, SEGSPEC *srchd, SEGSPEC *stahd, float cclength);

void GenCCFPath(char *ccf_path, char *src_path, char *sta_path, char *output_dir);

char *GetNcfPath(char *src_path, char *sta_path, char *output_dir);

char *GetEachNcfPath(char *src_path, char *sta_path, char *output_dir);

#endif