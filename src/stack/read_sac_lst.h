#ifndef _FILE2LST_H
#define _FILE2LST_H
#include "sac.h"
#include "util.h"
#include <ctype.h>
#include <libgen.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#define MAXLINE 8192
#define MAXPATH 8192
#define MAXNAME 255

/* calculate dist gcarc az baz */
typedef struct FilePaths
{
    char **paths;
    int count;
} FilePaths;

char* my_strdup(const char* s);

FilePaths *read_sac_lst(char *dir);

FilePaths *filter_by_npts(FilePaths *input);

void freeFilePaths(FilePaths *fp);

#endif