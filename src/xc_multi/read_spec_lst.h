#ifndef _READ_SPEC_LST_H
#define _READ_SPEC_LST_H

#include "config.h"
#include "node_util.h"
#include "segspec.h"
#include "read_segspec.h"
#include "util.h"
#include <string.h>
#include <dirent.h>
#include <stdlib.h>

FilePaths *read_spec_lst(char *spec_lst_file);

void freeFilePaths(FilePaths *fp);

#endif