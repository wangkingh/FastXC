#ifndef _PATHLIST_H_
#define _PATHLIST_H_
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <libgen.h>
#include "in_out_node.h"
#include "config.h"
#include "util.h"

/* Read in a file which each line is a sac file */
PathNode *readPathList(const char *filename);

/* Create directory in a recursiveS way */
void createDirectoryRecursively(char *dir);

/* Create directory using a path list */
void createDirectories(PathNode *pathList);

/* Convert PathList to Array */
FilePathArray PathList2Array(PathNode *head);

#endif