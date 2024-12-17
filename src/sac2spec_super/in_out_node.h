#ifndef _IN_OUT_NODE_H
#define _IN_OUT_NODE_H
#include "sac.h"
#include <stdio.h>
#include "complex.h"
#include "segspec.h"

typedef struct InOutNode
{
    char *sacpath;

    char *specpath;

    float *sac_data;

    SACHEAD *sac_hd;
    SEGSPEC *segspec_hd;

    complex *spectrum;

    float nstep;
    float nspec;
    float df;
    float dt;
} InOutNode;

typedef struct PathNode
{
    char *path;
    struct PathNode *next;
} PathNode;

typedef struct FilePathArray
{
    char **paths;
    int count;
} FilePathArray;
#endif