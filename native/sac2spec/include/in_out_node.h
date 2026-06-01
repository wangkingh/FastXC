#ifndef _IN_OUT_NODE_H
#define _IN_OUT_NODE_H

#include "complex.h"
#include "config.h"
#include "sac.h"
#include <stdio.h>
#include <string.h>
#include "segspec.h"

typedef struct SacIndexMeta
{
    int gnsl_id;
    char timestamp[MAXNAME];
    char nsl_id[MAXNAME];
    char network[MAXNAME];
    char station[MAXNAME];
    char location[MAXNAME];
    char component[MAXNAME];
} SacIndexMeta;

typedef struct InOutNode
{
    char *sacpath;
    const SacIndexMeta *meta;

    float *sac_data;

    SACHEAD *sac_hd;

    complex *spectrum;

    int nstep;
    int nspec;
    float df;
    float dt;
} InOutNode;

static inline void FillSegspecHeaderFromNode(SEGSPEC *segspec_hd,
                                             const InOutNode *node)
{
    memset(segspec_hd, 0, sizeof(*segspec_hd));
    segspec_hd->stla = node->sac_hd->stla;
    segspec_hd->stlo = node->sac_hd->stlo;
    segspec_hd->nstep = node->nstep;
    segspec_hd->nspec = node->nspec;
    segspec_hd->df = node->df;
    segspec_hd->dt = node->dt;
    segspec_hd->gnsl_id = node->meta->gnsl_id;
}

typedef struct FilePathArray
{
    char **paths;
    int count;
} FilePathArray;

typedef struct SacIndexMetaArray
{
    SacIndexMeta *values;
    int count;
} SacIndexMetaArray;

#endif
