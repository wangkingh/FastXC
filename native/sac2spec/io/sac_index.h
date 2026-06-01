#ifndef _SAC_INDEX_H_
#define _SAC_INDEX_H_
#include "in_out_node.h"

typedef struct SacIndexPaths
{
    FilePathArray in_paths;
    SacIndexMetaArray meta;
} SacIndexPaths;

/* Read one strict TSV SAC index and generate input paths plus metadata. */
SacIndexPaths readSacIndexPaths(const char *index_path, int num_ch);

#endif
