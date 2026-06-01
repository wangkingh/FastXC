#ifndef PARALLEL_IO_H
#define PARALLEL_IO_H

#include <stddef.h>

#include "in_out_node.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct ThreadPoolRead ThreadPoolRead;

ThreadPoolRead *CreateReadIoPool(size_t num_threads);

int ReadSacBatchParallel(ThreadPoolRead *pool, size_t item_count,
                         InOutNode *items, int num_threads,
                         int target_npts, float expected_delta);

void DestroyReadIoPool(ThreadPoolRead *pool);

#ifdef __cplusplus
}
#endif

#endif
