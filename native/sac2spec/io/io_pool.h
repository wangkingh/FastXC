#ifndef SAC2SPEC_IO_POOL_H
#define SAC2SPEC_IO_POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IoPool IoPool;

typedef int (*IoPoolItemFn)(void *context, size_t index);

IoPool *IoPoolCreate(size_t num_threads, const char *kind_name);

int IoPoolRun(IoPool *pool, size_t item_count, int requested_threads,
              const char *phase, const char *start_event,
              const char *done_event, const char *failed_event,
              IoPoolItemFn item_fn, void *item_context);

void IoPoolDestroy(IoPool *pool);

#ifdef __cplusplus
}
#endif

#endif
