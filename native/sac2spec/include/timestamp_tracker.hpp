#ifndef SAC2SPEC_TIMESTAMP_TRACKER_HPP
#define SAC2SPEC_TIMESTAMP_TRACKER_HPP

#include <cstddef>
#include <pthread.h>

extern "C"
{
#include "in_out_node.h"
}

typedef struct TimestampTrackerEntry
{
    char timestamp[MAXNAME];
    size_t start_group;
    size_t group_count;
    size_t done_group_count;
    int success_written;
} TimestampTrackerEntry;

typedef struct TimestampTracker
{
    char *root;
    TimestampTrackerEntry *entries;
    int *group_to_entry;
    size_t entry_count;
    size_t total_groups;
    int num_ch;
    pthread_mutex_t mutex;
    int mutex_initialized;
} TimestampTracker;

int TimestampTrackerInit(TimestampTracker *tracker,
                         const SacIndexMetaArray *meta,
                         int num_ch,
                         size_t total_groups,
                         const char *spack_root);
int TimestampTrackerMarkBatchDone(TimestampTracker *tracker,
                                  size_t start_group,
                                  size_t group_count);
void TimestampTrackerDestroy(TimestampTracker *tracker);

#endif
