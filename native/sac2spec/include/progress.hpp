#ifndef SAC2SPEC_PROGRESS_HPP
#define SAC2SPEC_PROGRESS_HPP

#include <cstddef>
#include <pthread.h>

typedef struct ProgressState
{
    const char *path;
    size_t total_rows;
    size_t completed_rows;
    int enabled;
    int initialized;
    pthread_mutex_t mutex;
} ProgressState;

void ProgressInit(ProgressState *progress, const char *path, size_t total_rows);
void ProgressAdd(ProgressState *progress, size_t completed_rows,
                 int gpu_id, size_t start_group, size_t group_count);
void ProgressFinish(ProgressState *progress, const char *status, int complete);
void ProgressDestroy(ProgressState *progress);

#endif
