#include "progress.hpp"
#include "logger.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

static int ProgressPathEnabled(const char *path)
{
    return path != NULL && path[0] != '\0' && strcmp(path, "NONE") != 0;
}

static void ProgressWriteLocked(ProgressState *progress, const char *status,
                                int gpu_id, size_t batch_rows,
                                size_t start_group, size_t group_count)
{
    if (progress == NULL || !progress->enabled)
    {
        return;
    }

    char tmp_path[PATH_MAX];
    int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", progress->path);
    if (n < 0 || n >= (int)sizeof(tmp_path))
    {
        LOG_WARN("progress_path_too_long", "path=\"%s\"", progress->path);
        progress->enabled = 0;
        return;
    }

    FILE *fp = fopen(tmp_path, "w");
    if (fp == NULL)
    {
        LOG_WARN("progress_write_failed", "path=\"%s\"", tmp_path);
        progress->enabled = 0;
        return;
    }

    fprintf(fp, "task\tstatus\tcompleted\ttotal\tunit\tdetail\n");
    fprintf(fp, "overall\t%s\t%zu\t%zu\tsac_rows\tgpu_id=%d last_batch_rows=%zu start_group=%zu group_count=%zu\n",
            status, progress->completed_rows, progress->total_rows,
            gpu_id, batch_rows, start_group, group_count);
    if (fclose(fp) != 0)
    {
        LOG_WARN("progress_close_failed", "path=\"%s\"", tmp_path);
        progress->enabled = 0;
        return;
    }
    if (rename(tmp_path, progress->path) != 0)
    {
        LOG_WARN("progress_rename_failed", "tmp=\"%s\" path=\"%s\"", tmp_path, progress->path);
        progress->enabled = 0;
        return;
    }
}

void ProgressInit(ProgressState *progress, const char *path, size_t total_rows)
{
    memset(progress, 0, sizeof(*progress));
    if (!ProgressPathEnabled(path))
    {
        return;
    }

    progress->path = path;
    progress->total_rows = total_rows;
    progress->completed_rows = 0;
    progress->enabled = 1;
    pthread_mutex_init(&progress->mutex, NULL);
    progress->initialized = 1;

    pthread_mutex_lock(&progress->mutex);
    ProgressWriteLocked(progress, "RUNNING", -1, 0, 0, 0);
    pthread_mutex_unlock(&progress->mutex);
}

void ProgressAdd(ProgressState *progress, size_t completed_rows,
                 int gpu_id, size_t start_group, size_t group_count)
{
    if (progress == NULL || !progress->enabled)
    {
        return;
    }

    pthread_mutex_lock(&progress->mutex);
    progress->completed_rows += completed_rows;
    if (progress->completed_rows > progress->total_rows)
    {
        progress->completed_rows = progress->total_rows;
    }
    ProgressWriteLocked(progress, "RUNNING", gpu_id, completed_rows,
                        start_group, group_count);
    pthread_mutex_unlock(&progress->mutex);
}

void ProgressFinish(ProgressState *progress, const char *status, int complete)
{
    if (progress == NULL || !progress->enabled)
    {
        return;
    }

    pthread_mutex_lock(&progress->mutex);
    if (complete)
    {
        progress->completed_rows = progress->total_rows;
    }
    ProgressWriteLocked(progress, status, -1, 0, 0, 0);
    pthread_mutex_unlock(&progress->mutex);
}

void ProgressDestroy(ProgressState *progress)
{
    if (progress != NULL && progress->initialized)
    {
        pthread_mutex_destroy(&progress->mutex);
        progress->initialized = 0;
        progress->enabled = 0;
    }
}
