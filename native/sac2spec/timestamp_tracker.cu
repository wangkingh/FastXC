#include "include/timestamp_tracker.hpp"

#include "include/logger.h"
#include "include/path_utils.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int writeTimestampSuccess(const TimestampTracker *tracker,
                                 const TimestampTrackerEntry *entry)
{
    char leaf[MAXNAME];
    char *timestamp_dir;
    char *success_path;
    FILE *fp;

    PathSafeTimestampLeaf(leaf, sizeof(leaf), entry->timestamp);
    timestamp_dir = PathJoinAlloc(tracker->root, leaf);
    if (timestamp_dir == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=timestamp_success_dir");
        return -1;
    }
    if (PathMakeDirectoryRecursive(timestamp_dir) != 0)
    {
        free(timestamp_dir);
        return -1;
    }
    success_path = PathJoinAlloc(timestamp_dir, "_SUCCESS");
    free(timestamp_dir);
    if (success_path == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=timestamp_success_path");
        return -1;
    }

    fp = fopen(success_path, "wb");
    if (fp == NULL)
    {
        LOG_ERROR("write_timestamp_success_failed",
                  "path=\"%s\" errno=%d", success_path, errno);
        free(success_path);
        return -1;
    }
    fprintf(fp, "timestamp\t%s\ngroups\t%zu\n",
            entry->timestamp, entry->group_count);
    if (fclose(fp) != 0)
    {
        LOG_ERROR("close_timestamp_success_failed", "path=\"%s\"", success_path);
        free(success_path);
        return -1;
    }
    LOG_INFO("timestamp_stepack_done",
             "timestamp=%s groups=%zu path=\"%s\"",
             entry->timestamp, entry->group_count, success_path);
    free(success_path);
    return 0;
}

static int timestampSeenBefore(const TimestampTracker *tracker,
                               size_t entry_count,
                               const char *timestamp)
{
    for (size_t i = 0; i < entry_count; i++)
    {
        if (strcmp(tracker->entries[i].timestamp, timestamp) == 0)
        {
            return 1;
        }
    }
    return 0;
}

int TimestampTrackerInit(TimestampTracker *tracker,
                         const SacIndexMetaArray *meta,
                         int num_ch,
                         size_t total_groups,
                         const char *stepack_root)
{
    if (tracker == NULL || meta == NULL || meta->values == NULL ||
        num_ch < 1 || total_groups < 1 || stepack_root == NULL)
    {
        LOG_ERROR("timestamp_tracker_invalid_input",
                  "tracker=%p meta=%p num_ch=%d total_groups=%zu stepack_root=%p",
                  (void *)tracker, (void *)meta, num_ch, total_groups,
                  (const void *)stepack_root);
        return -1;
    }

    memset(tracker, 0, sizeof(*tracker));
    tracker->root = PathStringDup(stepack_root);
    tracker->entries = (TimestampTrackerEntry *)calloc(total_groups, sizeof(TimestampTrackerEntry));
    tracker->group_to_entry = (int *)malloc(total_groups * sizeof(int));
    if (tracker->root == NULL || tracker->entries == NULL || tracker->group_to_entry == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=timestamp_tracker groups=%zu", total_groups);
        TimestampTrackerDestroy(tracker);
        return -1;
    }
    tracker->total_groups = total_groups;
    tracker->num_ch = num_ch;

    for (size_t group = 0; group < total_groups; group++)
    {
        size_t first_row = group * (size_t)num_ch;
        const char *timestamp = meta->values[first_row].timestamp;
        if (timestamp == NULL || timestamp[0] == '\0')
        {
            LOG_ERROR("timestamp_empty", "group=%zu", group);
            TimestampTrackerDestroy(tracker);
            return -1;
        }
        if (!PathTimestampLeafIsSafe(timestamp))
        {
            LOG_ERROR("timestamp_unsafe_for_path",
                      "timestamp=%s group=%zu action=fix_sac_index_timestamp",
                      timestamp, group);
            TimestampTrackerDestroy(tracker);
            return -1;
        }
        for (int ch = 1; ch < num_ch; ch++)
        {
            const char *other = meta->values[first_row + (size_t)ch].timestamp;
            if (strcmp(timestamp, other) != 0)
            {
                LOG_ERROR("timestamp_group_mismatch",
                          "group=%zu ch0=%s ch%d=%s",
                          group, timestamp, ch, other);
                TimestampTrackerDestroy(tracker);
                return -1;
            }
        }

        if (tracker->entry_count == 0 ||
            strcmp(tracker->entries[tracker->entry_count - 1].timestamp, timestamp) != 0)
        {
            if (timestampSeenBefore(tracker, tracker->entry_count, timestamp))
            {
                LOG_ERROR("timestamp_not_contiguous",
                          "timestamp=%s group=%zu", timestamp, group);
                TimestampTrackerDestroy(tracker);
                return -1;
            }
            TimestampTrackerEntry *entry = &tracker->entries[tracker->entry_count++];
            snprintf(entry->timestamp, sizeof(entry->timestamp), "%s", timestamp);
            entry->start_group = group;
            entry->group_count = 0;
            entry->done_group_count = 0;
            entry->success_written = 0;
        }
        TimestampTrackerEntry *current = &tracker->entries[tracker->entry_count - 1];
        current->group_count++;
        tracker->group_to_entry[group] = (int)(tracker->entry_count - 1);
    }

    if (pthread_mutex_init(&tracker->mutex, NULL) != 0)
    {
        LOG_ERROR("mutex_init_failed", "target=timestamp_tracker");
        TimestampTrackerDestroy(tracker);
        return -1;
    }
    tracker->mutex_initialized = 1;
    LOG_INFO("timestamp_tracker_ready",
             "timestamps=%zu groups=%zu", tracker->entry_count, total_groups);
    return 0;
}

int TimestampTrackerMarkBatchDone(TimestampTracker *tracker,
                                  size_t start_group,
                                  size_t group_count)
{
    size_t end_group;
    if (tracker == NULL || group_count == 0)
    {
        return 0;
    }
    if (start_group >= tracker->total_groups ||
        group_count > tracker->total_groups - start_group)
    {
        LOG_ERROR("timestamp_batch_range_invalid",
                  "start_group=%zu group_count=%zu total_groups=%zu",
                  start_group, group_count, tracker->total_groups);
        return -1;
    }

    end_group = start_group + group_count;
    pthread_mutex_lock(&tracker->mutex);
    for (size_t group = start_group; group < end_group; group++)
    {
        int entry_index = tracker->group_to_entry[group];
        TimestampTrackerEntry *entry = &tracker->entries[entry_index];
        entry->done_group_count++;
        if (entry->done_group_count > entry->group_count)
        {
            LOG_ERROR("timestamp_done_count_overflow",
                      "timestamp=%s done=%zu total=%zu",
                      entry->timestamp, entry->done_group_count,
                      entry->group_count);
            pthread_mutex_unlock(&tracker->mutex);
            return -1;
        }
        if (!entry->success_written &&
            entry->done_group_count == entry->group_count)
        {
            if (writeTimestampSuccess(tracker, entry) != 0)
            {
                pthread_mutex_unlock(&tracker->mutex);
                return -1;
            }
            entry->success_written = 1;
        }
    }
    pthread_mutex_unlock(&tracker->mutex);
    return 0;
}

void TimestampTrackerDestroy(TimestampTracker *tracker)
{
    if (tracker == NULL)
    {
        return;
    }
    if (tracker->mutex_initialized)
    {
        pthread_mutex_destroy(&tracker->mutex);
        tracker->mutex_initialized = 0;
    }
    free(tracker->root);
    free(tracker->entries);
    free(tracker->group_to_entry);
    tracker->root = NULL;
    tracker->entries = NULL;
    tracker->group_to_entry = NULL;
    tracker->entry_count = 0;
    tracker->total_groups = 0;
}
