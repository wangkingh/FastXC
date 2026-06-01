#include "sac_index.h"
#include "logger.h"
#include "util.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct SacIndexRow
{
    char timestamp[MAXNAME];
    char nsl_id[MAXNAME];
    char network[MAXNAME];
    char station[MAXNAME];
    char location[MAXNAME];
    char component[MAXNAME];
    char *sac_path;
} SacIndexRow;

typedef struct SacIndexPathBuilder
{
    char **in_paths;
    SacIndexMeta *meta_rows;
    size_t count;
    size_t capacity;
} SacIndexPathBuilder;

static char *trimWhitespace(char *text)
{
    char *end;
    while (*text != '\0' && isspace((unsigned char)*text))
    {
        text++;
    }
    if (*text == '\0')
    {
        return text;
    }

    end = text + strlen(text) - 1;
    while (end > text && isspace((unsigned char)*end))
    {
        *end = '\0';
        end--;
    }
    return text;
}

static void copyCleanField(char *dst, size_t dst_size, const char *src)
{
    size_t j = 0;
    int last_was_space = 0;
    if (dst_size == 0)
    {
        return;
    }

    while (*src != '\0' && j + 1 < dst_size)
    {
        if (isspace((unsigned char)*src))
        {
            if (j > 0 && !last_was_space)
            {
                dst[j++] = '_';
            }
            last_was_space = 1;
        }
        else
        {
            dst[j++] = *src;
            last_was_space = 0;
        }
        src++;
    }
    if (j > 0 && dst[j - 1] == '_')
    {
        j--;
    }
    dst[j] = '\0';
}

static void freeSacIndexRow(SacIndexRow *row)
{
    free(row->sac_path);
    row->sac_path = NULL;
}

static int parseNslIdNumber(const char *text, const char *index_path)
{
    char *endptr = NULL;
    long value;
    errno = 0;
    value = strtol(text, &endptr, 10);
    if (errno != 0 || endptr == text || *endptr != '\0' ||
        value < 0 || value > 9999)
    {
        LOG_ERROR("sac_index_nsl_id_invalid",
                  "nsl_id=\"%s\" path=\"%s\" expected=0000..9999",
                  text, index_path);
        exit(1);
    }
    return (int)value;
}

static void appendPathPair(SacIndexPathBuilder *builder, SacIndexRow *row)
{
    if (builder->count == builder->capacity)
    {
        size_t new_capacity = builder->capacity == 0 ? 1024 : builder->capacity * 2;
        char **new_in_paths = (char **)malloc(new_capacity * sizeof(char *));
        SacIndexMeta *new_meta_rows = (SacIndexMeta *)malloc(new_capacity * sizeof(SacIndexMeta));
        if (new_in_paths == NULL || new_meta_rows == NULL)
        {
            LOG_ERROR("alloc_failed", "target=sac_index_path_arrays capacity=%lu",
                      (unsigned long)new_capacity);
            free(new_in_paths);
            free(new_meta_rows);
            exit(1);
        }
        if (builder->count > 0)
        {
            memcpy(new_in_paths, builder->in_paths, builder->count * sizeof(char *));
            memcpy(new_meta_rows, builder->meta_rows, builder->count * sizeof(SacIndexMeta));
        }
        free(builder->in_paths);
        free(builder->meta_rows);
        builder->in_paths = new_in_paths;
        builder->meta_rows = new_meta_rows;
        builder->capacity = new_capacity;
    }

    builder->in_paths[builder->count] = row->sac_path;
    builder->meta_rows[builder->count].gnsl_id = parseNslIdNumber(row->nsl_id, row->sac_path);
    copyCleanField(builder->meta_rows[builder->count].timestamp,
                   sizeof(builder->meta_rows[builder->count].timestamp),
                   row->timestamp);
    copyCleanField(builder->meta_rows[builder->count].nsl_id,
                   sizeof(builder->meta_rows[builder->count].nsl_id),
                   row->nsl_id);
    copyCleanField(builder->meta_rows[builder->count].network,
                   sizeof(builder->meta_rows[builder->count].network),
                   row->network);
    copyCleanField(builder->meta_rows[builder->count].station,
                   sizeof(builder->meta_rows[builder->count].station),
                   row->station);
    copyCleanField(builder->meta_rows[builder->count].location,
                   sizeof(builder->meta_rows[builder->count].location),
                   row->location);
    copyCleanField(builder->meta_rows[builder->count].component,
                   sizeof(builder->meta_rows[builder->count].component),
                   row->component);
    builder->count++;

    row->sac_path = NULL;
}

static int splitTsvRow(char *line, char **fields)
{
    char *cursor = line;
    int i;
    for (i = 0; i < 6; i++)
    {
        char *tab = strchr(cursor, '\t');
        if (tab == NULL)
        {
            return 0;
        }
        *tab = '\0';
        fields[i] = trimWhitespace(cursor);
        cursor = tab + 1;
    }
    fields[6] = trimWhitespace(cursor);
    return 1;
}

static int parseIndexRow(char *line, SacIndexRow *row, char *timestamp,
                         size_t timestamp_size, const char *index_path, int line_no)
{
    char tab_line[MAXLINE];
    char *trimmed = trimWhitespace(line);
    char *fields[7] = {0};

    if (*trimmed == '\0' || *trimmed == '#')
    {
        return 0;
    }
    if (strncmp(trimmed, "timestamp", 9) == 0 || strncmp(trimmed, "nsl_id", 6) == 0)
    {
        return 0;
    }

    strncpy(tab_line, trimmed, MAXLINE - 1);
    tab_line[MAXLINE - 1] = '\0';
    if (!splitTsvRow(tab_line, fields))
    {
        LOG_ERROR("sac_index_row_malformed_tsv",
                  "path=\"%s\" line=%d expected=7_tab_separated_fields",
                  index_path, line_no);
        exit(1);
    }

    memset(row, 0, sizeof(SacIndexRow));
    copyCleanField(timestamp, timestamp_size, fields[0]);
    copyCleanField(row->timestamp, sizeof(row->timestamp), fields[0]);
    copyCleanField(row->nsl_id, sizeof(row->nsl_id), fields[1]);
    copyCleanField(row->network, sizeof(row->network), fields[2]);
    copyCleanField(row->station, sizeof(row->station), fields[3]);
    copyCleanField(row->location, sizeof(row->location), fields[4]);
    copyCleanField(row->component, sizeof(row->component), fields[5]);

    fields[6] = trimWhitespace(fields[6]);
    if (timestamp[0] == '\0' || row->nsl_id[0] == '\0' ||
        row->network[0] == '\0' || row->station[0] == '\0' ||
        row->location[0] == '\0' || row->component[0] == '\0' ||
        fields[6][0] == '\0')
    {
        LOG_ERROR("sac_index_row_incomplete", "path=\"%s\" line=%d", index_path, line_no);
        exit(1);
    }

    row->sac_path = my_strdup(fields[6]);
    if (row->sac_path == NULL)
    {
        LOG_ERROR("alloc_failed", "target=sac_path");
        exit(1);
    }
    return 1;
}

static void ensureGroupCapacity(SacIndexRow **rows, size_t *capacity, size_t needed)
{
    SacIndexRow *new_rows;
    if (needed <= *capacity)
    {
        return;
    }

    *capacity = *capacity == 0 ? 8 : *capacity * 2;
    while (needed > *capacity)
    {
        *capacity *= 2;
    }

    new_rows = (SacIndexRow *)realloc(*rows, *capacity * sizeof(SacIndexRow));
    if (new_rows == NULL)
    {
        LOG_ERROR("alloc_failed", "target=sac_index_group_buffer capacity=%lu",
                  (unsigned long)*capacity);
        exit(1);
    }
    *rows = new_rows;
}

static void flushSacIndexGroup(SacIndexPathBuilder *builder, SacIndexRow *rows,
                               size_t row_count, int num_ch,
                               const char *index_path, const char *timestamp)
{
    size_t i;
    if (row_count == 0)
    {
        return;
    }

    if (num_ch == 1 || row_count == (size_t)num_ch)
    {
        for (i = 0; i < row_count; i++)
        {
            appendPathPair(builder, &rows[i]);
        }
        return;
    }

    LOG_WARN("sac_index_group_skipped",
             "timestamp=%s nsl_id=%s path=\"%s\" expected_components=%d got_components=%lu",
             timestamp, rows[0].nsl_id, index_path, num_ch, (unsigned long)row_count);
    for (i = 0; i < row_count; i++)
    {
        freeSacIndexRow(&rows[i]);
    }
}

static SacIndexPaths finishSacIndexPathBuilder(SacIndexPathBuilder *builder, const char *source_path)
{
    SacIndexPaths result;

    if (builder->count == 0)
    {
        LOG_ERROR("sac_index_empty", "path=\"%s\"", source_path);
        exit(1);
    }
    if (builder->count > (size_t)INT_MAX)
    {
        LOG_ERROR("sac_index_too_large", "rows=%lu max=%d",
                  (unsigned long)builder->count, INT_MAX);
        exit(1);
    }

    result.in_paths.paths = builder->in_paths;
    result.in_paths.count = (int)builder->count;
    result.meta.values = builder->meta_rows;
    result.meta.count = (int)builder->count;

    LOG_INFO("sac_index_loaded", "rows=%lu path=\"%s\"",
             (unsigned long)builder->count, source_path);
    return result;
}

SacIndexPaths readSacIndexPaths(const char *index_path, int num_ch)
{
    FILE *fp = fopen(index_path, "r");
    char line[MAXLINE];
    char current_timestamp[MAXNAME] = "";
    char current_nsl[MAXNAME] = "";
    SacIndexRow *group_rows = NULL;
    size_t group_count = 0;
    size_t group_capacity = 0;
    int line_no = 0;
    SacIndexPathBuilder builder = {0};

    if (fp == NULL)
    {
        LOG_ERROR("open_sac_index_failed", "path=\"%s\"", index_path);
        exit(1);
    }

    while (fgets(line, MAXLINE, fp) != NULL)
    {
        SacIndexRow row;
        char timestamp[MAXNAME];
        line_no++;
        if (!parseIndexRow(line, &row, timestamp, sizeof(timestamp), index_path, line_no))
        {
            continue;
        }

        if (current_nsl[0] != '\0' &&
            (strcmp(current_nsl, row.nsl_id) != 0 || strcmp(current_timestamp, timestamp) != 0))
        {
            flushSacIndexGroup(&builder, group_rows, group_count, num_ch, index_path, current_timestamp);
            group_count = 0;
        }

        if (current_nsl[0] == '\0' ||
            strcmp(current_nsl, row.nsl_id) != 0 ||
            strcmp(current_timestamp, timestamp) != 0)
        {
            strncpy(current_nsl, row.nsl_id, sizeof(current_nsl) - 1);
            current_nsl[sizeof(current_nsl) - 1] = '\0';
            strncpy(current_timestamp, timestamp, sizeof(current_timestamp) - 1);
            current_timestamp[sizeof(current_timestamp) - 1] = '\0';
        }

        ensureGroupCapacity(&group_rows, &group_capacity, group_count + 1);
        group_rows[group_count++] = row;
    }

    flushSacIndexGroup(&builder, group_rows, group_count, num_ch, index_path, current_timestamp);

    free(group_rows);
    fclose(fp);

    return finishSacIndexPathBuilder(&builder, index_path);
}
