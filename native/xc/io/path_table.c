#include "path_table.h"
#include "logger.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int make_allowed_path_id(int src_gnsl_id, int rec_gnsl_id)
{
  if (src_gnsl_id <= 0 || src_gnsl_id > 9999 ||
      rec_gnsl_id <= 0 || rec_gnsl_id > 9999)
  {
    return -1;
  }
  return src_gnsl_id * 10000 + rec_gnsl_id;
}

void canonical_gnsl_pair(int a_gnsl_id, int b_gnsl_id, int *src_gnsl_id, int *rec_gnsl_id)
{
  if (a_gnsl_id <= b_gnsl_id)
  {
    if (src_gnsl_id)
      *src_gnsl_id = a_gnsl_id;
    if (rec_gnsl_id)
      *rec_gnsl_id = b_gnsl_id;
  }
  else
  {
    if (src_gnsl_id)
      *src_gnsl_id = b_gnsl_id;
    if (rec_gnsl_id)
      *rec_gnsl_id = a_gnsl_id;
  }
}

static int cmp_allowed_path_record(const void *a, const void *b)
{
  const AllowedPathRecord *ra = (const AllowedPathRecord *)a;
  const AllowedPathRecord *rb = (const AllowedPathRecord *)b;
  if (ra->path_id < rb->path_id)
    return -1;
  if (ra->path_id > rb->path_id)
    return 1;
  return 0;
}

const AllowedPathRecord *find_allowed_path_id(const AllowedPathTable *table, int path_id)
{
  size_t lo = 0;
  size_t hi = table ? table->count : 0;
  while (lo < hi)
  {
    size_t mid = lo + (hi - lo) / 2;
    int mid_id = table->records[mid].path_id;
    if (mid_id == path_id)
      return &table->records[mid];
    if (mid_id < path_id)
      lo = mid + 1;
    else
      hi = mid;
  }
  return NULL;
}

void init_allowed_path_table(AllowedPathTable *table)
{
  if (!table)
    return;
  table->records = NULL;
  table->count = 0;
}

void free_allowed_path_table(AllowedPathTable *table)
{
  if (!table)
    return;
  free(table->records);
  init_allowed_path_table(table);
}

int load_allowed_path_table(const char *path, AllowedPathTable *table)
{
  FILE *fp = NULL;
  char line[4096];
  size_t capacity = 0;

  if (!path || !table)
    return -1;

  init_allowed_path_table(table);
  fp = fopen(path, "r");
  if (!fp)
  {
    LOG_ERROR("allowed_path_table_open_failed", "path=\"%s\" error=\"%s\"",
              path, strerror(errno));
    return -1;
  }

  while (fgets(line, sizeof(line), fp) != NULL)
  {
    char *fields[9] = {0};
    char *saveptr = NULL;
    char *token = NULL;
    int field_count = 0;
    AllowedPathRecord record;

    if (line[0] == '\0' || line[0] == '\n' || line[0] == '#')
      continue;
    if (strncmp(line, "path_id", 7) == 0)
      continue;

    token = strtok_r(line, "\t\r\n", &saveptr);
    while (token != NULL && field_count < 9)
    {
      fields[field_count++] = token;
      token = strtok_r(NULL, "\t\r\n", &saveptr);
    }
    if (field_count < 8)
      continue;

    record.path_id = atoi(fields[0]);
    record.src_gnsl_id = atoi(fields[1]);
    record.rec_gnsl_id = atoi(fields[2]);
    if (field_count >= 9)
    {
      record.great_circle_deg = (float)atof(fields[5]);
      record.distance_km = (float)atof(fields[6]);
      record.azimuth_deg = (float)atof(fields[7]);
      record.back_azimuth_deg = (float)atof(fields[8]);
    }
    else
    {
      record.great_circle_deg = (float)(atof(fields[5]) / 111.195);
      record.distance_km = (float)atof(fields[5]);
      record.azimuth_deg = (float)atof(fields[6]);
      record.back_azimuth_deg = (float)atof(fields[7]);
    }

    if (record.path_id <= 0 ||
        record.path_id != make_allowed_path_id(record.src_gnsl_id, record.rec_gnsl_id) ||
        record.src_gnsl_id > record.rec_gnsl_id)
    {
      LOG_WARN("allowed_path_row_skipped",
               "path=\"%s\" path_id=%d src=%d rec=%d reason=malformed_or_noncanonical",
               path, record.path_id, record.src_gnsl_id, record.rec_gnsl_id);
      continue;
    }

    if (table->count == capacity)
    {
      size_t new_capacity = capacity == 0 ? 1024 : capacity * 2;
      AllowedPathRecord *new_records =
          (AllowedPathRecord *)realloc(table->records, new_capacity * sizeof(AllowedPathRecord));
      if (!new_records)
      {
        size_t records_read = table->count;
        fclose(fp);
        free_allowed_path_table(table);
        LOG_ERROR("allowed_path_table_oom", "path=\"%s\" records=%zu",
                  path, records_read);
        return -1;
      }
      table->records = new_records;
      capacity = new_capacity;
    }

    table->records[table->count++] = record;
  }

  fclose(fp);
  if (table->count > 1)
  {
    qsort(table->records, table->count, sizeof(AllowedPathRecord), cmp_allowed_path_record);
  }
  return 0;
}

const AllowedPathRecord *find_allowed_path_canonical(const AllowedPathTable *table,
                                                     int a_gnsl_id,
                                                     int b_gnsl_id)
{
  int src_id = 0;
  int rec_id = 0;
  int path_id = -1;
  if (!table)
    return NULL;
  canonical_gnsl_pair(a_gnsl_id, b_gnsl_id, &src_id, &rec_id);
  path_id = make_allowed_path_id(src_id, rec_id);
  if (path_id <= 0)
    return NULL;
  return find_allowed_path_id(table, path_id);
}

const AllowedPathRecord *find_allowed_path_pair(const AllowedPathTable *table,
                                                int src_gnsl_id,
                                                int rec_gnsl_id,
                                                int *is_reversed)
{
  int path_id = make_allowed_path_id(src_gnsl_id, rec_gnsl_id);
  const AllowedPathRecord *record = NULL;

  if (is_reversed)
    *is_reversed = 0;
  if (!table)
    return NULL;

  if (path_id > 0)
  {
    record = find_allowed_path_id(table, path_id);
    if (record)
      return record;
  }

  path_id = make_allowed_path_id(rec_gnsl_id, src_gnsl_id);
  if (path_id > 0)
  {
    record = find_allowed_path_id(table, path_id);
    if (record && is_reversed)
      *is_reversed = 1;
    return record;
  }

  return NULL;
}
