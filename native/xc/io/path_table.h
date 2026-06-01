#ifndef PATH_TABLE_H
#define PATH_TABLE_H

#include <stddef.h>

typedef struct AllowedPathRecord
{
  int path_id;
  int src_gnsl_id;
  int rec_gnsl_id;
  float great_circle_deg;
  float distance_km;
  float azimuth_deg;
  float back_azimuth_deg;
} AllowedPathRecord;

typedef struct AllowedPathTable
{
  AllowedPathRecord *records;
  size_t count;
} AllowedPathTable;

#ifdef __cplusplus
extern "C"
{
#endif

void init_allowed_path_table(AllowedPathTable *table);
int load_allowed_path_table(const char *path, AllowedPathTable *table);
void free_allowed_path_table(AllowedPathTable *table);
int make_allowed_path_id(int src_gnsl_id, int rec_gnsl_id);
void canonical_gnsl_pair(int a_gnsl_id, int b_gnsl_id, int *src_gnsl_id, int *rec_gnsl_id);
const AllowedPathRecord *find_allowed_path_id(const AllowedPathTable *table, int path_id);
const AllowedPathRecord *find_allowed_path_canonical(const AllowedPathTable *table,
                                                     int a_gnsl_id,
                                                     int b_gnsl_id);
const AllowedPathRecord *find_allowed_path_pair(const AllowedPathTable *table,
                                                int src_gnsl_id,
                                                int rec_gnsl_id,
                                                int *is_reversed);

#ifdef __cplusplus
}
#endif

#endif
