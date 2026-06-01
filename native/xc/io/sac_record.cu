#include "sac_record.hpp"

#include "pack_writer.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstring>
#include <limits.h>

extern "C"
{
#include "fs.h"
}

static bool is_leap_year(int year)
{
  return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

static int days_in_year(int year)
{
  return is_leap_year(year) ? 366 : 365;
}

static int day_of_year_from_ymd(int year, int month, int day)
{
  static const int month_days[12] = {
      31, 28, 31, 30, 31, 30,
      31, 31, 30, 31, 30, 31};
  if (month < 1 || month > 12)
    return 0;
  int dim = month_days[month - 1];
  if (month == 2 && is_leap_year(year))
    ++dim;
  if (day < 1 || day > dim)
    return 0;
  int jday = day;
  for (int m = 1; m < month; ++m)
  {
    jday += month_days[m - 1];
    if (m == 2 && is_leap_year(year))
      ++jday;
  }
  return jday;
}

static bool parse_digits(const std::string &text, size_t pos, size_t width, int *out)
{
  if (pos + width > text.size())
    return false;
  int value = 0;
  for (size_t i = 0; i < width; ++i)
  {
    unsigned char ch = (unsigned char)text[pos + i];
    if (!std::isdigit(ch))
      return false;
    value = value * 10 + (int)(ch - '0');
  }
  *out = value;
  return true;
}

static bool valid_time(int hour, int minute)
{
  return hour >= 0 && hour <= 23 && minute >= 0 && minute <= 59;
}

static bool parse_legacy_timestamp_text(const std::string &timestamp,
                                        XcTimeData *time_info)
{
  int year = 0, jday = 0, hhmm = 0;
  int consumed = 0;
  if (sscanf(timestamp.c_str(), "%d.%d.%d%n", &year, &jday, &hhmm, &consumed) != 3 ||
      consumed != (int)timestamp.size())
    return false;
  int hour = hhmm / 100;
  int minute = hhmm % 100;
  if (year <= 0 || jday < 1 || jday > days_in_year(year) || !valid_time(hour, minute))
    return false;
  time_info->year = year;
  time_info->day_of_year = jday;
  time_info->hour = hour;
  time_info->minute = minute;
  return true;
}

static bool parse_xcache_timestamp_text(const std::string &timestamp,
                                        XcTimeData *time_info)
{
  int year = 0, month = 0, day = 0, hour = 0, minute = 0;
  size_t time_pos = 0;
  if (timestamp.size() >= 13 && (timestamp[8] == 'T' || timestamp[8] == 't'))
  {
    if (!parse_digits(timestamp, 0, 4, &year) ||
        !parse_digits(timestamp, 4, 2, &month) ||
        !parse_digits(timestamp, 6, 2, &day))
      return false;
    time_pos = 9;
  }
  else if (timestamp.size() >= 15 &&
           timestamp[4] == '-' && timestamp[7] == '-' &&
           (timestamp[10] == 'T' || timestamp[10] == 't'))
  {
    if (!parse_digits(timestamp, 0, 4, &year) ||
        !parse_digits(timestamp, 5, 2, &month) ||
        !parse_digits(timestamp, 8, 2, &day))
      return false;
    time_pos = 11;
  }
  else
  {
    return false;
  }

  if (time_pos + 5 <= timestamp.size() && timestamp[time_pos + 2] == ':')
  {
    if (!parse_digits(timestamp, time_pos, 2, &hour) ||
        !parse_digits(timestamp, time_pos + 3, 2, &minute))
      return false;
  }
  else if (time_pos + 4 <= timestamp.size())
  {
    if (!parse_digits(timestamp, time_pos, 2, &hour) ||
        !parse_digits(timestamp, time_pos + 2, 2, &minute))
      return false;
  }
  else
  {
    return false;
  }

  int jday = day_of_year_from_ymd(year, month, day);
  if (year <= 0 || jday == 0 || !valid_time(hour, minute))
    return false;
  time_info->year = year;
  time_info->day_of_year = jday;
  time_info->hour = hour;
  time_info->minute = minute;
  return true;
}

bool parse_timestamp_text(const std::string &timestamp, XcTimeData *time_info)
{
  memset(time_info, 0, sizeof(*time_info));
  return parse_legacy_timestamp_text(timestamp, time_info) ||
         parse_xcache_timestamp_text(timestamp, time_info);
}

static void copy_sac_text(char *dst, size_t dst_size, const std::string &src)
{
  memset(dst, ' ', dst_size);
  size_t n = std::min(dst_size, src.size());
  memcpy(dst, src.data(), n);
}

int build_output_path(char *out,
                      size_t out_size,
                      const char *root,
                      const SpecMeta &src,
                      const SpecMeta &rec,
                      bool create_dirs)
{
  char dir1[PATH_MAX], dir2[PATH_MAX];
  int n = snprintf(dir1, sizeof(dir1), "%s/%s.%s",
                   root, src.network.c_str(), src.station.c_str());
  if (n < 0 || n >= (int)sizeof(dir1))
    return -1;
  if (create_dirs && mkdir_p(dir1, 0755) != 0)
    return -1;
  n = snprintf(dir2, sizeof(dir2), "%s/%s.%s",
               dir1, rec.network.c_str(), rec.station.c_str());
  if (n < 0 || n >= (int)sizeof(dir2))
    return -1;
  if (create_dirs && mkdir_p(dir2, 0755) != 0)
    return -1;
  n = snprintf(out, out_size, "%s/%s-%s.%s-%s.%s-%s.bigsac",
               dir2,
               src.network.c_str(), rec.network.c_str(),
               src.station.c_str(), rec.station.c_str(),
               src.component.c_str(), rec.component.c_str());
  if (n < 0 || n >= (int)out_size)
    return -1;
  return 0;
}

void build_sac_header_for_task(SACHEAD *hd,
                               const RuntimeShape *shape,
                               const XcTask &task,
                               const SpecMeta &src,
                               const SpecMeta &rec,
                               const XcTimeData *time_info)
{
  SacheadProcess(hd,
                 rec.stla, rec.stlo,
                 src.stla, src.stlo,
                 task.path_record.great_circle_deg,
                 task.path_record.azimuth_deg,
                 task.path_record.back_azimuth_deg,
                 task.path_record.distance_km,
                 shape->dt,
                 shape->cc_size,
                 shape->half_cc * shape->dt,
                 time_info);
  copy_sac_text(hd->kstnm, sizeof(hd->kstnm), rec.station);
  copy_sac_text(hd->knetwk, sizeof(hd->knetwk), rec.network);
  copy_sac_text(hd->kcmpnm, sizeof(hd->kcmpnm), src.component + "-" + rec.component);
  copy_sac_text(hd->kevnm, sizeof(hd->kevnm), src.network + "." + src.station);
  hd->user0 = (float)src.gnsl_id;
  hd->user1 = (float)rec.gnsl_id;
  hd->user2 = (float)task.path_id;
}

int make_sac_record(std::vector<char> *record,
                    SACHEAD hd,
                    const float *trace,
                    size_t trace_count)
{
  size_t trace_bytes = 0;
  size_t record_bytes = 0;
  if (!checked_mul_size(trace_count, sizeof(float), &trace_bytes) ||
      !checked_add_size(sizeof(SACHEAD), trace_bytes, &record_bytes))
    return -1;
  try
  {
    record->resize(record_bytes);
  }
  catch (...)
  {
    return -1;
  }
  memcpy(record->data(), &hd, sizeof(SACHEAD));
  memcpy(record->data() + sizeof(SACHEAD), trace, trace_bytes);
#ifdef BYTE_SWAP
  if (trace_bytes > (size_t)INT_MAX)
    return -1;
  swab4(record->data() + sizeof(SACHEAD), (int)trace_bytes);
  swab4(record->data(), HD_SIZE);
#endif
  return 0;
}

void fill_pack_record_meta(XcPackRecordMeta *meta,
                           const TimestampWork *timestamp,
                           const RuntimeShape *shape,
                           const WorkerConfig *cfg,
                           const RowBatchJob *job,
                           const XcTask &task,
                           const SpecMeta &src,
                           const SpecMeta &rec,
                           const char *final_pair_path)
{
  meta->timestamp = timestamp->timestamp;
  meta->worker_id = cfg->worker_id;
  if (job)
  {
    meta->anchor_block = job->anchor_block;
    meta->target_begin_block = job->target_begin_block;
    meta->target_end_block = job->target_end_block;
    meta->block_size = job->block_size;
    meta->anchor_begin = job->anchor_begin;
    meta->anchor_end = job->anchor_end;
    meta->target_begin = job->target_begin;
    meta->target_end = job->target_end;
  }
  meta->src_id = src.gnsl_id;
  meta->rec_id = rec.gnsl_id;
  meta->src_network = src.network;
  meta->src_station = src.station;
  meta->src_location = src.location;
  meta->src_component = src.component;
  meta->rec_network = rec.network;
  meta->rec_station = rec.station;
  meta->rec_location = rec.location;
  meta->rec_component = rec.component;
  meta->npts = shape->cc_size;
  meta->dt = shape->dt;
  meta->dist = task.path_record.distance_km;
  meta->az = task.path_record.azimuth_deg;
  meta->baz = task.path_record.back_azimuth_deg;
  meta->final_pair_path = final_pair_path ? final_pair_path : "";
}
