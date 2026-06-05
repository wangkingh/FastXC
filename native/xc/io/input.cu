#include "input.hpp"
#include "logger.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <limits.h>
#include <map>
#include <stdint.h>
#include <strings.h>
#include <string>
#include <sys/stat.h>
#include <ctime>
#include <unistd.h>
#include <vector>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static const uint32_t kNslcEntrySize = 128;
static const uint32_t kStepackVersion = 3;
static const uint32_t kStepackLayoutPitchedStepNslcFreq = 2;

#pragma pack(push, 1)
struct NslcEntryRaw
{
  uint32_t file_index;
  uint32_t nsl_id;
  float stla;
  float stlo;
  char network[16];
  char station[32];
  char location[16];
  char component[16];
  unsigned char reserved[32];
};

struct StepackHeaderRaw
{
  char magic[8];
  uint32_t version;
  uint32_t header_size;
  uint32_t nslc_entry_size;
  uint32_t layout;
  uint64_t batch_seq;
  uint64_t start_group;
  uint64_t group_count;
  uint32_t worker_id;
  uint32_t nstep;
  uint32_t nslc_count;
  uint32_t nspec;
  float dt;
  float df;
  uint64_t nslc_table_bytes;
  uint64_t payload_offset;
  uint64_t payload_bytes;
  uint64_t pitch_step_bytes;
  char first_timestamp[64];
  unsigned char reserved[64];
};
#pragma pack(pop)

static_assert(sizeof(NslcEntryRaw) == kNslcEntrySize, "bad NSLC entry ABI");
static_assert(sizeof(StepackHeaderRaw) == 232, "bad .stepack header ABI");

static std::string trim(const std::string &text)
{
  size_t b = 0;
  size_t e = text.size();
  while (b < e && strchr(" \t\r\n", text[b]))
    ++b;
  while (e > b && strchr(" \t\r\n", text[e - 1]))
    --e;
  return text.substr(b, e - b);
}

static std::string dirname_of(const std::string &path)
{
  size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos)
    return ".";
  return pos == 0 ? path.substr(0, 1) : path.substr(0, pos);
}

static std::string resolve_relative_path(const std::string &path,
                                         const std::string &base_dir)
{
  return !path.empty() && path[0] != '/' ? base_dir + "/" + path : path;
}

static std::string fixed_string(const char *text, size_t size)
{
  size_t n = 0;
  while (n < size && text[n] != '\0')
    ++n;
  return std::string(text, n);
}

static bool is_leap_year(int year)
{
  return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

static bool date_from_year_jday(int year, int jday, int *month, int *day)
{
  static const int mdays_common[12] =
      {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  int max_day = is_leap_year(year) ? 366 : 365;
  if (year < 1 || jday < 1 || jday > max_day)
    return false;

  int remaining = jday;
  for (int m = 0; m < 12; ++m)
  {
    int mdays = mdays_common[m] + (m == 1 && is_leap_year(year) ? 1 : 0);
    if (remaining <= mdays)
    {
      *month = m + 1;
      *day = remaining;
      return true;
    }
    remaining -= mdays;
  }
  return false;
}

static bool valid_clock(int hour, int minute, int second)
{
  return hour >= 0 && hour <= 23 &&
         minute >= 0 && minute <= 59 &&
         second >= 0 && second <= 59;
}

static std::string format_xc_timestamp(int year, int month, int day,
                                       int hour, int minute)
{
  char out[32];
  snprintf(out, sizeof(out), "%04d%02d%02dT%02d:%02d",
           year, month, day, hour, minute);
  return std::string(out);
}

static std::string normalize_timestamp_for_xc(const std::string &text)
{
  std::string raw = trim(text);
  int year = 0, month = 0, day = 0, jday = 0;
  int hour = 0, minute = 0, second = 0;
  int n = 0;

  if (sscanf(raw.c_str(), "%d.%d.%2d%2d%n",
             &year, &jday, &hour, &minute, &n) == 4 &&
      n == (int)raw.size() &&
      date_from_year_jday(year, jday, &month, &day) &&
      valid_clock(hour, minute, 0))
    return format_xc_timestamp(year, month, day, hour, minute);

  n = 0;
  if (sscanf(raw.c_str(), "%d.%d.%d:%d%n",
             &year, &jday, &hour, &minute, &n) == 4 &&
      n == (int)raw.size() &&
      date_from_year_jday(year, jday, &month, &day) &&
      valid_clock(hour, minute, 0))
    return format_xc_timestamp(year, month, day, hour, minute);

  n = 0;
  if (sscanf(raw.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d%n",
             &year, &month, &day, &hour, &minute, &second, &n) == 6 &&
      n == (int)raw.size() && valid_clock(hour, minute, second))
    return format_xc_timestamp(year, month, day, hour, minute);

  n = 0;
  if (sscanf(raw.c_str(), "%4d-%2d-%2dT%2d:%2d%n",
             &year, &month, &day, &hour, &minute, &n) == 5 &&
      n == (int)raw.size() && valid_clock(hour, minute, 0))
    return format_xc_timestamp(year, month, day, hour, minute);

  n = 0;
  if (sscanf(raw.c_str(), "%4d-%2d-%2d %2d:%2d:%2d%n",
             &year, &month, &day, &hour, &minute, &second, &n) == 6 &&
      n == (int)raw.size() && valid_clock(hour, minute, second))
    return format_xc_timestamp(year, month, day, hour, minute);

  n = 0;
  if (sscanf(raw.c_str(), "%4d-%2d-%2d %2d:%2d%n",
             &year, &month, &day, &hour, &minute, &n) == 5 &&
      n == (int)raw.size() && valid_clock(hour, minute, 0))
    return format_xc_timestamp(year, month, day, hour, minute);

  n = 0;
  if (sscanf(raw.c_str(), "%4d%2d%2dT%2d:%2d:%2d%n",
             &year, &month, &day, &hour, &minute, &second, &n) == 6 &&
      n == (int)raw.size() && valid_clock(hour, minute, second))
    return format_xc_timestamp(year, month, day, hour, minute);

  n = 0;
  if (sscanf(raw.c_str(), "%4d%2d%2dT%2d:%2d%n",
             &year, &month, &day, &hour, &minute, &n) == 5 &&
      n == (int)raw.size() && valid_clock(hour, minute, 0))
    return format_xc_timestamp(year, month, day, hour, minute);

  n = 0;
  if (sscanf(raw.c_str(), "%4d%2d%2dT%2d%2d%n",
             &year, &month, &day, &hour, &minute, &n) == 5 &&
      n == (int)raw.size() && valid_clock(hour, minute, 0))
    return format_xc_timestamp(year, month, day, hour, minute);

  n = 0;
  if (sscanf(raw.c_str(), "%4d%2d%2d%2d%2d%n",
             &year, &month, &day, &hour, &minute, &n) == 5 &&
      n == (int)raw.size() && valid_clock(hour, minute, 0))
    return format_xc_timestamp(year, month, day, hour, minute);

  LOG_ERROR("stepack_timestamp_unsupported",
            "timestamp=\"%s\" expected=\"YYYY.JJJ.HHMM or YYYYMMDDTHH:MM\"",
            text.c_str());
  return std::string();
}

static std::vector<std::string> split_tab(const std::string &text)
{
  std::vector<std::string> fields;
  size_t begin = 0;
  while (begin <= text.size())
  {
    size_t tab = text.find('\t', begin);
    fields.push_back(text.substr(begin, tab == std::string::npos ? tab : tab - begin));
    if (tab == std::string::npos)
      break;
    begin = tab + 1;
  }
  return fields;
}

static int column_index(const std::vector<std::string> &header, const char *name)
{
  for (size_t i = 0; i < header.size(); ++i)
    if (strcasecmp(header[i].c_str(), name) == 0)
      return (int)i;
  return -1;
}

static bool parse_u64_strict(const std::string &text, uint64_t *out)
{
  char *end = NULL;
  unsigned long long value = 0;
  errno = 0;
  if (text.empty())
    return false;
  value = strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0')
    return false;
  *out = (uint64_t)value;
  return true;
}

static bool parse_size_strict(const std::string &text, size_t *out)
{
  uint64_t value = 0;
  if (!parse_u64_strict(text, &value) ||
      value > (uint64_t)std::numeric_limits<size_t>::max())
    return false;
  *out = (size_t)value;
  return true;
}

static bool parse_float_strict(const std::string &text, float *out)
{
  char *end = NULL;
  errno = 0;
  float value = strtof(text.c_str(), &end);
  if (errno != 0 || end == text.c_str() || *end != '\0')
    return false;
  *out = value;
  return true;
}

static bool is_dir_path(const std::string &path)
{
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool is_file_path(const std::string &path)
{
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static bool has_suffix(const std::string &text, const char *suffix)
{
  size_t n = strlen(suffix);
  return text.size() >= n &&
         strcasecmp(text.c_str() + text.size() - n, suffix) == 0;
}

static std::vector<std::string> list_files_with_suffix(const std::string &dir,
                                                       const char *suffix)
{
  std::vector<std::string> paths;
  DIR *d = opendir(dir.c_str());
  if (!d)
  {
    LOG_ERROR("input_dir_open_failed", "path=\"%s\" error=\"%s\"",
              dir.c_str(), strerror(errno));
    exit(1);
  }

  struct dirent *ent = NULL;
  while ((ent = readdir(d)) != NULL)
  {
    std::string name = ent->d_name;
    if (name == "." || name == ".." || !has_suffix(name, suffix))
      continue;
    std::string path = dir + "/" + name;
    if (is_file_path(path))
      paths.push_back(path);
  }
  closedir(d);
  std::sort(paths.begin(), paths.end());
  return paths;
}

static int read_exact_at(int fd, size_t offset, void *dst, size_t bytes)
{
  char *out = (char *)dst;
  size_t done = 0;
  while (done < bytes)
  {
    size_t chunk = std::min(bytes - done, (size_t)1 << 30);
    if (offset + done > (size_t)std::numeric_limits<off_t>::max())
      return -1;
    ssize_t n = pread(fd, out + done, chunk, (off_t)(offset + done));
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return -1;
    done += (size_t)n;
  }
  return 0;
}

static size_t infer_num_channels(const TimestampWork *work)
{
  size_t expected = 0;
  size_t i = 0;
  int last_id = 0;

  while (i < work->specs.size())
  {
    int id = work->specs[i].gnsl_id;
    size_t begin = i;
    if (id <= last_id)
    {
      LOG_ERROR("nslc_table_unsorted", "path=\"%s\" nsl_id=%d previous=%d",
                work->input_pack_path.c_str(), id, last_id);
      return 0;
    }
    while (i < work->specs.size() && work->specs[i].gnsl_id == id)
      ++i;

    size_t count = i - begin;
    if (expected == 0)
      expected = count;
    else if (count != expected)
    {
      LOG_ERROR("nslc_channel_group_incomplete",
                "path=\"%s\" nsl_id=%d channels=%zu expected=%zu",
                work->input_pack_path.c_str(), id, count, expected);
      return 0;
    }
    last_id = id;
  }
  return expected;
}

static bool first_tsv_header_has_columns(const std::string &path,
                                         const char *required_a,
                                         const char *required_b)
{
  FILE *fp = fopen(path.c_str(), "r");
  if (!fp)
    return false;

  char line[PATH_MAX * 4];
  bool ok = false;
  while (fgets(line, sizeof(line), fp))
  {
    std::string text = trim(line);
    if (text.empty() || text[0] == '#')
      continue;
    std::vector<std::string> header = split_tab(text);
    ok = column_index(header, required_a) >= 0 &&
         column_index(header, required_b) >= 0;
    break;
  }
  fclose(fp);
  return ok;
}

static void add_stepack_fragment(std::vector<TimestampInput> *inputs,
                                 std::map<std::string, size_t> *by_timestamp,
                                 const StepackInputFragment &fragment,
                                 const std::string &timestamp)
{
  std::map<std::string, size_t>::iterator it = by_timestamp->find(timestamp);
  if (it == by_timestamp->end())
  {
    TimestampInput input;
    input.timestamp = timestamp;
    input.input_pack_path = fragment.pack_path;
    inputs->push_back(input);
    size_t index = inputs->size() - 1;
    (*by_timestamp)[timestamp] = index;
    it = by_timestamp->find(timestamp);
  }

  TimestampInput &input = (*inputs)[it->second];
  input.stepack_fragments.push_back(fragment);
  input.file_count_hint += fragment.nslc_count;
}

static bool parse_stepack_fragment_row(const std::vector<std::string> &fields,
                                       const std::vector<std::string> &header,
                                       const std::string &base_dir,
                                       StepackInputFragment *fragment,
                                       std::string *timestamp,
                                       const std::string &text)
{
  int timestamp_col = column_index(header, "timestamp");
  int worker_col = column_index(header, "worker_id");
  int batch_col = column_index(header, "batch_seq");
  int start_col = column_index(header, "start_group");
  int group_count_col = column_index(header, "group_count");
  int path_col = column_index(header, "pack_path");
  int nstep_col = column_index(header, "nstep");
  int nslc_start_col = column_index(header, "nslc_start");
  int nslc_count_col = column_index(header, "nslc_count");
  int batch_nslc_col = column_index(header, "batch_nslc_count");
  int nspec_col = column_index(header, "nspec");
  int dt_col = column_index(header, "dt");
  int df_col = column_index(header, "df");
  int payload_offset_col = column_index(header, "payload_offset");
  int payload_col = column_index(header, "payload_bytes");
  int step_bytes_col = column_index(header, "step_bytes");
  int pitch_step_col = column_index(header, "pitch_step_bytes");
  int nslc_step_col = column_index(header, "nslc_step_bytes");

  int required[] = {
      timestamp_col, path_col, nstep_col, nslc_start_col,
      nslc_count_col, batch_nslc_col, nspec_col, dt_col, df_col,
      payload_offset_col, payload_col, step_bytes_col, pitch_step_col,
      nslc_step_col};
  for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i)
  {
    if (required[i] < 0 || required[i] >= (int)fields.size())
      return false;
  }

  StepackInputFragment f;
  std::string normalized_timestamp =
      normalize_timestamp_for_xc(fields[timestamp_col]);
  if (normalized_timestamp.empty())
    return false;
  *timestamp = normalized_timestamp;
  f.pack_path = resolve_relative_path(fields[path_col], base_dir);

  if ((worker_col >= 0 && worker_col < (int)fields.size() &&
       !parse_size_strict(fields[worker_col], &f.worker_id)) ||
      (batch_col >= 0 && batch_col < (int)fields.size() &&
       !parse_size_strict(fields[batch_col], &f.batch_seq)) ||
      (start_col >= 0 && start_col < (int)fields.size() &&
       !parse_size_strict(fields[start_col], &f.start_group)) ||
      (group_count_col >= 0 && group_count_col < (int)fields.size() &&
       !parse_size_strict(fields[group_count_col], &f.group_count)) ||
      !parse_size_strict(fields[nstep_col], &f.nstep) ||
      !parse_size_strict(fields[nslc_start_col], &f.nslc_start) ||
      !parse_size_strict(fields[nslc_count_col], &f.nslc_count) ||
      !parse_size_strict(fields[batch_nslc_col], &f.batch_nslc_count) ||
      !parse_size_strict(fields[nspec_col], &f.nspec) ||
      !parse_float_strict(fields[dt_col], &f.dt) ||
      !parse_float_strict(fields[df_col], &f.df) ||
      !parse_size_strict(fields[payload_offset_col], &f.payload_offset) ||
      !parse_size_strict(fields[payload_col], &f.payload_bytes) ||
      !parse_size_strict(fields[step_bytes_col], &f.step_bytes) ||
      !parse_size_strict(fields[pitch_step_col], &f.pitch_step_bytes) ||
      !parse_size_strict(fields[nslc_step_col], &f.nslc_step_bytes))
  {
    LOG_ERROR("stepack_tsv_row_numeric_invalid", "row=\"%s\"", text.c_str());
    return false;
  }

  size_t expected_nslc_step_bytes = 0;
  size_t expected_step_bytes = 0;
  size_t expected_payload_bytes = 0;
  if (f.nslc_count == 0 || f.nstep == 0 || f.nspec < 2 ||
      !checked_mul_size(f.nspec, sizeof(complex), &expected_nslc_step_bytes) ||
      expected_nslc_step_bytes != f.nslc_step_bytes ||
      !checked_mul_size(f.nslc_count, f.nslc_step_bytes, &expected_step_bytes) ||
      expected_step_bytes != f.step_bytes ||
      !checked_mul_size(f.nstep, f.step_bytes, &expected_payload_bytes) ||
      expected_payload_bytes != f.payload_bytes)
  {
    LOG_ERROR("stepack_tsv_row_shape_invalid", "row=\"%s\"", text.c_str());
    return false;
  }

  *fragment = f;
  return true;
}

static std::vector<TimestampInput>
load_stepack_timestamp_inputs_from_tsvs(const std::vector<std::string> &tsvs)
{
  std::vector<TimestampInput> inputs;
  std::map<std::string, size_t> by_timestamp;

  for (const std::string &tsv : tsvs)
  {
    FILE *fp = fopen(tsv.c_str(), "r");
    if (!fp)
    {
      LOG_ERROR("stepack_tsv_open_failed", "path=\"%s\" error=\"%s\"",
                tsv.c_str(), strerror(errno));
      exit(1);
    }

    char line[PATH_MAX * 4];
    std::vector<std::string> header;
    std::string base_dir = dirname_of(tsv);
    while (fgets(line, sizeof(line), fp))
    {
      std::string text = trim(line);
      if (text.empty() || text[0] == '#')
        continue;
      std::vector<std::string> fields = split_tab(text);
      if (header.empty())
      {
        header = fields;
        if (column_index(header, "pack_path") < 0 ||
            column_index(header, "nslc_start") < 0 ||
            column_index(header, "pitch_step_bytes") < 0)
        {
          LOG_ERROR("stepack_tsv_columns_missing", "path=\"%s\"", tsv.c_str());
          fclose(fp);
          exit(1);
        }
        continue;
      }

      StepackInputFragment fragment;
      std::string timestamp;
      if (!parse_stepack_fragment_row(fields, header, base_dir,
                                      &fragment, &timestamp, text))
      {
        LOG_ERROR("stepack_tsv_row_malformed", "path=\"%s\" row=\"%s\"",
                  tsv.c_str(), text.c_str());
        fclose(fp);
        exit(1);
      }
      add_stepack_fragment(&inputs, &by_timestamp, fragment, timestamp);
    }
    fclose(fp);
  }

  std::sort(inputs.begin(), inputs.end(),
            [](const TimestampInput &a, const TimestampInput &b) {
              return a.timestamp < b.timestamp;
            });
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    std::sort(inputs[i].stepack_fragments.begin(),
              inputs[i].stepack_fragments.end(),
              [](const StepackInputFragment &a, const StepackInputFragment &b) {
                if (a.batch_seq != b.batch_seq)
                  return a.batch_seq < b.batch_seq;
                if (a.start_group != b.start_group)
                  return a.start_group < b.start_group;
                if (a.nslc_start != b.nslc_start)
                  return a.nslc_start < b.nslc_start;
                return a.pack_path < b.pack_path;
              });
    if (!inputs[i].stepack_fragments.empty())
      inputs[i].input_pack_path = inputs[i].stepack_fragments[0].pack_path;
  }

  LOG_INFO("stepack_index_loaded", "timestamps=%zu tsvs=%zu",
           inputs.size(), tsvs.size());
  return inputs;
}

static std::vector<TimestampInput>
load_stepack_timestamp_inputs_from_path(const std::string &input_path)
{
  if (is_dir_path(input_path))
  {
    std::string dir = input_path;
    std::string nested = dir + "/stepack";
    if (is_dir_path(nested))
      dir = nested;
    std::vector<std::string> tsvs = list_files_with_suffix(dir, ".tsv");
    if (tsvs.empty())
    {
      LOG_ERROR("stepack_tsv_not_found", "path=\"%s\"", dir.c_str());
      exit(1);
    }
    return load_stepack_timestamp_inputs_from_tsvs(tsvs);
  }

  std::vector<std::string> tsvs;
  tsvs.push_back(input_path);
  return load_stepack_timestamp_inputs_from_tsvs(tsvs);
}

std::vector<TimestampInput> load_timestamp_inputs(const ARGUTYPE *args)
{
  std::string input_path = args->input_path ? args->input_path : "";
  if (is_dir_path(input_path))
    return load_stepack_timestamp_inputs_from_path(input_path);
  if (first_tsv_header_has_columns(input_path, "pack_path", "nslc_start"))
    return load_stepack_timestamp_inputs_from_path(input_path);

  LOG_ERROR("stepack_input_invalid",
            "path=\"%s\" expected=stepack_directory_or_stepack_tsv",
            input_path.c_str());
  exit(1);
}

static bool valid_stepack_header(const StepackHeaderRaw &h,
                                 const StepackInputFragment &fragment,
                                 const RuntimeShape *expected,
                                 size_t file_size,
                                 const char *path)
{
  size_t nslc_table_bytes = 0;
  size_t nslc_table_end = 0;
  size_t payload_end = 0;
  if (memcmp(h.magic, "FXCSTPK", 7) != 0 ||
      h.version != kStepackVersion ||
      h.header_size != sizeof(StepackHeaderRaw) ||
      h.nslc_entry_size != kNslcEntrySize ||
      h.layout != kStepackLayoutPitchedStepNslcFreq)
  {
    LOG_ERROR("stepack_format_unsupported", "path=\"%s\"", path);
    return false;
  }

  if (h.nstep == 0 || h.nspec < 2 || h.nslc_count == 0 ||
      h.dt <= 0.0f || h.df <= 0.0f)
  {
    LOG_ERROR("stepack_shape_invalid", "path=\"%s\"", path);
    return false;
  }

  if ((size_t)h.nstep != fragment.nstep ||
      (size_t)h.nspec != fragment.nspec ||
      (size_t)h.nslc_count != fragment.batch_nslc_count ||
      (size_t)h.payload_offset != fragment.payload_offset ||
      (size_t)h.pitch_step_bytes != fragment.pitch_step_bytes ||
      fabsf(h.dt - fragment.dt) > 1e-6f ||
      fabsf(h.df - fragment.df) > 1e-9f)
  {
    LOG_ERROR("stepack_tsv_header_mismatch", "path=\"%s\"", path);
    return false;
  }

  if (expected->nspec > 0 &&
      ((int)h.nspec != expected->nspec ||
       (int)h.nstep != expected->nstep ||
       fabsf(h.dt - expected->dt) > 1e-6f ||
       fabsf(h.df - expected->df) > 1e-9f))
  {
    LOG_ERROR("stepack_shape_mismatch", "path=\"%s\"", path);
    return false;
  }

  if (fragment.nslc_start >= fragment.batch_nslc_count ||
      fragment.nslc_count > fragment.batch_nslc_count - fragment.nslc_start ||
      !checked_mul_size((size_t)h.nslc_count, (size_t)h.nslc_entry_size, &nslc_table_bytes) ||
      nslc_table_bytes != (size_t)h.nslc_table_bytes ||
      !checked_add_size((size_t)h.header_size, nslc_table_bytes, &nslc_table_end) ||
      (size_t)h.payload_offset < nslc_table_end ||
      !checked_add_size((size_t)h.payload_offset, (size_t)h.payload_bytes, &payload_end) ||
      payload_end > file_size)
  {
    LOG_ERROR("stepack_layout_invalid", "path=\"%s\"", path);
    return false;
  }
  return true;
}

struct StepackNslcTemp
{
  SpecMeta meta;
  NslcLocator locator;
  size_t original_order = 0;
};

static int append_stepack_fragment_nslcs(TimestampWork *work,
                                         size_t fragment_index,
                                         std::vector<StepackNslcTemp> *nslcs)
{
  const StepackFragment &fragment = work->stepack_fragments[fragment_index];
  for (size_t i = 0; i < fragment.nslc_count; ++i)
  {
    NslcEntryRaw row;
    size_t file_index = fragment.nslc_start + i;
    size_t offset = sizeof(StepackHeaderRaw) + file_index * sizeof(row);
    if (read_exact_at(fragment.fd, offset, &row, sizeof(row)) != 0)
    {
      LOG_ERROR("stepack_nslc_table_read_failed", "path=\"%s\" index=%zu error=\"%s\"",
                fragment.pack_path.c_str(), file_index, strerror(errno));
      return -1;
    }
    if (row.file_index != file_index || row.nsl_id == 0)
    {
      LOG_ERROR("stepack_nslc_entry_invalid", "path=\"%s\" index=%zu nsl_id=%u",
                fragment.pack_path.c_str(), file_index, row.nsl_id);
      return -1;
    }

    StepackNslcTemp temp;
    temp.meta.path = fragment.pack_path;
    temp.meta.group = std::to_string(row.nsl_id);
    temp.meta.network = fixed_string(row.network, sizeof(row.network));
    temp.meta.station = fixed_string(row.station, sizeof(row.station));
    temp.meta.location = fixed_string(row.location, sizeof(row.location));
    temp.meta.component = fixed_string(row.component, sizeof(row.component));
    temp.meta.stla = row.stla;
    temp.meta.stlo = row.stlo;
    temp.meta.gnsl_id = (int)row.nsl_id;
    temp.locator.fragment_index = fragment_index;
    temp.locator.fragment_local_index = i;
    temp.original_order = nslcs->size();
    nslcs->push_back(temp);
  }
  return 0;
}

static int open_stepack_timestamp_work(const TimestampInput &input,
                                       TimestampWork *work,
                                       const RuntimeShape *expected,
                                       RuntimeShape *shape)
{
  if (input.stepack_fragments.empty())
  {
    LOG_ERROR("stepack_timestamp_empty", "timestamp=\"%s\"",
              input.timestamp.c_str());
    return -1;
  }

  work->input_path = input.input_pack_path;
  work->timestamp = input.timestamp;
  work->input_pack_path = input.input_pack_path;
  work->manifest_path = input.manifest_path;

  std::vector<StepackNslcTemp> nslcs;
  for (size_t i = 0; i < input.stepack_fragments.size(); ++i)
  {
    const StepackInputFragment &in_fragment = input.stepack_fragments[i];
    int fd = open(in_fragment.pack_path.c_str(), O_RDONLY);
    if (fd < 0)
    {
      LOG_ERROR("stepack_open_failed", "path=\"%s\" error=\"%s\"",
                in_fragment.pack_path.c_str(), strerror(errno));
      return -1;
    }

    StepackHeaderRaw header;
    struct stat st;
    if (fstat(fd, &st) != 0 ||
        read_exact_at(fd, 0, &header, sizeof(header)) != 0 ||
        !valid_stepack_header(header, in_fragment, expected,
                              (size_t)st.st_size, in_fragment.pack_path.c_str()))
    {
      LOG_ERROR("stepack_header_read_failed", "path=\"%s\"",
                in_fragment.pack_path.c_str());
      close(fd);
      return -1;
    }

    StepackFragment fragment;
    fragment.pack_path = in_fragment.pack_path;
    fragment.fd = fd;
    fragment.nslc_start = in_fragment.nslc_start;
    fragment.nslc_count = in_fragment.nslc_count;
    fragment.batch_nslc_count = in_fragment.batch_nslc_count;
    fragment.payload_offset = in_fragment.payload_offset;
    fragment.payload_bytes = in_fragment.payload_bytes;
    fragment.step_bytes = in_fragment.step_bytes;
    fragment.pitch_step_bytes = in_fragment.pitch_step_bytes;
    fragment.nslc_step_bytes = in_fragment.nslc_step_bytes;
    work->stepack_fragments.push_back(fragment);

    if (append_stepack_fragment_nslcs(work,
                                      work->stepack_fragments.size() - 1,
                                      &nslcs) != 0)
      return -1;

    if (i == 0)
    {
      shape->nspec = (int)header.nspec;
      shape->nstep = (int)header.nstep;
      shape->nfft = 2 * ((int)header.nspec - 1);
      shape->dt = header.dt;
      shape->df = header.df;
      work->logical_step_bytes = 0;
      work->logical_payload_bytes = 0;
    }
  }

  std::stable_sort(nslcs.begin(), nslcs.end(),
                   [](const StepackNslcTemp &a, const StepackNslcTemp &b) {
                     if (a.meta.gnsl_id != b.meta.gnsl_id)
                       return a.meta.gnsl_id < b.meta.gnsl_id;
                     return a.original_order < b.original_order;
                   });

  work->specs.clear();
  work->nslc_locators.clear();
  work->specs.reserve(nslcs.size());
  work->nslc_locators.reserve(nslcs.size());
  for (size_t i = 0; i < nslcs.size(); ++i)
  {
    work->specs.push_back(nslcs[i].meta);
    work->nslc_locators.push_back(nslcs[i].locator);
  }

  shape->num_channels = 0;
  shape->half_cc = 0;
  shape->cc_size = 0;
  if (!checked_mul_size((size_t)shape->nspec, sizeof(complex), &shape->step_bytes) ||
      !checked_mul_size((size_t)shape->nstep, (size_t)shape->nspec, &shape->vec_count) ||
      !checked_mul_size(shape->vec_count, sizeof(complex), &shape->vec_bytes) ||
      !checked_mul_size(work->specs.size(), shape->step_bytes, &work->logical_step_bytes) ||
      !checked_mul_size((size_t)shape->nstep, work->logical_step_bytes,
                        &work->logical_payload_bytes))
  {
    return -1;
  }
  work->num_channels = infer_num_channels(work);
  if (work->num_channels == 0 ||
      (expected->num_channels > 0 && work->num_channels != expected->num_channels))
  {
    if (expected->num_channels > 0 && work->num_channels != expected->num_channels)
    {
      LOG_ERROR("timestamp_channel_count_mismatch",
                "timestamp=\"%s\" num_channels=%zu expected=%zu",
                work->timestamp.c_str(), work->num_channels, expected->num_channels);
    }
    return -1;
  }
  shape->num_channels = work->num_channels;

  LOG_INFO("stepack_timestamp_opened",
           "timestamp=\"%s\" fragments=%zu nslcs=%zu",
           work->timestamp.c_str(), work->stepack_fragments.size(),
           work->specs.size());
  return 0;
}

int open_timestamp_work(const TimestampInput &input,
                        TimestampWork *work,
                        const RuntimeShape *expected,
                        RuntimeShape *shape)
{
  close_timestamp_work(work);
  *work = TimestampWork();
  if (open_stepack_timestamp_work(input, work, expected, shape) != 0)
  {
    close_timestamp_work(work);
    return -1;
  }
  return 0;
}

void close_timestamp_work(TimestampWork *work)
{
  if (work)
  {
    for (size_t i = 0; i < work->stepack_fragments.size(); ++i)
    {
      if (work->stepack_fragments[i].fd >= 0)
      {
        close(work->stepack_fragments[i].fd);
        work->stepack_fragments[i].fd = -1;
      }
    }
  }
}

static bool copy_stepack_payload_to_cache(TimestampWork *work)
{
  if (!work ||
      work->stepack_fragments.empty() ||
      work->nslc_locators.size() != work->specs.size() ||
      work->logical_step_bytes == 0 ||
      work->logical_payload_bytes == 0 ||
      work->logical_payload_bytes % work->logical_step_bytes != 0)
    return false;

  size_t nstep = work->logical_payload_bytes / work->logical_step_bytes;
  for (size_t step_idx = 0; step_idx < nstep; ++step_idx)
  {
    size_t local_begin = 0;
    while (local_begin < work->nslc_locators.size())
    {
      const NslcLocator first = work->nslc_locators[local_begin];
      if (first.fragment_index >= work->stepack_fragments.size())
      {
        LOG_WARN("stepack_cache_nslc_locator_invalid",
                 "timestamp=\"%s\" nslc_index=%zu fragment=%zu fragments=%zu fallback=pread",
                 work->timestamp.c_str(), local_begin, first.fragment_index,
                 work->stepack_fragments.size());
        return false;
      }

      size_t local_end = local_begin + 1;
      while (local_end < work->nslc_locators.size())
      {
        const NslcLocator &prev = work->nslc_locators[local_end - 1];
        const NslcLocator &cur = work->nslc_locators[local_end];
        if (cur.fragment_index != first.fragment_index ||
            cur.fragment_local_index != prev.fragment_local_index + 1)
          break;
        ++local_end;
      }

      const StepackFragment &fragment = work->stepack_fragments[first.fragment_index];
      size_t expected_logical_step_bytes = 0;
      if (fragment.fd < 0 ||
          fragment.nslc_step_bytes == 0 ||
          !checked_mul_size(fragment.nslc_step_bytes, work->specs.size(),
                            &expected_logical_step_bytes) ||
          expected_logical_step_bytes != work->logical_step_bytes ||
          first.fragment_local_index >= fragment.nslc_count)
      {
        LOG_WARN("stepack_cache_fragment_invalid",
                 "timestamp=\"%s\" path=\"%s\" fd=%d nslc_step_bytes=%zu logical_step_bytes=%zu fallback=pread",
                 work->timestamp.c_str(), fragment.pack_path.c_str(),
                 fragment.fd, fragment.nslc_step_bytes, work->logical_step_bytes);
        return false;
      }

      size_t nslc_index = fragment.nslc_start + first.fragment_local_index;
      size_t step_offset = 0;
      size_t nslc_offset = 0;
      size_t payload_offset = 0;
      size_t absolute_offset = 0;
      size_t read_bytes = 0;
      size_t cache_step_offset = 0;
      size_t cache_nslc_offset = 0;
      size_t cache_offset = 0;

      if (!checked_mul_size(step_idx, fragment.pitch_step_bytes, &step_offset) ||
          !checked_mul_size(nslc_index, fragment.nslc_step_bytes, &nslc_offset) ||
          !checked_add_size(step_offset, nslc_offset, &payload_offset) ||
          !checked_add_size(fragment.payload_offset, payload_offset, &absolute_offset) ||
          !checked_mul_size(local_end - local_begin, fragment.nslc_step_bytes, &read_bytes) ||
          !checked_mul_size(step_idx, work->logical_step_bytes, &cache_step_offset) ||
          !checked_mul_size(local_begin, fragment.nslc_step_bytes, &cache_nslc_offset) ||
          !checked_add_size(cache_step_offset, cache_nslc_offset, &cache_offset) ||
          cache_offset > work->logical_payload_bytes ||
          read_bytes > work->logical_payload_bytes - cache_offset)
      {
        LOG_WARN("stepack_cache_offset_overflow",
                 "timestamp=\"%s\" path=\"%s\" step=%zu fallback=pread",
                 work->timestamp.c_str(), fragment.pack_path.c_str(), step_idx);
        return false;
      }

      char *target = (char *)work->payload_cache.data() + cache_offset;
      if (read_exact_at(fragment.fd, absolute_offset, target, read_bytes) != 0)
      {
        LOG_WARN("stepack_cache_read_failed",
                 "timestamp=\"%s\" path=\"%s\" step=%zu nslc=%zu count=%zu error=\"%s\" fallback=pread",
                 work->timestamp.c_str(), fragment.pack_path.c_str(), step_idx,
                 nslc_index, local_end - local_begin, strerror(errno));
        return false;
      }
      local_begin = local_end;
    }
  }
  return true;
}

bool try_cache_timestamp_payload(TimestampWork *work)
{
  if (!work || work->payload_cache_enabled)
    return work && work->payload_cache_enabled;
  if (work->logical_payload_bytes == 0 ||
      work->logical_payload_bytes % sizeof(complex) != 0)
    return false;

  size_t values = work->logical_payload_bytes / sizeof(complex);
  try
  {
    work->payload_cache.resize(values);
  }
  catch (...)
  {
    LOG_WARN("input_cache_alloc_failed",
             "timestamp=\"%s\" bytes_gib=%.3f fallback=pread",
             work->timestamp.c_str(), bytes_to_gib(work->logical_payload_bytes));
    work->payload_cache.clear();
    return false;
  }

  bool ok = copy_stepack_payload_to_cache(work);

  if (!ok)
  {
    work->payload_cache.clear();
    return false;
  }

  work->payload_cache_enabled = true;
  LOG_INFO("input_cache_ready",
           "timestamp=\"%s\" mode=%s bytes_gib=%.3f fragments=%zu",
           work->timestamp.c_str(),
           "stepack_logical_host_payload",
           bytes_to_gib(work->logical_payload_bytes),
           work->stepack_fragments.size());
  return true;
}

bool finalize_shape(RuntimeShape *shape, float cc_length)
{
  size_t tmp = 0;
  if (shape->nspec < 2 || shape->nstep <= 0 ||
      shape->dt <= 0.0f || cc_length <= 0.0f)
    return false;
  shape->nfft = 2 * (shape->nspec - 1);
  shape->half_cc = (int)std::llround((double)cc_length / (double)shape->dt);
  if (shape->half_cc >= shape->nspec)
    shape->half_cc = shape->nspec - 1;
  shape->cc_size = 2 * shape->half_cc + 1;
  if (shape->cc_size <= 0 || shape->cc_size > shape->nfft)
    return false;
  if (!checked_mul_size((size_t)shape->nspec, sizeof(complex), &shape->step_bytes))
    return false;
  if (!checked_mul_size((size_t)shape->nstep, (size_t)shape->nspec, &shape->vec_count))
    return false;
  if (!checked_mul_size(shape->vec_count, sizeof(complex), &tmp))
    return false;
  shape->vec_bytes = tmp;
  return true;
}

void load_job_step_input(const TimestampWork *work,
                         const RuntimeShape *shape,
                         const std::vector<size_t> &meta_indices,
                         size_t step_idx,
                         complex *dst)
{
  if (step_idx >= (size_t)shape->nstep ||
      work->nslc_locators.size() != work->specs.size())
  {
    LOG_ERROR("stepack_step_read_invalid", "timestamp=\"%s\" step=%zu nstep=%d",
              work->timestamp.c_str(), step_idx, shape->nstep);
    exit(1);
  }

  if (work->payload_cache_enabled)
  {
    size_t local_begin = 0;
    while (local_begin < meta_indices.size())
    {
      size_t first_meta = meta_indices[local_begin];
      if (first_meta >= work->specs.size())
      {
        LOG_ERROR("stepack_cache_nslc_index_oob",
                  "timestamp=\"%s\" nslc_index=%zu nslc_count=%zu",
                  work->timestamp.c_str(), first_meta, work->specs.size());
        exit(1);
      }

      size_t local_end = local_begin + 1;
      while (local_end < meta_indices.size() &&
             meta_indices[local_end] == meta_indices[local_end - 1] + 1)
        ++local_end;

      size_t step_offset = 0;
      size_t nslc_offset = 0;
      size_t payload_offset = 0;
      size_t read_bytes = 0;
      if (!checked_mul_size(step_idx, work->logical_step_bytes, &step_offset) ||
          !checked_mul_size(first_meta, shape->step_bytes, &nslc_offset) ||
          !checked_add_size(step_offset, nslc_offset, &payload_offset) ||
          !checked_mul_size(local_end - local_begin, shape->step_bytes, &read_bytes) ||
          payload_offset > work->logical_payload_bytes ||
          read_bytes > work->logical_payload_bytes - payload_offset)
      {
        LOG_ERROR("stepack_cache_step_offset_overflow",
                  "timestamp=\"%s\" step=%zu nslc=%zu",
                  work->timestamp.c_str(), step_idx, first_meta);
        exit(1);
      }

      complex *target = dst + local_begin * (size_t)shape->nspec;
      memcpy(target, (const char *)work->payload_cache.data() + payload_offset,
             read_bytes);
      local_begin = local_end;
    }
    return;
  }

  size_t local_begin = 0;
  while (local_begin < meta_indices.size())
  {
    size_t first_meta = meta_indices[local_begin];
    if (first_meta >= work->nslc_locators.size())
    {
      LOG_ERROR("stepack_nslc_index_oob",
                "timestamp=\"%s\" nslc_index=%zu nslc_count=%zu",
                work->timestamp.c_str(), first_meta, work->nslc_locators.size());
      exit(1);
    }

    NslcLocator first = work->nslc_locators[first_meta];
    if (first.fragment_index >= work->stepack_fragments.size())
    {
      LOG_ERROR("stepack_fragment_index_oob",
                "timestamp=\"%s\" fragment=%zu fragments=%zu",
                work->timestamp.c_str(), first.fragment_index,
                work->stepack_fragments.size());
      exit(1);
    }

    size_t local_end = local_begin + 1;
    while (local_end < meta_indices.size())
    {
      size_t prev_meta = meta_indices[local_end - 1];
      size_t cur_meta = meta_indices[local_end];
      if (prev_meta >= work->nslc_locators.size() ||
          cur_meta >= work->nslc_locators.size())
        break;
      const NslcLocator &prev = work->nslc_locators[prev_meta];
      const NslcLocator &cur = work->nslc_locators[cur_meta];
      if (cur.fragment_index != first.fragment_index ||
          cur.fragment_local_index != prev.fragment_local_index + 1)
        break;
      ++local_end;
    }

    const StepackFragment &fragment = work->stepack_fragments[first.fragment_index];
    if (fragment.fd < 0 || fragment.nslc_step_bytes != shape->step_bytes)
    {
      LOG_ERROR("stepack_fragment_invalid",
                "timestamp=\"%s\" path=\"%s\" fd=%d nslc_step_bytes=%zu expected=%zu",
                work->timestamp.c_str(), fragment.pack_path.c_str(),
                fragment.fd, fragment.nslc_step_bytes, shape->step_bytes);
      exit(1);
    }

    size_t nslc_index = fragment.nslc_start + first.fragment_local_index;
    size_t step_offset = 0;
    size_t nslc_offset = 0;
    size_t payload_offset = 0;
    size_t absolute_offset = 0;
    size_t read_bytes = 0;
    if (!checked_mul_size(step_idx, fragment.pitch_step_bytes, &step_offset) ||
        !checked_mul_size(nslc_index, fragment.nslc_step_bytes, &nslc_offset) ||
        !checked_add_size(step_offset, nslc_offset, &payload_offset) ||
        !checked_add_size(fragment.payload_offset, payload_offset, &absolute_offset) ||
        !checked_mul_size(local_end - local_begin, shape->step_bytes, &read_bytes))
    {
      LOG_ERROR("stepack_step_offset_overflow",
                "timestamp=\"%s\" path=\"%s\" step=%zu",
                work->timestamp.c_str(), fragment.pack_path.c_str(), step_idx);
      exit(1);
    }

    complex *target = dst + local_begin * (size_t)shape->nspec;
    if (read_exact_at(fragment.fd, absolute_offset, target, read_bytes) != 0)
    {
      LOG_ERROR("stepack_step_read_failed",
                "path=\"%s\" timestamp=\"%s\" step=%zu nslc=%zu count=%zu error=\"%s\"",
                fragment.pack_path.c_str(), work->timestamp.c_str(), step_idx,
                nslc_index,
                local_end - local_begin, strerror(errno));
      exit(1);
    }
    local_begin = local_end;
  }
}
