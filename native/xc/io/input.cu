#include "input.hpp"
#include "logger.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <limits.h>
#include <stdint.h>
#include <strings.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static const uint32_t kXcspecVersion = 1;
static const uint32_t kXcspecEndianTag = 0x01020304;
static const uint32_t kXcspecHeaderSize = 256;
static const uint32_t kXcspecSourceEntrySize = 128;
static const uint32_t kXcspecLayoutStepFileFreq = 1;
static const uint32_t kXcspecDtypeComplex64 = 1;
static const uint32_t kXcspecStringAsciiNul = 1;

#pragma pack(push, 1)
struct XcspecHeaderRaw
{
  char magic[8];
  uint32_t version;
  uint32_t endian_tag;
  uint32_t header_size;
  uint32_t source_entry_size;
  uint64_t source_table_offset;
  uint32_t source_count;
  uint32_t file_count;
  uint64_t payload_offset;
  uint32_t layout;
  uint32_t dtype;
  uint32_t string_encoding;
  uint32_t reserved0;
  char timestamp[64];
  uint32_t nstep;
  uint32_t nspec;
  uint32_t nfft;
  uint32_t reserved1;
  float dt;
  float df;
  uint64_t step_bytes;
  uint64_t payload_bytes;
  uint64_t manifest_hash_u64;
  uint64_t source_table_bytes;
  unsigned char reserved2[72];
};

struct XcspecSourceEntryRaw
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
#pragma pack(pop)

static_assert(sizeof(XcspecHeaderRaw) == kXcspecHeaderSize, "bad .xcspec header ABI");
static_assert(sizeof(XcspecSourceEntryRaw) == kXcspecSourceEntrySize, "bad .xcspec source ABI");

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

static std::string basename_of(const std::string &path)
{
  size_t pos = path.find_last_of("/\\");
  return pos == std::string::npos ? path : path.substr(pos + 1);
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

static std::string timestamp_from_xcspec_path(const std::string &path)
{
  std::string base = basename_of(path);
  const char *suffix = ".xcspec";
  size_t suffix_len = strlen(suffix);
  if (base.size() >= suffix_len &&
      strcasecmp(base.c_str() + base.size() - suffix_len, suffix) == 0)
    base.resize(base.size() - suffix_len);
  return base;
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

static size_t parse_size_hint(const std::string &text)
{
  char *end = NULL;
  unsigned long long value = 0;
  errno = 0;
  if (text.empty())
    return 0;
  value = strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' || value == 0)
    return 0;
  if (value > (unsigned long long)std::numeric_limits<size_t>::max())
    return std::numeric_limits<size_t>::max();
  return (size_t)value;
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

static bool valid_xcspec_header(const XcspecHeaderRaw &h,
                                const RuntimeShape *expected,
                                size_t file_size,
                                const char *path)
{
  size_t source_bytes = 0;
  size_t source_end = 0;
  size_t step_bytes = 0;
  size_t payload_bytes = 0;
  size_t payload_end = 0;

  if (sizeof(complex) != 8 ||
      memcmp(h.magic, "FXCXSPEC", 7) != 0 ||
      h.version != kXcspecVersion ||
      h.endian_tag != kXcspecEndianTag ||
      h.header_size != kXcspecHeaderSize ||
      h.source_entry_size != kXcspecSourceEntrySize ||
      h.layout != kXcspecLayoutStepFileFreq ||
      h.dtype != kXcspecDtypeComplex64 ||
      h.string_encoding != kXcspecStringAsciiNul)
  {
    LOG_ERROR("xcspec_format_unsupported", "path=\"%s\"", path);
    return false;
  }

  if (h.file_count == 0 || h.source_count != h.file_count ||
      h.nstep == 0 || h.nspec < 2 || h.dt <= 0.0f || h.df <= 0.0f ||
      h.nfft != 2 * (h.nspec - 1))
  {
    LOG_ERROR("xcspec_shape_invalid", "path=\"%s\"", path);
    return false;
  }

  if (expected->nspec > 0 &&
      ((int)h.nspec != expected->nspec ||
       (int)h.nstep != expected->nstep ||
       fabsf(h.dt - expected->dt) > 1e-6f ||
       fabsf(h.df - expected->df) > 1e-9f))
  {
    LOG_ERROR("xcspec_shape_mismatch", "path=\"%s\"", path);
    return false;
  }

  if (!checked_mul_size((size_t)h.file_count, sizeof(XcspecSourceEntryRaw), &source_bytes) ||
      source_bytes != (size_t)h.source_table_bytes ||
      !checked_add_size((size_t)h.source_table_offset, source_bytes, &source_end) ||
      h.source_table_offset < h.header_size || h.payload_offset < source_end ||
      !checked_mul_size((size_t)h.file_count, (size_t)h.nspec, &step_bytes) ||
      !checked_mul_size(step_bytes, sizeof(complex), &step_bytes) ||
      step_bytes != (size_t)h.step_bytes ||
      !checked_mul_size((size_t)h.nstep, step_bytes, &payload_bytes) ||
      payload_bytes != (size_t)h.payload_bytes ||
      !checked_add_size((size_t)h.payload_offset, payload_bytes, &payload_end) ||
      payload_end > file_size)
  {
    LOG_ERROR("xcspec_layout_invalid", "path=\"%s\"", path);
    return false;
  }
  return true;
}

static int load_source_table(int fd,
                             const XcspecHeaderRaw &header,
                             TimestampWork *work)
{
  work->specs.clear();
  work->specs.reserve(header.file_count);

  for (size_t i = 0; i < (size_t)header.file_count; ++i)
  {
    XcspecSourceEntryRaw row;
    size_t offset = (size_t)header.source_table_offset + i * sizeof(row);
    if (read_exact_at(fd, offset, &row, sizeof(row)) != 0)
    {
      LOG_ERROR("xcspec_source_table_read_failed", "path=\"%s\" error=\"%s\"",
                work->xcspec_path.c_str(), strerror(errno));
      return -1;
    }
    if (row.file_index != i || row.nsl_id == 0)
    {
      LOG_ERROR("xcspec_source_entry_invalid", "path=\"%s\" entry=%zu",
                work->xcspec_path.c_str(), i);
      return -1;
    }

    SpecMeta meta;
    meta.path = work->xcspec_path;
    meta.group = std::to_string(row.nsl_id);
    meta.network = fixed_string(row.network, sizeof(row.network));
    meta.station = fixed_string(row.station, sizeof(row.station));
    meta.location = fixed_string(row.location, sizeof(row.location));
    meta.component = fixed_string(row.component, sizeof(row.component));
    meta.stla = row.stla;
    meta.stlo = row.stlo;
    meta.gnsl_id = (int)row.nsl_id;
    work->specs.push_back(meta);
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
      LOG_ERROR("xcspec_source_table_unsorted", "path=\"%s\" nsl_id=%d previous=%d",
                work->xcspec_path.c_str(), id, last_id);
      return 0;
    }
    while (i < work->specs.size() && work->specs[i].gnsl_id == id)
      ++i;

    size_t count = i - begin;
    if (expected == 0)
      expected = count;
    else if (count != expected)
    {
      LOG_ERROR("xcspec_channel_group_incomplete",
                "path=\"%s\" nsl_id=%d channels=%zu expected=%zu",
                work->xcspec_path.c_str(), id, count, expected);
      return 0;
    }
    last_id = id;
  }
  return expected;
}

std::vector<TimestampInput> load_timestamp_inputs(const ARGUTYPE *args)
{
  std::vector<TimestampInput> inputs;
  if (args->single_timestamp_path)
  {
    TimestampInput input;
    input.xcspec_path = args->single_timestamp_path;
    input.timestamp = timestamp_from_xcspec_path(input.xcspec_path);
    inputs.push_back(input);
    return inputs;
  }

  FILE *fp = fopen(args->timestamp_index_path, "r");
  if (!fp)
  {
    LOG_ERROR("xcspec_index_open_failed", "path=\"%s\" error=\"%s\"",
              args->timestamp_index_path, strerror(errno));
    exit(1);
  }

  char line[PATH_MAX * 4];
  std::vector<std::string> header;
  std::string base_dir = dirname_of(args->timestamp_index_path);
  int timestamp_col = -1;
  int xcspec_col = -1;
  int manifest_col = -1;
  int file_count_col = -1;

  while (fgets(line, sizeof(line), fp))
  {
    std::string text = trim(line);
    if (text.empty() || text[0] == '#')
      continue;

    std::vector<std::string> fields = split_tab(text);
    if (header.empty())
    {
      header = fields;
      timestamp_col = column_index(header, "timestamp");
      xcspec_col = column_index(header, "xcspec_path");
      manifest_col = column_index(header, "manifest_path");
      file_count_col = column_index(header, "file_count");
      if (timestamp_col < 0 || xcspec_col < 0)
      {
        LOG_ERROR("xcspec_index_columns_missing", "path=\"%s\"",
                  args->timestamp_index_path);
        fclose(fp);
        exit(1);
      }
      continue;
    }

    if (timestamp_col >= (int)fields.size() || xcspec_col >= (int)fields.size())
    {
      LOG_ERROR("xcspec_index_row_malformed", "row=\"%s\"", text.c_str());
      fclose(fp);
      exit(1);
    }

    TimestampInput input;
    input.timestamp = fields[timestamp_col];
    input.xcspec_path = resolve_relative_path(fields[xcspec_col], base_dir);
    if (manifest_col >= 0 && manifest_col < (int)fields.size())
      input.manifest_path = resolve_relative_path(fields[manifest_col], base_dir);
    if (file_count_col >= 0 && file_count_col < (int)fields.size())
      input.file_count_hint = parse_size_hint(fields[file_count_col]);
    inputs.push_back(input);
  }
  fclose(fp);
  return inputs;
}

int open_timestamp_work(const TimestampInput &input,
                        TimestampWork *work,
                        const RuntimeShape *expected,
                        RuntimeShape *shape)
{
  close_timestamp_work(work);
  *work = TimestampWork();
  work->input_path = input.xcspec_path;
  work->timestamp = input.timestamp;
  work->xcspec_path = input.xcspec_path;
  work->manifest_path = input.manifest_path;

  int fd = open(work->xcspec_path.c_str(), O_RDONLY);
  if (fd < 0)
  {
    LOG_ERROR("xcspec_open_failed", "path=\"%s\" error=\"%s\"",
              work->xcspec_path.c_str(), strerror(errno));
    return -1;
  }
  work->xcspec_fd = fd;

  struct stat st;
  XcspecHeaderRaw header;
  if (fstat(fd, &st) != 0 ||
      read_exact_at(fd, 0, &header, sizeof(header)) != 0 ||
      !valid_xcspec_header(header, expected, (size_t)st.st_size,
                           work->xcspec_path.c_str()))
  {
    LOG_ERROR("xcspec_header_read_failed", "path=\"%s\"",
              work->xcspec_path.c_str());
    close_timestamp_work(work);
    return -1;
  }

  std::string header_timestamp = fixed_string(header.timestamp, sizeof(header.timestamp));
  if (!input.timestamp.empty() && input.timestamp != header_timestamp)
  {
    LOG_WARN("xcspec_timestamp_mismatch",
             "index_timestamp=\"%s\" header_timestamp=\"%s\" path=\"%s\" action=use_header",
             input.timestamp.c_str(), header_timestamp.c_str(), work->xcspec_path.c_str());
  }
  work->timestamp = header_timestamp.empty() ? timestamp_from_xcspec_path(work->xcspec_path)
                                             : header_timestamp;
  work->xcspec_payload_offset = (size_t)header.payload_offset;
  work->xcspec_step_bytes = (size_t)header.step_bytes;
  work->xcspec_payload_bytes = (size_t)header.payload_bytes;
  work->manifest_hash_u64 = header.manifest_hash_u64;

  shape->nspec = (int)header.nspec;
  shape->nstep = (int)header.nstep;
  shape->nfft = (int)header.nfft;
  shape->dt = header.dt;
  shape->df = header.df;
  shape->num_channels = 0;
  shape->half_cc = 0;
  shape->cc_size = 0;
  if (!checked_mul_size((size_t)shape->nspec, sizeof(complex), &shape->step_bytes) ||
      !checked_mul_size((size_t)shape->nstep, (size_t)shape->nspec, &shape->vec_count) ||
      !checked_mul_size(shape->vec_count, sizeof(complex), &shape->vec_bytes) ||
      load_source_table(fd, header, work) != 0)
  {
    close_timestamp_work(work);
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
    close_timestamp_work(work);
    return -1;
  }
  shape->num_channels = work->num_channels;
  return 0;
}

void close_timestamp_work(TimestampWork *work)
{
  if (work && work->xcspec_fd >= 0)
  {
    close(work->xcspec_fd);
    work->xcspec_fd = -1;
  }
}

bool try_cache_timestamp_payload(TimestampWork *work)
{
  if (!work || work->payload_cache_enabled)
    return work && work->payload_cache_enabled;
  if (work->xcspec_fd < 0 || work->xcspec_payload_bytes == 0 ||
      work->xcspec_payload_bytes % sizeof(complex) != 0)
    return false;

  size_t values = work->xcspec_payload_bytes / sizeof(complex);
  try
  {
    work->payload_cache.resize(values);
  }
  catch (...)
  {
    LOG_WARN("input_cache_alloc_failed",
             "timestamp=\"%s\" bytes_gib=%.3f fallback=pread",
             work->timestamp.c_str(), bytes_to_gib(work->xcspec_payload_bytes));
    work->payload_cache.clear();
    return false;
  }

  if (read_exact_at(work->xcspec_fd,
                    work->xcspec_payload_offset,
                    work->payload_cache.data(),
                    work->xcspec_payload_bytes) != 0)
  {
    LOG_WARN("input_cache_read_failed",
             "timestamp=\"%s\" error=\"%s\" fallback=pread",
             work->timestamp.c_str(), strerror(errno));
    work->payload_cache.clear();
    return false;
  }

  work->payload_cache_enabled = true;
  LOG_INFO("input_cache_ready", "timestamp=\"%s\" mode=host_payload bytes_gib=%.3f",
           work->timestamp.c_str(), bytes_to_gib(work->xcspec_payload_bytes));
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
  if (work->xcspec_fd < 0 || step_idx >= (size_t)shape->nstep)
  {
    LOG_ERROR("xcspec_step_read_invalid", "timestamp=\"%s\" step=%zu nstep=%d",
              work->timestamp.c_str(), step_idx, shape->nstep);
    exit(1);
  }

  size_t local_begin = 0;
  while (local_begin < meta_indices.size())
  {
    size_t first_file = meta_indices[local_begin];
    size_t local_end = local_begin + 1;
    while (local_end < meta_indices.size() &&
           meta_indices[local_end] == meta_indices[local_end - 1] + 1)
      ++local_end;

    size_t payload_offset = step_idx * work->xcspec_step_bytes +
                            first_file * shape->step_bytes;
    size_t read_bytes = (local_end - local_begin) * shape->step_bytes;
    complex *target = dst + local_begin * (size_t)shape->nspec;
    if (first_file >= work->specs.size())
    {
      LOG_ERROR("xcspec_source_index_oob",
                "path=\"%s\" source_index=%zu file_count=%zu",
                work->xcspec_path.c_str(), first_file, work->specs.size());
      exit(1);
    }
    if (work->payload_cache_enabled)
    {
      memcpy(target, (const char *)work->payload_cache.data() + payload_offset, read_bytes);
    }
    else if (read_exact_at(work->xcspec_fd,
                           work->xcspec_payload_offset + payload_offset,
                           target,
                           read_bytes) != 0)
    {
      LOG_ERROR("xcspec_step_read_failed",
                "path=\"%s\" step=%zu file=%zu count=%zu error=\"%s\"",
                work->xcspec_path.c_str(), step_idx, first_file,
                local_end - local_begin, strerror(errno));
      exit(1);
    }
    local_begin = local_end;
  }
}
