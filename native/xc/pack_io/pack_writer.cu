#include "pack_writer.hpp"

#include "fs.h"
#include "logger.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static std::string clean_tsv_field(const std::string &text)
{
  std::string out = text;
  for (size_t i = 0; i < out.size(); ++i)
  {
    if (out[i] == '\t' || out[i] == '\r' || out[i] == '\n')
      out[i] = ' ';
  }
  return out;
}

std::string xc_pack_root_dir(const char *output_dir)
{
  if (!output_dir || output_dir[0] == '\0')
    return std::string();
  return std::string(output_dir) + "/xcpack";
}

int xc_pack_prepare_root(const char *output_dir)
{
  std::string root = xc_pack_root_dir(output_dir);
  if (root.empty())
    return -1;
  if (mkdir_p(root.c_str(), 0755) != 0)
  {
    LOG_ERROR("xcpack_dir_create_failed", "path=\"%s\" error=\"%s\"",
              root.c_str(), strerror(errno));
    return -1;
  }
  return 0;
}

static int build_pack_paths(XcPackWriter *writer)
{
  char pack_path[PATH_MAX];
  char tsv_path[PATH_MAX];
  int n = snprintf(pack_path, sizeof(pack_path),
                   "%s/%s.i%06zu.j%06zu.w%03zu.p%03zu.xcpack",
                   writer->root_dir.c_str(), writer->timestamp.c_str(),
                   writer->anchor_block, writer->target_begin_block,
                   writer->worker_id, writer->part_id);
  if (n < 0 || n >= (int)sizeof(pack_path))
    return -1;
  n = snprintf(tsv_path, sizeof(tsv_path),
               "%s/%s.i%06zu.j%06zu.w%03zu.p%03zu.tsv",
               writer->root_dir.c_str(), writer->timestamp.c_str(),
               writer->anchor_block, writer->target_begin_block,
               writer->worker_id, writer->part_id);
  if (n < 0 || n >= (int)sizeof(tsv_path))
    return -1;
  writer->pack_path = pack_path;
  writer->tsv_path = tsv_path;
  return 0;
}

static int open_part(XcPackWriter *writer)
{
  if (build_pack_paths(writer) != 0)
  {
    LOG_ERROR("xcpack_path_build_failed",
              "root=\"%s\" timestamp=\"%s\" block_i=%zu block_j=%zu worker=%zu part=%zu",
              writer->root_dir.c_str(), writer->timestamp.c_str(),
              writer->anchor_block, writer->target_begin_block,
              writer->worker_id, writer->part_id);
    return -1;
  }

  writer->pack_fp = fopen(writer->pack_path.c_str(), "ab");
  if (!writer->pack_fp)
  {
    LOG_ERROR("xcpack_open_failed", "path=\"%s\" error=\"%s\"",
              writer->pack_path.c_str(), strerror(errno));
    return -1;
  }
  writer->tsv_fp = fopen(writer->tsv_path.c_str(), "ab");
  if (!writer->tsv_fp)
  {
    LOG_ERROR("xcpack_tsv_open_failed", "path=\"%s\" error=\"%s\"",
              writer->tsv_path.c_str(), strerror(errno));
    fclose(writer->pack_fp);
    writer->pack_fp = NULL;
    return -1;
  }

  if (fseeko(writer->pack_fp, 0, SEEK_END) != 0 ||
      fseeko(writer->tsv_fp, 0, SEEK_END) != 0)
  {
    LOG_ERROR("xcpack_seek_end_failed", "pack=\"%s\" tsv=\"%s\" error=\"%s\"",
              writer->pack_path.c_str(), writer->tsv_path.c_str(), strerror(errno));
    fclose(writer->tsv_fp);
    fclose(writer->pack_fp);
    writer->tsv_fp = NULL;
    writer->pack_fp = NULL;
    return -1;
  }

  off_t tsv_size = ftello(writer->tsv_fp);
  off_t pack_size = ftello(writer->pack_fp);
  if (tsv_size < 0 || pack_size < 0)
  {
    LOG_ERROR("xcpack_tell_failed", "pack=\"%s\" tsv=\"%s\" error=\"%s\"",
              writer->pack_path.c_str(), writer->tsv_path.c_str(), strerror(errno));
    fclose(writer->tsv_fp);
    fclose(writer->pack_fp);
    writer->tsv_fp = NULL;
    writer->pack_fp = NULL;
    return -1;
  }

  try
  {
    writer->pack_buffer.clear();
    writer->tsv_buffer.clear();
    writer->pack_buffer.reserve(kXcPackBinaryBufferBytes);
    writer->tsv_buffer.reserve(kXcPackTsvBufferBytes);
  }
  catch (...)
  {
    LOG_ERROR("xcpack_buffer_reserve_failed",
              "path=\"%s\" binary_buffer_bytes=%zu tsv_buffer_bytes=%zu",
              writer->pack_path.c_str(),
              kXcPackBinaryBufferBytes,
              kXcPackTsvBufferBytes);
    fclose(writer->tsv_fp);
    fclose(writer->pack_fp);
    writer->tsv_fp = NULL;
    writer->pack_fp = NULL;
    return -1;
  }

  if (tsv_size == 0)
  {
    writer->tsv_buffer.append(
        "timestamp\tworker_id\tanchor_block\ttarget_begin_block\ttarget_end_block\t"
        "block_size\tanchor_begin\tanchor_end\ttarget_begin\ttarget_end\t"
        "pack_path\toffset\tbytes\t"
        "src_id\trec_id\tsrc_network\tsrc_station\tsrc_location\tsrc_component\t"
        "rec_network\trec_station\trec_location\trec_component\t"
        "npts\tdt\tdist\taz\tbaz\tfinal_pair_path\n");
  }
  writer->current_bytes = (uint64_t)pack_size;
  return 0;
}

static int flush_part(XcPackWriter *writer)
{
  if (!writer)
    return -1;
  if (writer->pack_fp && !writer->pack_buffer.empty())
  {
    if (fwrite(writer->pack_buffer.data(), writer->pack_buffer.size(), 1,
               writer->pack_fp) != 1)
    {
      LOG_ERROR("xcpack_flush_failed", "path=\"%s\" bytes=%zu error=\"%s\"",
                writer->pack_path.c_str(), writer->pack_buffer.size(), strerror(errno));
      return -1;
    }
    writer->pack_buffer.clear();
  }
  if (writer->tsv_fp && !writer->tsv_buffer.empty())
  {
    if (fwrite(writer->tsv_buffer.data(), writer->tsv_buffer.size(), 1,
               writer->tsv_fp) != 1)
    {
      LOG_ERROR("xcpack_tsv_flush_failed", "path=\"%s\" bytes=%zu error=\"%s\"",
                writer->tsv_path.c_str(), writer->tsv_buffer.size(), strerror(errno));
      return -1;
    }
    writer->tsv_buffer.clear();
  }
  return 0;
}

static int close_part(XcPackWriter *writer)
{
  int rc = flush_part(writer);
  if (writer->pack_fp)
  {
    if (fclose(writer->pack_fp) != 0)
    {
      LOG_ERROR("xcpack_close_failed", "path=\"%s\" error=\"%s\"",
                writer->pack_path.c_str(), strerror(errno));
      rc = -1;
    }
  }
  if (writer->tsv_fp)
  {
    if (fclose(writer->tsv_fp) != 0)
    {
      LOG_ERROR("xcpack_tsv_close_failed", "path=\"%s\" error=\"%s\"",
                writer->tsv_path.c_str(), strerror(errno));
      rc = -1;
    }
  }
  writer->pack_fp = NULL;
  writer->tsv_fp = NULL;
  writer->pack_buffer.clear();
  writer->tsv_buffer.clear();
  return rc;
}

int xc_pack_writer_open(XcPackWriter *writer,
                        const char *root_dir,
                        const char *timestamp,
                        size_t worker_id,
                        const RowBatchJob *job,
                        uint64_t max_pack_bytes)
{
  if (!writer || !root_dir || !timestamp || !job)
    return -1;
  *writer = XcPackWriter();
  writer->root_dir = root_dir;
  writer->timestamp = timestamp;
  writer->worker_id = worker_id;
  writer->anchor_block = job->anchor_block;
  writer->target_begin_block = job->target_begin_block;
  writer->target_end_block = job->target_end_block;
  writer->block_size = job->block_size;
  writer->max_pack_bytes = max_pack_bytes == 0 ? kXcPackDefaultMaxBytes : max_pack_bytes;
  if (mkdir_p(root_dir, 0755) != 0)
  {
    LOG_ERROR("xcpack_dir_create_failed", "path=\"%s\" error=\"%s\"",
              root_dir, strerror(errno));
    return -1;
  }
  return open_part(writer);
}

int xc_pack_writer_append(XcPackWriter *writer,
                          const XcPackRecordMeta *meta,
                          const void *record,
                          uint64_t record_bytes)
{
  if (!writer || !meta || !record || record_bytes == 0)
    return -1;
  if (!writer->pack_fp || !writer->tsv_fp)
    return -1;
  if (record_bytes > (uint64_t)SIZE_MAX ||
      record_bytes > UINT64_MAX - writer->current_bytes)
  {
    LOG_ERROR("xcpack_record_size_invalid",
              "path=\"%s\" record_bytes=%" PRIu64 " current_bytes=%" PRIu64,
              writer->pack_path.c_str(), record_bytes, writer->current_bytes);
    return -1;
  }
  if (writer->current_bytes > 0 &&
      writer->current_bytes + record_bytes > writer->max_pack_bytes)
  {
    if (close_part(writer) != 0)
      return -1;
    ++writer->part_id;
    if (open_part(writer) != 0)
      return -1;
  }

  uint64_t offset = writer->current_bytes;
  std::string timestamp = clean_tsv_field(meta->timestamp);
  std::string pack_path = clean_tsv_field(writer->pack_path);
  std::string src_network = clean_tsv_field(meta->src_network);
  std::string src_station = clean_tsv_field(meta->src_station);
  std::string src_location = clean_tsv_field(meta->src_location);
  std::string src_component = clean_tsv_field(meta->src_component);
  std::string rec_network = clean_tsv_field(meta->rec_network);
  std::string rec_station = clean_tsv_field(meta->rec_station);
  std::string rec_location = clean_tsv_field(meta->rec_location);
  std::string rec_component = clean_tsv_field(meta->rec_component);
  std::string final_pair_path = clean_tsv_field(meta->final_pair_path);
  const char *fmt =
      "%s\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t"
      "%s\t%" PRIu64 "\t%" PRIu64 "\t"
      "%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t"
      "%d\t%.9g\t%.9g\t%.9g\t%.9g\t%s\n";
  int row_len = snprintf(NULL, 0, fmt,
                         timestamp.c_str(),
                         meta->worker_id,
                         meta->anchor_block,
                         meta->target_begin_block,
                         meta->target_end_block,
                         meta->block_size,
                         meta->anchor_begin,
                         meta->anchor_end,
                         meta->target_begin,
                         meta->target_end,
                         pack_path.c_str(),
                         offset,
                         record_bytes,
                         meta->src_id,
                         meta->rec_id,
                         src_network.c_str(),
                         src_station.c_str(),
                         src_location.c_str(),
                         src_component.c_str(),
                         rec_network.c_str(),
                         rec_station.c_str(),
                         rec_location.c_str(),
                         rec_component.c_str(),
                         meta->npts,
                         meta->dt,
                         meta->dist,
                         meta->az,
                         meta->baz,
                         final_pair_path.c_str());
  if (row_len < 0)
  {
    LOG_ERROR("xcpack_tsv_row_format_failed", "path=\"%s\"",
              writer->tsv_path.c_str());
    return -1;
  }
  std::vector<char> row;
  try
  {
    row.resize((size_t)row_len + 1);
  }
  catch (...)
  {
    LOG_ERROR("xcpack_tsv_row_alloc_failed",
              "path=\"%s\" row_bytes=%d",
              writer->tsv_path.c_str(), row_len);
    return -1;
  }
  snprintf(row.data(), row.size(), fmt,
           timestamp.c_str(),
           meta->worker_id,
           meta->anchor_block,
           meta->target_begin_block,
           meta->target_end_block,
           meta->block_size,
           meta->anchor_begin,
           meta->anchor_end,
           meta->target_begin,
           meta->target_end,
           pack_path.c_str(),
           offset,
           record_bytes,
           meta->src_id,
           meta->rec_id,
           src_network.c_str(),
           src_station.c_str(),
           src_location.c_str(),
           src_component.c_str(),
           rec_network.c_str(),
           rec_station.c_str(),
           rec_location.c_str(),
           rec_component.c_str(),
           meta->npts,
           meta->dt,
           meta->dist,
           meta->az,
           meta->baz,
           final_pair_path.c_str());

  try
  {
    writer->pack_buffer.reserve(writer->pack_buffer.size() + (size_t)record_bytes);
    writer->tsv_buffer.reserve(writer->tsv_buffer.size() + (size_t)row_len);
    const char *record_bytes_ptr = (const char *)record;
    writer->pack_buffer.insert(writer->pack_buffer.end(),
                               record_bytes_ptr,
                               record_bytes_ptr + (size_t)record_bytes);
    writer->tsv_buffer.append(row.data(), (size_t)row_len);
  }
  catch (...)
  {
    LOG_ERROR("xcpack_buffer_append_failed",
              "pack=\"%s\" tsv=\"%s\" record_bytes=%" PRIu64 " row_bytes=%d",
              writer->pack_path.c_str(), writer->tsv_path.c_str(),
              record_bytes, row_len);
    return -1;
  }
  writer->current_bytes += record_bytes;

  if (writer->pack_buffer.size() >= kXcPackBinaryBufferBytes ||
      writer->tsv_buffer.size() >= kXcPackTsvBufferBytes)
  {
    if (flush_part(writer) != 0)
      return -1;
  }
  return 0;
}

void xc_pack_writer_close(XcPackWriter *writer)
{
  if (!writer)
    return;
  close_part(writer);
}

static int write_marker(const char *path)
{
  FILE *fp = fopen(path, "w");
  if (!fp)
    return -1;
  if (fprintf(fp, "DONE\n") < 0)
  {
    fclose(fp);
    return -1;
  }
  return fclose(fp) == 0 ? 0 : -1;
}

int xc_pack_write_timestamp_done(const char *root_dir, const char *timestamp)
{
  char path[PATH_MAX];
  if (!root_dir || !timestamp)
    return -1;
  if (mkdir_p(root_dir, 0755) != 0)
  {
    LOG_ERROR("xcpack_dir_create_failed", "path=\"%s\" error=\"%s\"",
              root_dir, strerror(errno));
    return -1;
  }
  int n = snprintf(path, sizeof(path), "%s/%s.done", root_dir, timestamp);
  if (n < 0 || n >= (int)sizeof(path))
    return -1;
  return write_marker(path);
}

int xc_pack_write_success(const char *root_dir)
{
  char path[PATH_MAX];
  if (!root_dir)
    return -1;
  if (mkdir_p(root_dir, 0755) != 0)
  {
    LOG_ERROR("xcpack_dir_create_failed", "path=\"%s\" error=\"%s\"",
              root_dir, strerror(errno));
    return -1;
  }
  int n = snprintf(path, sizeof(path), "%s/_SUCCESS", root_dir);
  if (n < 0 || n >= (int)sizeof(path))
    return -1;
  return write_marker(path);
}

int xc_pack_write_timestamp_done_for_output(const char *output_dir, const char *timestamp)
{
  std::string root = xc_pack_root_dir(output_dir);
  if (root.empty())
    return -1;
  return xc_pack_write_timestamp_done(root.c_str(), timestamp);
}

int xc_pack_write_success_for_output(const char *output_dir)
{
  std::string root = xc_pack_root_dir(output_dir);
  if (root.empty())
    return -1;
  return xc_pack_write_success(root.c_str());
}
