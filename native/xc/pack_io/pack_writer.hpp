#ifndef XC_PACK_WRITER_HPP
#define XC_PACK_WRITER_HPP

#include "runtime.hpp"

#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

static const uint64_t kXcPackDefaultMaxBytes = 4ULL << 30;
static const size_t kXcPackBinaryBufferBytes = 32ULL << 20;
static const size_t kXcPackTsvBufferBytes = 4ULL << 20;

struct XcPackRecordMeta
{
  std::string timestamp;
  size_t worker_id = 0;
  size_t anchor_block = 0;
  size_t target_begin_block = 0;
  size_t target_end_block = 0;
  size_t block_size = 0;
  size_t anchor_begin = 0;
  size_t anchor_end = 0;
  size_t target_begin = 0;
  size_t target_end = 0;
  int src_id = 0;
  int rec_id = 0;
  std::string src_network;
  std::string src_station;
  std::string src_location;
  std::string src_component;
  std::string rec_network;
  std::string rec_station;
  std::string rec_location;
  std::string rec_component;
  int npts = 0;
  float dt = 0.0f;
  float dist = 0.0f;
  float az = 0.0f;
  float baz = 0.0f;
  std::string final_pair_path;
};

struct XcPackWriter
{
  std::string root_dir;
  std::string timestamp;
  size_t worker_id = 0;
  size_t anchor_block = 0;
  size_t target_begin_block = 0;
  size_t target_end_block = 0;
  size_t block_size = 0;
  size_t part_id = 0;
  uint64_t max_pack_bytes = kXcPackDefaultMaxBytes;
  uint64_t current_bytes = 0;
  FILE *pack_fp = NULL;
  FILE *tsv_fp = NULL;
  std::string pack_path;
  std::string tsv_path;
  std::vector<char> pack_buffer;
  std::string tsv_buffer;
};

int xc_pack_writer_open(XcPackWriter *writer,
                        const char *root_dir,
                        const char *timestamp,
                        size_t worker_id,
                        const RowBatchJob *job,
                        uint64_t max_pack_bytes);

int xc_pack_writer_append(XcPackWriter *writer,
                          const XcPackRecordMeta *meta,
                          const void *record,
                          uint64_t record_bytes);

void xc_pack_writer_close(XcPackWriter *writer);

std::string xc_pack_root_dir(const char *output_dir);
int xc_pack_prepare_root(const char *output_dir);
int xc_pack_write_timestamp_done(const char *root_dir, const char *timestamp);
int xc_pack_write_success(const char *root_dir);
int xc_pack_write_timestamp_done_for_output(const char *output_dir, const char *timestamp);
int xc_pack_write_success_for_output(const char *output_dir);

#endif
