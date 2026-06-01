#ifndef XC_SAC_RECORD_HPP
#define XC_SAC_RECORD_HPP

#include "runtime.hpp"

#include <stddef.h>
#include <string>
#include <vector>

extern "C"
{
#include "sac_header.h"
}

struct XcPackRecordMeta;

bool parse_timestamp_text(const std::string &timestamp, XcTimeData *time_info);

int build_output_path(char *out,
                      size_t out_size,
                      const char *root,
                      const SpecMeta &src,
                      const SpecMeta &rec,
                      bool create_dirs);

void build_sac_header_for_task(SACHEAD *hd,
                               const RuntimeShape *shape,
                               const XcTask &task,
                               const SpecMeta &src,
                               const SpecMeta &rec,
                               const XcTimeData *time_info);

int make_sac_record(std::vector<char> *record,
                    SACHEAD hd,
                    const float *trace,
                    size_t trace_count);

void fill_pack_record_meta(XcPackRecordMeta *meta,
                           const TimestampWork *timestamp,
                           const RuntimeShape *shape,
                           const WorkerConfig *cfg,
                           const RowBatchJob *job,
                           const XcTask &task,
                           const SpecMeta &src,
                           const SpecMeta &rec,
                           const char *final_pair_path);

#endif
