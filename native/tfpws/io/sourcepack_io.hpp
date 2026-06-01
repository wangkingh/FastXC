#ifndef TFPWS_SOURCEPACK_IO_HPP
#define TFPWS_SOURCEPACK_IO_HPP

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <map>
#include <queue>
#include <string>
#include <vector>

extern "C"
{
#include "sac.h"
}

struct SourcePackRecord
{
    std::string timestamp;
    std::string path_id;
    int component_slot;
    std::string source_key;
    std::string receiver_key;
    std::string src_id;
    std::string rec_id;
    std::string src_network;
    std::string src_station;
    std::string src_location;
    std::string src_component;
    std::string rec_network;
    std::string rec_station;
    std::string rec_location;
    std::string rec_component;
    std::string dist;
    std::string az;
    std::string baz;
    std::string final_pair_path;
    std::string record_path;
    long long record_offset;
    long long record_bytes;
};

struct TfpwsSourcePackItem
{
    std::string label;
    SourcePackRecord record;
    SACHEAD header;
    unsigned num_segments;
    unsigned nsamples;
    unsigned ngroups;
    float *prestack_data;
    float *linear_stack;
    float *group_trace_weights;
};

std::string sourcepack_trim_copy(const std::string &s);
std::string tfpws_join_path(const std::string &dir, const char *name);
std::string tfpws_absolute_path(const std::string &path);
const char *tfpws_sourcepack_index_header();

int read_sourcepack_list(const char *path, std::vector<std::string> *indexes);

class SourcePackStream
{
public:
    explicit SourcePackStream(const std::string &path);
    int open();
    bool next(SourcePackRecord *record);

private:
    std::string value(const std::vector<std::string> &row, const char *name) const;

    std::string path_;
    std::ifstream in_;
    std::map<std::string, int> fields_;
};

class SourcePackGroupReader
{
public:
    explicit SourcePackGroupReader(const std::vector<std::string> &paths);
    ~SourcePackGroupReader();

    int open();
    bool next_group(std::vector<SourcePackRecord> *records);

private:
    struct HeapItem
    {
        SourcePackRecord record;
        std::size_t stream_index;
        std::size_t sequence;
    };

    struct HeapCompare
    {
        bool operator()(const HeapItem &a, const HeapItem &b) const;
    };

    static bool timestamp_less(const SourcePackRecord &a, const SourcePackRecord &b);
    void push_record(const SourcePackRecord &record, std::size_t stream_index);

    std::vector<std::string> paths_;
    std::vector<SourcePackStream *> streams_;
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapCompare> heap_;
    std::size_t sequence_;
};

class PackFileCache
{
public:
    PackFileCache();
    ~PackFileCache();

    FILE *get(const std::string &path);
    void close_all();

private:
    std::map<std::string, FILE *> files_;
};

void cleanup_tfpws_sourcepack_item(TfpwsSourcePackItem *item);
unsigned tfpws_sac_data_count(const SACHEAD *hd);
int prepare_tfpws_sourcepack_item(const std::vector<SourcePackRecord> &records,
                                  int sub_stack_size,
                                  PackFileCache *cache,
                                  TfpwsSourcePackItem *item);

class TfpwsSourcePackShardWriter
{
public:
    TfpwsSourcePackShardWriter();
    ~TfpwsSourcePackShardWriter();

    int open(const std::string &output_dir, std::size_t worker_index);
    int append(const SourcePackRecord &record,
               const SACHEAD &header,
               const float *data);
    int close();

private:
    std::string final_pair_path(const SourcePackRecord &record) const;

    std::string output_dir_;
    std::string pack_path_;
    std::string index_path_;
    FILE *pack_;
    std::ofstream index_;
    std::size_t record_count_;
};

int merge_tfpws_sourcepack_shard_indexes(const std::string &output_dir,
                                         std::size_t worker_count,
                                         const std::string &final_index_path);

#endif
