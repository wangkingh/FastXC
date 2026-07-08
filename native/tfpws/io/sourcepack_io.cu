#include "sourcepack_io.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <unistd.h>

#include "logger.h"

extern "C"
{
#include "path_util.h"
}

static const std::size_t TFPWS_SOURCEPACK_FILE_CACHE_LIMIT = 256;

static std::vector<std::string> split_tab(const std::string &line)
{
    std::vector<std::string> out;
    std::size_t start = 0;
    while (true)
    {
        std::size_t pos = line.find('\t', start);
        if (pos == std::string::npos)
        {
            out.push_back(line.substr(start));
            break;
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

std::string sourcepack_trim_copy(const std::string &s)
{
    std::size_t first = 0;
    while (first < s.size() && (s[first] == ' ' || s[first] == '\t' || s[first] == '\r' || s[first] == '\n'))
        ++first;
    std::size_t last = s.size();
    while (last > first && (s[last - 1] == ' ' || s[last - 1] == '\t' || s[last - 1] == '\r' || s[last - 1] == '\n'))
        --last;
    return s.substr(first, last - first);
}

std::string tfpws_join_path(const std::string &dir, const char *name)
{
    if (dir.empty() || dir[dir.size() - 1] == '/')
        return dir + name;
    return dir + "/" + name;
}

std::string tfpws_absolute_path(const std::string &path)
{
    if (path.empty() || path[0] == '/')
        return path;

    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof(cwd)))
        return path;
    return tfpws_join_path(cwd, path.c_str());
}

const char *tfpws_sourcepack_index_header()
{
    return "timestamp\tpath_id\tcomponent_slot\tsource_key\treceiver_key\tsrc_id\trec_id\t"
           "src_network\tsrc_station\tsrc_location\tsrc_component\t"
           "rec_network\trec_station\trec_location\trec_component\t"
           "npts\tdt\tdist\taz\tbaz\trecord_path\trecord_offset\tbytes\tstorage_kind\tfinal_pair_path";
}

static int require_field(const std::map<std::string, int> &fields,
                         const char *name,
                         const char *path)
{
    std::map<std::string, int>::const_iterator it = fields.find(name);
    if (it == fields.end())
    {
        LOG_ERROR("sourcepack_index_missing_field",
                  "index=\"%s\" field=\"%s\"",
                  path,
                  name);
        return -1;
    }
    return it->second;
}

int read_sourcepack_list(const char *path, std::vector<std::string> *indexes)
{
    std::ifstream in(path);
    if (!in)
    {
        LOG_ERROR("open_sourcepack_list_failed",
                  "path=\"%s\"",
                  path);
        return 1;
    }

    std::string line;
    while (std::getline(in, line))
    {
        line = sourcepack_trim_copy(line);
        if (line.empty() || line[0] == '#')
            continue;
        indexes->push_back(line);
    }
    if (indexes->empty())
    {
        LOG_ERROR("sourcepack_list_empty",
                  "path=\"%s\"",
                  path);
        return 1;
    }
    return 0;
}

SourcePackStream::SourcePackStream(const std::string &path) : path_(path)
{
}

int SourcePackStream::open()
{
    in_.open(path_.c_str());
    if (!in_)
    {
        LOG_ERROR("open_sourcepack_index_failed",
                  "path=\"%s\"",
                  path_.c_str());
        return 1;
    }

    std::string header;
    if (!std::getline(in_, header))
    {
        LOG_ERROR("sourcepack_index_empty",
                  "path=\"%s\"",
                  path_.c_str());
        return 1;
    }
    std::vector<std::string> names = split_tab(header);
    for (std::size_t i = 0; i < names.size(); ++i)
        fields_[sourcepack_trim_copy(names[i])] = (int)i;

    const char *required[] = {
        "timestamp", "path_id", "component_slot",
        "source_key", "receiver_key", "src_id", "rec_id",
        "src_network", "src_station", "src_location", "src_component",
        "rec_network", "rec_station", "rec_location", "rec_component",
        "npts", "dt", "dist", "az", "baz",
        "record_path", "record_offset", "bytes", "final_pair_path"};
    for (std::size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i)
        if (require_field(fields_, required[i], path_.c_str()) < 0)
            return 1;
    return 0;
}

bool SourcePackStream::next(SourcePackRecord *record)
{
    std::string line;
    while (std::getline(in_, line))
    {
        if (line.empty())
            continue;
        std::vector<std::string> row = split_tab(line);
        if (row.size() < fields_.size())
            row.resize(fields_.size());

        record->timestamp = value(row, "timestamp");
        record->path_id = value(row, "path_id");
        record->component_slot = std::atoi(value(row, "component_slot").c_str());
        record->source_key = value(row, "source_key");
        record->receiver_key = value(row, "receiver_key");
        record->src_id = value(row, "src_id");
        record->rec_id = value(row, "rec_id");
        record->src_network = value(row, "src_network");
        record->src_station = value(row, "src_station");
        record->src_location = value(row, "src_location");
        record->src_component = value(row, "src_component");
        record->rec_network = value(row, "rec_network");
        record->rec_station = value(row, "rec_station");
        record->rec_location = value(row, "rec_location");
        record->rec_component = value(row, "rec_component");
        record->dist = value(row, "dist");
        record->az = value(row, "az");
        record->baz = value(row, "baz");
        record->final_pair_path = value(row, "final_pair_path");
        record->record_path = value(row, "record_path");
        record->record_offset = std::strtoll(value(row, "record_offset").c_str(), NULL, 10);
        record->record_bytes = std::strtoll(value(row, "bytes").c_str(), NULL, 10);
        return true;
    }
    return false;
}

std::string SourcePackStream::value(const std::vector<std::string> &row, const char *name) const
{
    std::map<std::string, int>::const_iterator it = fields_.find(name);
    if (it == fields_.end() || it->second < 0 || (std::size_t)it->second >= row.size())
        return "";
    return sourcepack_trim_copy(row[(std::size_t)it->second]);
}

bool SourcePackGroupReader::HeapCompare::operator()(const HeapItem &a, const HeapItem &b) const
{
    if (a.record.path_id != b.record.path_id)
        return a.record.path_id > b.record.path_id;
    if (a.record.component_slot != b.record.component_slot)
        return a.record.component_slot > b.record.component_slot;
    return a.sequence > b.sequence;
}

SourcePackGroupReader::SourcePackGroupReader(const std::vector<std::string> &paths)
    : paths_(paths), sequence_(0)
{
}

SourcePackGroupReader::~SourcePackGroupReader()
{
    for (std::size_t i = 0; i < streams_.size(); ++i)
        delete streams_[i];
}

int SourcePackGroupReader::open()
{
    streams_.reserve(paths_.size());
    for (std::size_t i = 0; i < paths_.size(); ++i)
    {
        SourcePackStream *stream = new SourcePackStream(paths_[i]);
        if (stream->open() != 0)
        {
            delete stream;
            return 1;
        }
        streams_.push_back(stream);
        SourcePackRecord record;
        if (stream->next(&record))
            push_record(record, i);
    }
    return 0;
}

bool SourcePackGroupReader::next_group(std::vector<SourcePackRecord> *records)
{
    records->clear();
    if (heap_.empty())
        return false;

    std::string path_id = heap_.top().record.path_id;
    int slot = heap_.top().record.component_slot;
    while (!heap_.empty() &&
           heap_.top().record.path_id == path_id &&
           heap_.top().record.component_slot == slot)
    {
        HeapItem item = heap_.top();
        heap_.pop();
        records->push_back(item.record);

        SourcePackRecord next;
        if (streams_[item.stream_index]->next(&next))
            push_record(next, item.stream_index);
    }
    std::sort(records->begin(), records->end(), timestamp_less);
    return true;
}

bool SourcePackGroupReader::timestamp_less(const SourcePackRecord &a, const SourcePackRecord &b)
{
    if (a.timestamp != b.timestamp)
        return a.timestamp < b.timestamp;
    if (a.record_path != b.record_path)
        return a.record_path < b.record_path;
    return a.record_offset < b.record_offset;
}

void SourcePackGroupReader::push_record(const SourcePackRecord &record, std::size_t stream_index)
{
    HeapItem item;
    item.record = record;
    item.stream_index = stream_index;
    item.sequence = sequence_++;
    heap_.push(item);
}

PackFileCache::PackFileCache()
{
}

PackFileCache::~PackFileCache()
{
    close_all();
}

FILE *PackFileCache::get(const std::string &path)
{
    std::map<std::string, FILE *>::iterator it = files_.find(path);
    if (it != files_.end())
        return it->second;

    if (files_.size() >= TFPWS_SOURCEPACK_FILE_CACHE_LIMIT)
        close_all();

    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp)
    {
        LOG_ERROR("open_sourcepack_record_failed",
                  "path=\"%s\" error=\"%s\"",
                  path.c_str(),
                  std::strerror(errno));
        return NULL;
    }
    files_[path] = fp;
    return fp;
}

void PackFileCache::close_all()
{
    for (std::map<std::string, FILE *>::iterator it = files_.begin(); it != files_.end(); ++it)
        std::fclose(it->second);
    files_.clear();
}

void cleanup_tfpws_sourcepack_item(TfpwsSourcePackItem *item)
{
    if (!item)
        return;
    std::free(item->prestack_data);
    std::free(item->linear_stack);
    std::free(item->group_trace_weights);
    item->prestack_data = NULL;
    item->linear_stack = NULL;
    item->group_trace_weights = NULL;
}

unsigned tfpws_sac_data_count(const SACHEAD *hd)
{
    return (hd->iftype == IXY) ? (unsigned)(hd->npts * 2) : (unsigned)hd->npts;
}

static int read_sourcepack_record(PackFileCache *cache,
                                  const SourcePackRecord &record,
                                  SACHEAD *header,
                                  float *data,
                                  unsigned expected_samples)
{
    FILE *fp = cache->get(record.record_path);
    if (!fp)
        return 1;
    if (fseeko(fp, (off_t)record.record_offset, SEEK_SET) != 0)
    {
        LOG_ERROR("seek_sourcepack_record_failed",
                  "record_path=\"%s\" record_offset=%lld",
                  record.record_path.c_str(),
                  record.record_offset);
        return 1;
    }
    if (std::fread(header, sizeof(SACHEAD), 1, fp) != 1)
    {
        LOG_ERROR("read_sourcepack_header_failed",
                  "record_path=\"%s\"",
                  record.record_path.c_str());
        return 1;
    }
#ifdef BYTE_SWAP
    swab4((char *)header, HD_SIZE);
#endif
    unsigned samples = tfpws_sac_data_count(header);
    if (samples != expected_samples)
    {
        LOG_ERROR("sourcepack_sample_count_mismatch",
                  "record_path=\"%s\" actual=%u expected=%u",
                  record.record_path.c_str(),
                  samples,
                  expected_samples);
        return 1;
    }
    long long expected_bytes = (long long)sizeof(SACHEAD) + (long long)samples * (long long)sizeof(float);
    if (record.record_bytes != expected_bytes)
    {
        LOG_ERROR("sourcepack_record_size_mismatch",
                  "record_path=\"%s\" actual_bytes=%lld expected_bytes=%lld",
                  record.record_path.c_str(),
                  record.record_bytes,
                  expected_bytes);
        return 1;
    }
    if (std::fread(data, sizeof(float), samples, fp) != samples)
    {
        LOG_ERROR("read_sourcepack_data_failed",
                  "record_path=\"%s\" samples=%u",
                  record.record_path.c_str(),
                  samples);
        return 1;
    }
#ifdef BYTE_SWAP
    swab4((char *)data, samples * sizeof(float));
#endif
    return 0;
}

int prepare_tfpws_sourcepack_item(const std::vector<SourcePackRecord> &records,
                                  int sub_stack_size,
                                  PackFileCache *cache,
                                  TfpwsSourcePackItem *item)
{
    if (records.empty())
        return 1;

    item->label = std::string("sourcepack:") + records[0].path_id + ":" +
                  records[0].src_component + "-" + records[0].rec_component;
    item->record = records[0];
    item->num_segments = 0;
    item->nsamples = 0;
    item->ngroups = 0;
    item->prestack_data = NULL;
    item->linear_stack = NULL;
    item->group_trace_weights = NULL;

    SACHEAD first_hd;
    FILE *fp = cache->get(records[0].record_path);
    if (!fp)
        return 1;
    if (fseeko(fp, (off_t)records[0].record_offset, SEEK_SET) != 0 ||
        std::fread(&first_hd, sizeof(SACHEAD), 1, fp) != 1)
    {
        LOG_ERROR("read_first_sourcepack_header_failed",
                  "record_path=\"%s\" record_offset=%lld",
                  records[0].record_path.c_str(),
                  records[0].record_offset);
        return 1;
    }
#ifdef BYTE_SWAP
    swab4((char *)&first_hd, HD_SIZE);
#endif

    unsigned nsamples = tfpws_sac_data_count(&first_hd);
    std::size_t group_sz = (sub_stack_size < 2) ? 1 : (std::size_t)sub_stack_size;
    unsigned num_segments = (unsigned)records.size();
    unsigned ngroups = (unsigned)((records.size() + group_sz - 1) / group_sz);
    std::size_t prestack_count = (std::size_t)ngroups * nsamples;

    item->header = first_hd;
    item->num_segments = num_segments;
    item->nsamples = nsamples;
    item->ngroups = ngroups;
    item->prestack_data = (float *)std::calloc(prestack_count, sizeof(float));
    item->linear_stack = (float *)std::calloc(nsamples, sizeof(float));
    item->group_trace_weights = (float *)std::malloc((std::size_t)ngroups * sizeof(float));
    if (!item->prestack_data || !item->linear_stack || !item->group_trace_weights)
    {
        LOG_ERROR("sourcepack_host_allocation_failed",
                  "path_id=\"%s\" component_slot=%d groups=%u samples=%u",
                  records[0].path_id.c_str(),
                  records[0].component_slot,
                  ngroups,
                  nsamples);
        cleanup_tfpws_sourcepack_item(item);
        return 1;
    }

    std::vector<float> buffer(nsamples);
    const float inv_total = 1.0f / (float)num_segments;
    for (unsigned g = 0; g < ngroups; ++g)
    {
        std::size_t first = (std::size_t)g * group_sz;
        std::size_t last = std::min(first + group_sz, records.size());
        std::size_t group_count = last - first;
        float *trace_sum = item->prestack_data + (std::size_t)g * nsamples;

        for (std::size_t i = first; i < last; ++i)
        {
            SACHEAD hd;
            if (read_sourcepack_record(cache, records[i], &hd, buffer.data(), nsamples) != 0)
            {
                cleanup_tfpws_sourcepack_item(item);
                return 1;
            }
            for (unsigned j = 0; j < nsamples; ++j)
                trace_sum[j] += buffer[j];
        }

        item->group_trace_weights[g] = (float)group_count * inv_total;
        for (unsigned j = 0; j < nsamples; ++j)
            item->linear_stack[j] += trace_sum[j] * inv_total;
    }

    return 0;
}

TfpwsSourcePackShardWriter::TfpwsSourcePackShardWriter()
    : pack_(NULL), record_count_(0)
{
}

TfpwsSourcePackShardWriter::~TfpwsSourcePackShardWriter()
{
    close();
}

int TfpwsSourcePackShardWriter::open(const std::string &output_dir, std::size_t worker_index)
{
    output_dir_ = output_dir;
    char pack_name[64];
    char index_name[64];
    std::snprintf(pack_name, sizeof(pack_name), "tfpws.w%03zu.pack", worker_index);
    std::snprintf(index_name, sizeof(index_name), "tfpws.w%03zu.tsv", worker_index);
    pack_path_ = tfpws_join_path(output_dir_, pack_name);
    index_path_ = tfpws_join_path(output_dir_, index_name);

    if (ensure_parent_dir(pack_path_.c_str()) != 0)
    {
        LOG_ERROR("create_output_directory_failed",
                  "path=\"%s\" error=\"%s\"",
                  pack_path_.c_str(),
                  std::strerror(errno));
        return 1;
    }

    pack_ = std::fopen(pack_path_.c_str(), "wb");
    if (!pack_)
    {
        LOG_ERROR("create_tfpws_pack_failed",
                  "path=\"%s\" error=\"%s\"",
                  pack_path_.c_str(),
                  std::strerror(errno));
        return 1;
    }
    index_.open(index_path_.c_str());
    if (!index_)
    {
        LOG_ERROR("create_tfpws_shard_index_failed",
                  "path=\"%s\"",
                  index_path_.c_str());
        close();
        return 1;
    }
    index_ << tfpws_sourcepack_index_header() << '\n';
    record_count_ = 0;
    return 0;
}

int TfpwsSourcePackShardWriter::append(const SourcePackRecord &record,
                                       const SACHEAD &header,
                                       const float *data)
{
    if (!pack_ || !index_)
        return 1;

    long long offset = (long long)ftello(pack_);
    if (offset < 0)
    {
        LOG_ERROR("tfpws_pack_ftello_failed",
                  "path=\"%s\" error=\"%s\"",
                  pack_path_.c_str(),
                  std::strerror(errno));
        return 1;
    }

    SACHEAD out_header = header;
    std::vector<float> out_data(data, data + tfpws_sac_data_count(&header));
#ifdef BYTE_SWAP
    swab4((char *)&out_header, HD_SIZE);
    swab4((char *)out_data.data(), out_data.size() * sizeof(float));
#endif
    if (std::fwrite(&out_header, sizeof(SACHEAD), 1, pack_) != 1 ||
        std::fwrite(out_data.data(), sizeof(float), out_data.size(), pack_) != out_data.size())
    {
        LOG_ERROR("write_tfpws_pack_failed",
                  "path=\"%s\" samples=%zu",
                  pack_path_.c_str(),
                  out_data.size());
        return 1;
    }

    long long bytes = (long long)sizeof(SACHEAD) + (long long)out_data.size() * (long long)sizeof(float);
    index_ << "STACK" << '\t'
           << record.path_id << '\t'
           << record.component_slot << '\t'
           << record.source_key << '\t'
           << record.receiver_key << '\t'
           << record.src_id << '\t'
           << record.rec_id << '\t'
           << record.src_network << '\t'
           << record.src_station << '\t'
           << record.src_location << '\t'
           << record.src_component << '\t'
           << record.rec_network << '\t'
           << record.rec_station << '\t'
           << record.rec_location << '\t'
           << record.rec_component << '\t'
           << header.npts << '\t'
           << header.delta << '\t'
           << record.dist << '\t'
           << record.az << '\t'
           << record.baz << '\t'
           << pack_path_ << '\t'
           << offset << '\t'
           << bytes << '\t'
           << "tfpws_pack" << '\t'
           << final_pair_path(record) << '\n';
    if (!index_)
        return 1;
    ++record_count_;
    return 0;
}

int TfpwsSourcePackShardWriter::close()
{
    int rc = 0;
    if (pack_)
    {
        if (std::fclose(pack_) != 0)
            rc = 1;
        pack_ = NULL;
    }
    if (index_.is_open())
        index_.close();
    return rc;
}

std::string TfpwsSourcePackShardWriter::final_pair_path(const SourcePackRecord &record) const
{
    std::string path = record.final_pair_path;
    if (path.empty())
    {
        path = record.src_network + "-" + record.rec_network + "." +
               record.src_station + "-" + record.rec_station + "." +
               record.src_component + "-" + record.rec_component;
    }
    const std::string sac_suffix = ".sac";
    std::size_t pos = path.rfind(sac_suffix);
    if (pos != std::string::npos && pos + sac_suffix.size() == path.size())
        path.replace(pos, sac_suffix.size(), ".tfpws.sac");
    else
        path += ".tfpws.sac";
    return path;
}

struct MergeRow
{
    std::string path_id;
    int component_slot;
    std::size_t worker_index;
    std::size_t sequence;
    std::string line;
};

static bool merge_row_less(const MergeRow &a, const MergeRow &b)
{
    if (a.path_id != b.path_id)
        return a.path_id < b.path_id;
    if (a.component_slot != b.component_slot)
        return a.component_slot < b.component_slot;
    if (a.worker_index != b.worker_index)
        return a.worker_index < b.worker_index;
    return a.sequence < b.sequence;
}

int merge_tfpws_sourcepack_shard_indexes(const std::string &output_dir,
                                         std::size_t worker_count,
                                         const std::string &final_index_path)
{
    std::vector<MergeRow> rows;
    for (std::size_t worker = 0; worker < worker_count; ++worker)
    {
        char index_name[64];
        std::snprintf(index_name, sizeof(index_name), "tfpws.w%03zu.tsv", worker);
        std::string shard_path = tfpws_join_path(output_dir, index_name);
        std::ifstream in(shard_path.c_str());
        if (!in)
        {
            LOG_ERROR("open_tfpws_shard_index_failed",
                      "path=\"%s\"",
                      shard_path.c_str());
            return 1;
        }

        std::string line;
        if (!std::getline(in, line))
            continue;
        std::size_t seq = 0;
        while (std::getline(in, line))
        {
            line = sourcepack_trim_copy(line);
            if (line.empty())
                continue;
            std::vector<std::string> fields = split_tab(line);
            if (fields.size() < 3)
            {
                LOG_ERROR("malformed_tfpws_shard_index_row",
                          "path=\"%s\"",
                          shard_path.c_str());
                return 1;
            }
            MergeRow row;
            row.path_id = sourcepack_trim_copy(fields[1]);
            row.component_slot = std::atoi(sourcepack_trim_copy(fields[2]).c_str());
            row.worker_index = worker;
            row.sequence = seq++;
            row.line = line;
            rows.push_back(row);
        }
    }

    std::sort(rows.begin(), rows.end(), merge_row_less);

    std::ofstream out(final_index_path.c_str());
    if (!out)
    {
        LOG_ERROR("create_sourcepack_index_failed",
                  "path=\"%s\"",
                  final_index_path.c_str());
        return 1;
    }
    out << tfpws_sourcepack_index_header() << '\n';
    for (std::size_t i = 0; i < rows.size(); ++i)
        out << rows[i].line << '\n';
    return out ? 0 : 1;
}
