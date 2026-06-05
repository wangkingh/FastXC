#include "stepack_writer.h"

#include "logger.h"
#include "path_utils.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t STEPACK_VERSION = 3;
static const uint32_t STEPACK_LAYOUT_PITCHED_STEP_NSLC_FREQ = 2;
static const char STEPACK_TSV_HEADER[] =
    "timestamp\tworker_id\tbatch_seq\tstart_group\tgroup_count\t"
    "pack_path\toffset\tbytes\tversion\tlayout\tnstep\tnslc_start\t"
    "nslc_count\tbatch_nslc_count\tnspec\tdt\tdf\tnslc_table_bytes\t"
    "payload_offset\tpayload_bytes\tbatch_payload_bytes\tstep_bytes\t"
    "pitch_step_bytes\tnslc_step_bytes\tfirst_nsl_id\tlast_nsl_id\n";

#pragma pack(push, 1)
typedef struct StepackBatchHeader
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
} StepackBatchHeader;

typedef struct StepackNslcEntry
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
} StepackNslcEntry;
#pragma pack(pop)

struct StepackWriter
{
    char *root;
    int worker_id;
};

static void copyFixedString(char *dst, size_t dst_size, const char *text)
{
    size_t i = 0;
    if (dst_size == 0)
    {
        return;
    }
    memset(dst, 0, dst_size);
    if (text == NULL)
    {
        return;
    }
    for (; i + 1 < dst_size && text[i] != '\0'; i++)
    {
        char ch = text[i];
        dst[i] = (ch == '\t' || ch == '\n' || ch == '\r') ? ' ' : ch;
    }
}

static void writeTsvField(FILE *fp, const char *text)
{
    const char *p = text == NULL ? "" : text;
    while (*p != '\0')
    {
        char ch = *p++;
        fputc((ch == '\t' || ch == '\n' || ch == '\r') ? ' ' : ch, fp);
    }
}

static int checkedMulU64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a != 0 && b > UINT64_MAX / a)
    {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int checkedAddU64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (b > UINT64_MAX - a)
    {
        return 0;
    }
    *out = a + b;
    return 1;
}

static char *buildBatchPath(const char *root, int worker_id,
                            size_t batch_seq, const char *suffix)
{
    char leaf[MAXPATH];
    int needed = snprintf(leaf, sizeof(leaf), "w%03d.b%06zu.%s",
                          worker_id, batch_seq, suffix);
    if (needed < 0 || (size_t)needed >= sizeof(leaf))
    {
        LOG_ERROR("stepack_path_too_long",
                  "worker_id=%d batch_seq=%zu suffix=%s",
                  worker_id, batch_seq, suffix);
        return NULL;
    }
    return PathJoinAlloc(root, leaf);
}

static int validateNodeForStepack(const InOutNode *node)
{
    if (node == NULL || node->meta == NULL || node->sac_hd == NULL ||
        node->spectrum == NULL || node->nstep <= 0 || node->nspec <= 0)
    {
        LOG_ERROR("stepack_append_invalid_input",
                  "node=%p meta=%p sac_hd=%p spectrum=%p nstep=%d nspec=%d",
                  (void *)node,
                  node == NULL ? NULL : (void *)node->meta,
                  node == NULL ? NULL : (void *)node->sac_hd,
                  node == NULL ? NULL : (void *)node->spectrum,
                  node == NULL ? 0 : node->nstep,
                  node == NULL ? 0 : node->nspec);
        return -1;
    }
    if (node->meta->timestamp[0] == '\0')
    {
        LOG_ERROR("stepack_timestamp_empty", "nsl_id=\"%s\"",
                  node->meta->nsl_id);
        return -1;
    }
    return 0;
}

static int validateBatchShape(const InOutNode *nodes, size_t node_count)
{
    int nstep = nodes[0].nstep;
    int nspec = nodes[0].nspec;
    float dt = nodes[0].dt;
    float df = nodes[0].df;

    for (size_t i = 0; i < node_count; i++)
    {
        if (validateNodeForStepack(&nodes[i]) != 0)
        {
            return -1;
        }
        if (nodes[i].nstep != nstep || nodes[i].nspec != nspec ||
            nodes[i].dt != dt || nodes[i].df != df)
        {
            LOG_ERROR("stepack_batch_shape_mismatch",
                      "index=%zu nstep=%d/%d nspec=%d/%d dt=%.9g/%.9g df=%.9g/%.9g",
                      i, nstep, nodes[i].nstep, nspec, nodes[i].nspec,
                      dt, nodes[i].dt, df, nodes[i].df);
            return -1;
        }
    }
    return 0;
}

static void fillNslcEntry(StepackNslcEntry *entry,
                          const InOutNode *node,
                          size_t file_index)
{
    memset(entry, 0, sizeof(*entry));
    entry->file_index = (uint32_t)file_index;
    entry->nsl_id = (uint32_t)node->meta->gnsl_id;
    entry->stla = node->sac_hd->stla;
    entry->stlo = node->sac_hd->stlo;
    copyFixedString(entry->network, sizeof(entry->network), node->meta->network);
    copyFixedString(entry->station, sizeof(entry->station), node->meta->station);
    copyFixedString(entry->location, sizeof(entry->location), node->meta->location);
    copyFixedString(entry->component, sizeof(entry->component), node->meta->component);
}

static StepackNslcEntry *buildNslcTable(const InOutNode *nodes,
                                        size_t node_count,
                                        uint64_t nslc_table_bytes)
{
    StepackNslcEntry *entries;
    if (nslc_table_bytes > (uint64_t)SIZE_MAX)
    {
        LOG_ERROR("stepack_nslc_table_too_large",
                  "nslc_count=%zu bytes=%" PRIu64,
                  node_count, nslc_table_bytes);
        return NULL;
    }
    entries = (StepackNslcEntry *)malloc((size_t)nslc_table_bytes);
    if (entries == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=stepack_nslc_table bytes=%" PRIu64,
                  nslc_table_bytes);
        return NULL;
    }
    for (size_t i = 0; i < node_count; i++)
    {
        fillNslcEntry(&entries[i], &nodes[i], i);
    }
    return entries;
}

static int computeBatchSizes(const InOutNode *nodes,
                             size_t node_count,
                             uint64_t *nslc_table_bytes,
                             uint64_t *nslc_step_bytes,
                             uint64_t *pitch_step_bytes,
                             uint64_t *payload_bytes,
                             uint64_t *record_bytes)
{
    uint64_t tmp;
    if (!checkedMulU64((uint64_t)node_count,
                       (uint64_t)sizeof(StepackNslcEntry),
                       nslc_table_bytes) ||
        !checkedMulU64((uint64_t)nodes[0].nspec,
                       (uint64_t)sizeof(complex),
                       nslc_step_bytes) ||
        !checkedMulU64((uint64_t)node_count,
                       *nslc_step_bytes,
                       pitch_step_bytes) ||
        !checkedMulU64((uint64_t)nodes[0].nstep,
                       *pitch_step_bytes,
                       payload_bytes) ||
        !checkedAddU64((uint64_t)sizeof(StepackBatchHeader),
                       *nslc_table_bytes,
                       &tmp) ||
        !checkedAddU64(tmp, *payload_bytes, record_bytes))
    {
        LOG_ERROR("stepack_size_overflow",
                  "nslc_count=%zu nstep=%d nspec=%d",
                  node_count, nodes[0].nstep, nodes[0].nspec);
        return -1;
    }
    return 0;
}

static int fillBatchHeader(StepackBatchHeader *header,
                           const InOutNode *nodes,
                           size_t node_count,
                           size_t batch_seq,
                           size_t start_group,
                           size_t group_count,
                           int worker_id,
                           uint64_t nslc_table_bytes,
                           uint64_t payload_offset,
                           uint64_t payload_bytes,
                           uint64_t pitch_step_bytes)
{
    if (node_count > UINT32_MAX ||
        (uint64_t)nodes[0].nstep > UINT32_MAX ||
        (uint64_t)nodes[0].nspec > UINT32_MAX)
    {
        LOG_ERROR("stepack_batch_too_large",
                  "node_count=%zu nstep=%d nspec=%d",
                  node_count, nodes[0].nstep, nodes[0].nspec);
        return -1;
    }

    memset(header, 0, sizeof(*header));
    memcpy(header->magic, "FXCSTPK", 7);
    header->version = STEPACK_VERSION;
    header->header_size = (uint32_t)sizeof(*header);
    header->nslc_entry_size = (uint32_t)sizeof(StepackNslcEntry);
    header->layout = STEPACK_LAYOUT_PITCHED_STEP_NSLC_FREQ;
    header->batch_seq = (uint64_t)batch_seq;
    header->start_group = (uint64_t)start_group;
    header->group_count = (uint64_t)group_count;
    header->worker_id = (uint32_t)worker_id;
    header->nstep = (uint32_t)nodes[0].nstep;
    header->nslc_count = (uint32_t)node_count;
    header->nspec = (uint32_t)nodes[0].nspec;
    header->dt = nodes[0].dt;
    header->df = nodes[0].df;
    header->nslc_table_bytes = nslc_table_bytes;
    header->payload_offset = payload_offset;
    header->payload_bytes = payload_bytes;
    header->pitch_step_bytes = pitch_step_bytes;
    copyFixedString(header->first_timestamp, sizeof(header->first_timestamp),
                    nodes[0].meta->timestamp);
    return 0;
}

static int writeTimestampRunRows(FILE *tsv_fp,
                                 const char *pack_path,
                                 int worker_id,
                                 size_t batch_seq,
                                 size_t start_group,
                                 size_t group_count,
                                 const InOutNode *nodes,
                                 size_t node_count,
                                 size_t num_ch,
                                 uint64_t record_offset,
                                 uint64_t record_bytes,
                                 uint64_t nslc_table_bytes,
                                 uint64_t payload_offset,
                                 uint64_t batch_payload_bytes,
                                 uint64_t nslc_step_bytes,
                                 uint64_t pitch_step_bytes)
{
    size_t begin = 0;
    while (begin < node_count)
    {
        const char *timestamp = nodes[begin].meta != NULL ? nodes[begin].meta->timestamp : "";
        size_t end = begin + 1;
        while (end < node_count &&
               nodes[end].meta != NULL &&
               strcmp(nodes[end].meta->timestamp, timestamp) == 0)
        {
            end++;
        }

        size_t run_nslc_count = end - begin;
        size_t run_start_group = start_group;
        size_t run_group_count = group_count;
        uint64_t run_step_bytes = 0;
        uint64_t run_payload_bytes = 0;

        if (num_ch > 0 && begin % num_ch == 0 && run_nslc_count % num_ch == 0)
        {
            run_start_group = start_group + begin / num_ch;
            run_group_count = run_nslc_count / num_ch;
        }
        if (!checkedMulU64((uint64_t)run_nslc_count,
                           nslc_step_bytes,
                           &run_step_bytes) ||
            !checkedMulU64((uint64_t)nodes[0].nstep,
                           run_step_bytes,
                           &run_payload_bytes))
        {
            LOG_ERROR("stepack_run_size_overflow",
                      "nslc_start=%zu nslc_count=%zu",
                      begin, run_nslc_count);
            return -1;
        }

        writeTsvField(tsv_fp, timestamp);
        fprintf(tsv_fp,
                "\t%d\t%zu\t%zu\t%zu\t",
                worker_id, batch_seq, run_start_group, run_group_count);
        writeTsvField(tsv_fp, pack_path);
        fprintf(tsv_fp,
                "\t%" PRIu64 "\t%" PRIu64 "\t%u\t%u\t%d\t%zu\t%zu\t%zu\t%d\t%.9g\t%.9g\t"
                "%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t"
                "%" PRIu64 "\t%" PRIu64 "\t",
                record_offset, record_bytes,
                STEPACK_VERSION, STEPACK_LAYOUT_PITCHED_STEP_NSLC_FREQ,
                nodes[0].nstep, begin, run_nslc_count, node_count,
                nodes[0].nspec, nodes[0].dt, nodes[0].df,
                nslc_table_bytes, payload_offset, run_payload_bytes,
                batch_payload_bytes, run_step_bytes, pitch_step_bytes,
                nslc_step_bytes);
        writeTsvField(tsv_fp, nodes[begin].meta->nsl_id);
        fputc('\t', tsv_fp);
        writeTsvField(tsv_fp, nodes[end - 1].meta->nsl_id);
        fputc('\n', tsv_fp);

        begin = end;
    }
    return 0;
}

StepackWriter *CreateStepackWriter(const char *stepack_root, int worker_id)
{
    StepackWriter *writer = (StepackWriter *)calloc(1, sizeof(*writer));
    if (writer == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=stepack_writer");
        return NULL;
    }

    writer->root = PathAbsoluteDup(stepack_root);
    if (writer->root == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=stepack_root");
        free(writer);
        return NULL;
    }
    if (PathMakeDirectoryRecursive(writer->root) != 0)
    {
        free(writer->root);
        free(writer);
        return NULL;
    }
    writer->worker_id = worker_id;
    return writer;
}

int StepackWriterAppendBatch(StepackWriter *writer, const InOutNode *nodes,
                           size_t node_count, size_t batch_seq,
                           size_t start_group, size_t group_count)
{
    char *pack_path = NULL;
    char *tsv_path = NULL;
    FILE *pack_fp = NULL;
    FILE *tsv_fp = NULL;
    StepackBatchHeader header;
    StepackNslcEntry *nslc_table = NULL;
    uint64_t nslc_table_bytes = 0;
    uint64_t nslc_step_bytes = 0;
    uint64_t pitch_step_bytes = 0;
    uint64_t payload_bytes = 0;
    uint64_t record_bytes = 0;
    uint64_t payload_offset = 0;
    size_t num_ch = 0;
    int status = -1;

    if (writer == NULL || nodes == NULL)
    {
        LOG_ERROR("stepack_append_batch_invalid_input",
                  "writer=%p nodes=%p node_count=%zu",
                  (void *)writer, (void *)nodes, node_count);
        return -1;
    }
    if (node_count == 0)
    {
        return 0;
    }
    if (validateBatchShape(nodes, node_count) != 0 ||
        computeBatchSizes(nodes, node_count,
                          &nslc_table_bytes,
                          &nslc_step_bytes,
                          &pitch_step_bytes,
                          &payload_bytes,
                          &record_bytes) != 0)
    {
        return -1;
    }
    if (group_count > 0 && node_count % group_count == 0)
    {
        num_ch = node_count / group_count;
    }

    payload_offset = (uint64_t)sizeof(StepackBatchHeader) + nslc_table_bytes;
    if (fillBatchHeader(&header, nodes, node_count, batch_seq,
                        start_group, group_count, writer->worker_id,
                        nslc_table_bytes, payload_offset,
                        payload_bytes, pitch_step_bytes) != 0)
    {
        return -1;
    }

    pack_path = buildBatchPath(writer->root, writer->worker_id, batch_seq, "stepack");
    tsv_path = buildBatchPath(writer->root, writer->worker_id, batch_seq, "tsv");
    if (pack_path == NULL || tsv_path == NULL)
    {
        goto done;
    }
    pack_fp = fopen(pack_path, "wb");
    tsv_fp = fopen(tsv_path, "w");
    if (pack_fp == NULL || tsv_fp == NULL)
    {
        LOG_ERROR("open_stepack_failed",
                  "pack=\"%s\" tsv=\"%s\" errno=%d",
                  pack_path, tsv_path, errno);
        goto done;
    }

    nslc_table = buildNslcTable(nodes, node_count, nslc_table_bytes);
    if (nslc_table == NULL)
    {
        goto done;
    }

    if (fwrite(&header, sizeof(header), 1, pack_fp) != 1 ||
        fwrite(nslc_table, (size_t)nslc_table_bytes, 1, pack_fp) != 1 ||
        fwrite(nodes[0].spectrum, (size_t)payload_bytes, 1, pack_fp) != 1)
    {
        LOG_ERROR("stepack_batch_write_failed",
                  "path=\"%s\" bytes=%" PRIu64 " errno=%d",
                  pack_path, record_bytes, errno);
        goto done;
    }

    fputs(STEPACK_TSV_HEADER, tsv_fp);
    if (writeTimestampRunRows(tsv_fp, pack_path, writer->worker_id,
                              batch_seq, start_group, group_count,
                              nodes, node_count, num_ch,
                              0, record_bytes,
                              nslc_table_bytes, payload_offset,
                              payload_bytes, nslc_step_bytes,
                              pitch_step_bytes) != 0)
    {
        goto done;
    }

    status = 0;

done:
    free(nslc_table);
    if (pack_fp != NULL && fclose(pack_fp) != 0)
    {
        LOG_ERROR("close_stepack_failed", "path=\"%s\" errno=%d",
                  pack_path != NULL ? pack_path : "", errno);
        status = -1;
    }
    if (tsv_fp != NULL && fclose(tsv_fp) != 0)
    {
        LOG_ERROR("close_stepack_tsv_failed", "path=\"%s\" errno=%d",
                  tsv_path != NULL ? tsv_path : "", errno);
        status = -1;
    }
    free(pack_path);
    free(tsv_path);
    return status;
}

void DestroyStepackWriter(StepackWriter *writer)
{
    if (writer == NULL)
    {
        return;
    }
    free(writer->root);
    free(writer);
}
