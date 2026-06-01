#include "spack_writer.h"

#include "logger.h"
#include "path_utils.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t SPACK_TARGET_BYTES = 4ULL * 1024ULL * 1024ULL * 1024ULL;
static const char SPACK_TSV_HEADER[] =
    "timestamp\tworker_id\tpack_path\toffset\tbytes\t"
    "nsl_id\tnetwork\tstation\tlocation\tcomponent\t"
    "nstep\tnspec\tdt\tdf\tstla\tstlo\n";

struct SpackWriter
{
    char *root;
    char *timestamp_dir;
    char *pack_path;
    char *tsv_path;
    FILE *pack_fp;
    FILE *tsv_fp;
    char current_timestamp[MAXNAME];
    uint64_t current_bytes;
    int worker_id;
    int part_id;
    int has_timestamp;
};

static int closeOpenFiles(SpackWriter *writer)
{
    int status = 0;
    if (writer->pack_fp != NULL)
    {
        if (fclose(writer->pack_fp) != 0)
        {
            LOG_ERROR("close_spack_failed", "path=\"%s\" errno=%d",
                      writer->pack_path != NULL ? writer->pack_path : "",
                      errno);
            status = -1;
        }
        writer->pack_fp = NULL;
    }
    if (writer->tsv_fp != NULL)
    {
        if (fclose(writer->tsv_fp) != 0)
        {
            LOG_ERROR("close_spack_tsv_failed", "path=\"%s\" errno=%d",
                      writer->tsv_path != NULL ? writer->tsv_path : "",
                      errno);
            status = -1;
        }
        writer->tsv_fp = NULL;
    }
    free(writer->pack_path);
    free(writer->tsv_path);
    writer->pack_path = NULL;
    writer->tsv_path = NULL;
    return status;
}

static int resetCurrentTimestamp(SpackWriter *writer)
{
    int status = closeOpenFiles(writer);
    free(writer->timestamp_dir);
    writer->timestamp_dir = NULL;
    writer->current_timestamp[0] = '\0';
    writer->current_bytes = 0;
    writer->part_id = 0;
    writer->has_timestamp = 0;
    return status;
}

static char *buildShardPath(const char *root, int worker_id,
                            int part_id, const char *suffix)
{
    char leaf[MAXPATH];
    int needed = snprintf(leaf, sizeof(leaf), "w%03d.p%06d.%s",
                          worker_id, part_id, suffix);
    if (needed < 0 || (size_t)needed >= sizeof(leaf))
    {
        LOG_ERROR("spack_path_too_long",
                  "worker_id=%d part_id=%d suffix=%s",
                  worker_id, part_id, suffix);
        return NULL;
    }
    return PathJoinAlloc(root, leaf);
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

static int openCurrentPart(SpackWriter *writer)
{
    const char *pack_mode = writer->current_bytes > 0 ? "ab" : "wb";
    const char *tsv_mode = writer->current_bytes > 0 ? "a" : "w";
    writer->pack_path = buildShardPath(writer->timestamp_dir, writer->worker_id,
                                       writer->part_id, "spack");
    writer->tsv_path = buildShardPath(writer->timestamp_dir, writer->worker_id,
                                      writer->part_id, "tsv");
    if (writer->pack_path == NULL || writer->tsv_path == NULL)
    {
        return -1;
    }

    writer->pack_fp = fopen(writer->pack_path, pack_mode);
    writer->tsv_fp = fopen(writer->tsv_path, tsv_mode);
    if (writer->pack_fp == NULL || writer->tsv_fp == NULL)
    {
        LOG_ERROR("open_spack_failed",
                  "pack=\"%s\" tsv=\"%s\" errno=%d",
                  writer->pack_path, writer->tsv_path, errno);
        (void)closeOpenFiles(writer);
        return -1;
    }

    if (writer->current_bytes == 0)
    {
        fputs(SPACK_TSV_HEADER, writer->tsv_fp);
    }
    return 0;
}

static int selectTimestamp(SpackWriter *writer, const char *timestamp)
{
    char leaf[MAXNAME];
    if (timestamp == NULL || timestamp[0] == '\0')
    {
        LOG_ERROR("spack_timestamp_empty", "worker_id=%d", writer->worker_id);
        return -1;
    }
    if (!PathTimestampLeafIsSafe(timestamp))
    {
        LOG_ERROR("spack_timestamp_unsafe_for_path",
                  "timestamp=%s worker_id=%d action=fix_sac_index_timestamp",
                  timestamp, writer->worker_id);
        return -1;
    }
    if (writer->has_timestamp &&
        strcmp(writer->current_timestamp, timestamp) == 0)
    {
        return 0;
    }

    if (resetCurrentTimestamp(writer) != 0)
    {
        return -1;
    }
    PathSafeTimestampLeaf(leaf, sizeof(leaf), timestamp);
    writer->timestamp_dir = PathJoinAlloc(writer->root, leaf);
    if (writer->timestamp_dir == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=spack_timestamp_dir");
        return -1;
    }
    if (PathMakeDirectoryRecursive(writer->timestamp_dir) != 0)
    {
        return -1;
    }
    snprintf(writer->current_timestamp, sizeof(writer->current_timestamp),
             "%s", timestamp);
    writer->has_timestamp = 1;
    writer->part_id = 0;
    writer->current_bytes = 0;
    return 0;
}

static int ensurePartForBytes(SpackWriter *writer, uint64_t record_bytes)
{
    if (writer->pack_fp != NULL &&
        writer->current_bytes > 0 &&
        writer->current_bytes + record_bytes > SPACK_TARGET_BYTES)
    {
        if (closeOpenFiles(writer) != 0)
        {
            return -1;
        }
        writer->part_id++;
        writer->current_bytes = 0;
    }
    else if (writer->pack_fp == NULL &&
             writer->current_bytes > 0 &&
             writer->current_bytes + record_bytes > SPACK_TARGET_BYTES)
    {
        writer->part_id++;
        writer->current_bytes = 0;
    }

    if (writer->pack_fp == NULL)
    {
        return openCurrentPart(writer);
    }
    return 0;
}

SpackWriter *CreateSpackWriter(const char *spack_root, int worker_id)
{
    SpackWriter *writer = (SpackWriter *)calloc(1, sizeof(*writer));
    if (writer == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=spack_writer");
        return NULL;
    }

    writer->root = PathAbsoluteDup(spack_root);
    if (writer->root == NULL)
    {
        LOG_ERROR("cpu_malloc_failed", "target=spack_root");
        free(writer);
        return NULL;
    }
    writer->worker_id = worker_id;
    writer->part_id = 0;
    return writer;
}

int SpackWriterAppend(SpackWriter *writer, const InOutNode *node)
{
    SEGSPEC segspec_hd;
    const SacIndexMeta *meta;
    uint64_t offset;
    uint64_t bytes;
    size_t payload_bytes;
    long long pos;

    if (writer == NULL || node == NULL || node->meta == NULL ||
        node->sac_hd == NULL || node->spectrum == NULL)
    {
        LOG_ERROR("spack_append_invalid_input",
                  "writer=%p node=%p meta=%p sac_hd=%p spectrum=%p",
                  (void *)writer, (void *)node,
                  node == NULL ? NULL : (void *)node->meta,
                  node == NULL ? NULL : (void *)node->sac_hd,
                  node == NULL ? NULL : (void *)node->spectrum);
        return -1;
    }

    meta = node->meta;
    if (selectTimestamp(writer, meta->timestamp) != 0)
    {
        return -1;
    }
    FillSegspecHeaderFromNode(&segspec_hd, node);

    payload_bytes = sizeof(complex) * (size_t)segspec_hd.nspec * (size_t)segspec_hd.nstep;
    bytes = (uint64_t)sizeof(SEGSPEC) + (uint64_t)payload_bytes;
    if (ensurePartForBytes(writer, bytes) != 0)
    {
        return -1;
    }

    pos = (long long)ftello(writer->pack_fp);
    if (pos < 0)
    {
        LOG_ERROR("spack_tell_failed", "path=\"%s\" errno=%d",
                  writer->pack_path, errno);
        return -1;
    }
    offset = (uint64_t)pos;

    if (fwrite(&segspec_hd, sizeof(SEGSPEC), 1, writer->pack_fp) != 1 ||
        fwrite(node->spectrum, payload_bytes, 1, writer->pack_fp) != 1)
    {
        LOG_ERROR("spack_write_failed",
                  "path=\"%s\" offset=%" PRIu64 " bytes=%" PRIu64,
                  writer->pack_path, offset, bytes);
        return -1;
    }
    writer->current_bytes = offset + bytes;

    writeTsvField(writer->tsv_fp, meta->timestamp);
    fprintf(writer->tsv_fp, "\t%d\t", writer->worker_id);
    writeTsvField(writer->tsv_fp, writer->pack_path);
    fprintf(writer->tsv_fp, "\t%" PRIu64 "\t%" PRIu64 "\t", offset, bytes);
    writeTsvField(writer->tsv_fp, meta->nsl_id);
    fputc('\t', writer->tsv_fp);
    writeTsvField(writer->tsv_fp, meta->network);
    fputc('\t', writer->tsv_fp);
    writeTsvField(writer->tsv_fp, meta->station);
    fputc('\t', writer->tsv_fp);
    writeTsvField(writer->tsv_fp, meta->location);
    fputc('\t', writer->tsv_fp);
    writeTsvField(writer->tsv_fp, meta->component);
    fprintf(writer->tsv_fp, "\t%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\n",
            segspec_hd.nstep, segspec_hd.nspec,
            segspec_hd.dt, segspec_hd.df,
            segspec_hd.stla, segspec_hd.stlo);

    return 0;
}

int SpackWriterCloseBatch(SpackWriter *writer)
{
    if (writer == NULL)
    {
        return 0;
    }
    return closeOpenFiles(writer);
}

void DestroySpackWriter(SpackWriter *writer)
{
    if (writer == NULL)
    {
        return;
    }

    (void)resetCurrentTimestamp(writer);
    free(writer->root);
    free(writer);
}
