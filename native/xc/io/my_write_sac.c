#include "my_write_sac.h"
#include "logger.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int sac_sample_count(SACHEAD hd, size_t *samples)
{
    size_t n;
    if (hd.npts <= 0)
        return -1;
    n = (size_t)hd.npts;
    if (hd.iftype == IXY)
    {
        if (n > SIZE_MAX / 2)
            return -1;
        n *= 2;
    }
    *samples = n;
    return 0;
}

static int sac_data_bytes(SACHEAD hd, size_t *bytes)
{
    size_t samples;
    if (sac_sample_count(hd, &samples) != 0)
        return -1;
    if (samples > SIZE_MAX / sizeof(float))
        return -1;
    *bytes = samples * sizeof(float);
    return 0;
}

float *read_sac_buffer(const char *name, SACHEAD *sac_hd, float *buffer)
{
    FILE *strm = NULL;
    size_t sz = 0;

    if (sac_data_bytes(*sac_hd, &sz) != 0)
    {
        LOG_ERROR("sac_data_size_invalid", "path=\"%s\" op=read", name);
        return NULL;
    }

    strm = fopen(name, "rb");
    if (!strm)
    {
        LOG_ERROR("sac_open_failed", "path=\"%s\" mode=read error=\"%s\"",
                  name, strerror(errno));
        return NULL;
    }
    if (fseek(strm, sizeof(SACHEAD), SEEK_SET) != 0)
    {
        LOG_ERROR("sac_header_seek_failed", "path=\"%s\" error=\"%s\"",
                  name, strerror(errno));
        fclose(strm);
        return NULL;
    }
    if (fread(buffer, sz, 1, strm) != 1)
    {
        LOG_ERROR("sac_data_read_failed", "path=\"%s\"", name);
        fclose(strm);
        return NULL;
    }
    fclose(strm);

#ifdef BYTE_SWAP
    if (sz > (size_t)INT_MAX)
    {
        LOG_ERROR("sac_byte_swap_size_exceeded", "path=\"%s\" bytes=%zu", name, sz);
        return NULL;
    }
    swab4((char *)buffer, (int)sz);
#endif
    return buffer;
}

static int file_exists(const char *name)
{
    struct stat st;
    return stat(name, &st) == 0;
}

static int file_size_bytes(const char *name, long long *size)
{
    struct stat st;
    if (stat(name, &st) != 0)
        return -1;
    *size = (long long)st.st_size;
    return 0;
}

static int write_new_sac(const char *name, SACHEAD hd, const float *ar)
{
    FILE *strm = NULL;
    size_t sz = 0;
    float *data = NULL;

    if (sac_data_bytes(hd, &sz) != 0)
    {
        LOG_ERROR("sac_data_size_invalid", "path=\"%s\" op=write_new", name);
        return -1;
    }

    data = (float *)malloc(sz);
    if (!data)
    {
        LOG_ERROR("sac_write_buffer_alloc_failed",
                  "path=\"%s\" bytes_mib=%.3f mode=new",
                  name, sz / (1024.0 * 1024.0));
        return -1;
    }
    memcpy(data, ar, sz);

#ifdef BYTE_SWAP
    if (sz > (size_t)INT_MAX)
    {
        LOG_ERROR("sac_byte_swap_size_exceeded", "path=\"%s\" bytes=%zu", name, sz);
        free(data);
        return -1;
    }
    swab4((char *)data, (int)sz);
    swab4((char *)&hd, HD_SIZE);
#endif

    strm = fopen(name, "wb");
    if (!strm)
    {
        LOG_ERROR("sac_open_failed", "path=\"%s\" mode=write error=\"%s\"",
                  name, strerror(errno));
        free(data);
        return -1;
    }
    if (fwrite(&hd, sizeof(SACHEAD), 1, strm) != 1 ||
        fwrite(data, sz, 1, strm) != 1)
    {
        LOG_ERROR("sac_write_failed", "path=\"%s\"", name);
        fclose(strm);
        free(data);
        return -1;
    }
    fclose(strm);
    free(data);
    return 0;
}

static int append_write_sac(const char *name, SACHEAD hd, const float *ar)
{
    FILE *strm = NULL;
    size_t sz = 0;
    float *data = NULL;

    if (!file_exists(name))
        return write_new_sac(name, hd, ar);
    if (sac_data_bytes(hd, &sz) != 0)
    {
        LOG_ERROR("sac_data_size_invalid", "path=\"%s\" op=append", name);
        return -1;
    }

    data = (float *)malloc(sz);
    if (!data)
    {
        LOG_ERROR("sac_write_buffer_alloc_failed",
                  "path=\"%s\" bytes_mib=%.3f mode=append",
                  name, sz / (1024.0 * 1024.0));
        return -1;
    }
    memcpy(data, ar, sz);

#ifdef BYTE_SWAP
    if (sz > (size_t)INT_MAX)
    {
        LOG_ERROR("sac_byte_swap_size_exceeded", "path=\"%s\" bytes=%zu", name, sz);
        free(data);
        return -1;
    }
    swab4((char *)data, (int)sz);
    swab4((char *)&hd, HD_SIZE);
#endif

    strm = fopen(name, "ab");
    if (!strm)
    {
        LOG_ERROR("sac_open_failed", "path=\"%s\" mode=append error=\"%s\"",
                  name, strerror(errno));
        free(data);
        return -1;
    }
    if (fwrite(&hd, sizeof(SACHEAD), 1, strm) != 1 ||
        fwrite(data, sz, 1, strm) != 1)
    {
        LOG_ERROR("sac_write_failed", "path=\"%s\" mode=append", name);
        fclose(strm);
        free(data);
        return -1;
    }
    fclose(strm);
    free(data);
    return 0;
}

static int aggregate_write_sac(const char *name, SACHEAD hd, const float *ar)
{
    SACHEAD old_hd;
    long long file_size = 0;
    long long expected_size = 0;
    size_t old_bytes = 0;
    size_t new_bytes = 0;
    size_t samples = 0;
    float *old_data = NULL;

    if (!file_exists(name))
        return write_new_sac(name, hd, ar);
    if (read_sachead(name, &old_hd) != 0)
    {
        LOG_ERROR("sac_existing_header_read_failed", "path=\"%s\"", name);
        return -1;
    }
    if (hd.npts != old_hd.npts || hd.iftype != old_hd.iftype)
    {
        LOG_ERROR("sac_aggregate_shape_mismatch",
                  "path=\"%s\" npts=%d old_npts=%d iftype=%d old_iftype=%d",
                  name, hd.npts, old_hd.npts, hd.iftype, old_hd.iftype);
        return -1;
    }
    if (sac_data_bytes(old_hd, &old_bytes) != 0 ||
        sac_data_bytes(hd, &new_bytes) != 0 ||
        sac_sample_count(hd, &samples) != 0 ||
        old_bytes > (size_t)LLONG_MAX - sizeof(SACHEAD))
    {
        LOG_ERROR("sac_data_size_invalid", "path=\"%s\" op=aggregate", name);
        return -1;
    }
    expected_size = (long long)(sizeof(SACHEAD) + old_bytes);
    if (file_size_bytes(name, &file_size) != 0 || file_size != expected_size)
    {
        LOG_ERROR("sac_aggregate_file_size_mismatch",
                  "path=\"%s\" file_size=%lld expected_size=%lld",
                  name, file_size, expected_size);
        return -1;
    }

    old_data = (float *)malloc(new_bytes);
    if (!old_data)
    {
        LOG_ERROR("sac_write_buffer_alloc_failed",
                  "path=\"%s\" bytes_mib=%.3f mode=aggregate",
                  name, new_bytes / (1024.0 * 1024.0));
        return -1;
    }
    if (read_sac_buffer(name, &old_hd, old_data) == NULL)
    {
        LOG_ERROR("sac_existing_data_read_failed", "path=\"%s\"", name);
        free(old_data);
        return -1;
    }

    for (size_t i = 0; i < samples; ++i)
        old_data[i] += ar[i];

    old_hd.unused27 = old_hd.unused27 + 1;
    hd.unused27 = old_hd.unused27;
    int rc = write_new_sac(name, hd, old_data);
    free(old_data);
    return rc;
}

int my_write_sac(const char *name, SACHEAD hd, const float *ar, int write_mode)
{
    switch (write_mode)
    {
    case MODE_APPEND:
        return append_write_sac(name, hd, ar);
    case MODE_AGGREGATE:
        return aggregate_write_sac(name, hd, ar);
    default:
        LOG_ERROR("sac_write_mode_unknown", "write_mode=%d", write_mode);
        return -1;
    }
}
