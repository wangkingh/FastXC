#include "include/path_utils.h"

#include "include/config.h"
#include "include/logger.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

static int pathEndsWithSeparator(const char *path)
{
    size_t len;
    if (path == NULL)
    {
        return 0;
    }
    len = strlen(path);
    return len > 0 && (path[len - 1] == '/' || path[len - 1] == '\\');
}

static int mkdirIfNeeded(const char *path)
{
    struct stat st;
    int mkdir_errno;

    if (mkdir(path, 0755) == 0)
    {
        return 0;
    }

    mkdir_errno = errno;
    if (mkdir_errno == EEXIST)
    {
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode))
        {
            return 0;
        }
        LOG_ERROR("path_exists_not_directory", "path=\"%s\"", path);
        return -1;
    }

    LOG_ERROR("mkdir_failed", "path=\"%s\" errno=%d", path, mkdir_errno);
    return -1;
}

char *PathStringDup(const char *text)
{
    const char *src = text == NULL ? "" : text;
    size_t len = strlen(src);
    char *copy = (char *)malloc(len + 1);
    if (copy == NULL)
    {
        return NULL;
    }
    memcpy(copy, src, len + 1);
    return copy;
}

char *PathJoinAlloc(const char *root, const char *leaf)
{
    const char *safe_root = root == NULL ? "" : root;
    const char *safe_leaf = leaf == NULL ? "" : leaf;
    const char *sep = pathEndsWithSeparator(safe_root) || safe_root[0] == '\0' ? "" : "/";
    int needed = snprintf(NULL, 0, "%s%s%s", safe_root, sep, safe_leaf);
    char *path;
    if (needed < 0)
    {
        return NULL;
    }

    path = (char *)malloc((size_t)needed + 1);
    if (path == NULL)
    {
        return NULL;
    }
    snprintf(path, (size_t)needed + 1, "%s%s%s", safe_root, sep, safe_leaf);
    return path;
}

char *PathAbsoluteDup(const char *path)
{
    char *resolved;
    if (path == NULL)
    {
        return NULL;
    }
    resolved = realpath(path, NULL);
    if (resolved != NULL)
    {
        return resolved;
    }
    return PathStringDup(path);
}

int PathMakeDirectoryRecursive(const char *path)
{
    char current[MAXPATH];
    size_t len;
    if (path == NULL || path[0] == '\0')
    {
        return 0;
    }
    len = strlen(path);
    if (len >= sizeof(current))
    {
        LOG_ERROR("path_too_long", "path=\"%s\"", path);
        return -1;
    }
    memcpy(current, path, len + 1);

    for (char *p = current + 1; *p != '\0'; p++)
    {
        if (*p == '/')
        {
            *p = '\0';
            if (mkdirIfNeeded(current) != 0)
            {
                return -1;
            }
            *p = '/';
        }
    }
    if (mkdirIfNeeded(current) != 0)
    {
        return -1;
    }
    return 0;
}

static int isSafeTimestampChar(char ch)
{
    return (ch >= '0' && ch <= '9') ||
           (ch >= 'A' && ch <= 'Z') ||
           (ch >= 'a' && ch <= 'z') ||
           ch == '.' || ch == '_' || ch == ':' || ch == '-';
}

void PathSafeTimestampLeaf(char *dst, size_t dst_size, const char *timestamp)
{
    size_t j = 0;
    const char *src = timestamp == NULL ? "" : timestamp;
    if (dst_size == 0)
    {
        return;
    }
    for (const char *p = src; *p != '\0' && j + 1 < dst_size; p++)
    {
        dst[j++] = isSafeTimestampChar(*p) ? *p : '-';
    }
    while (j > 0 && dst[j - 1] == '-')
    {
        j--;
    }
    dst[j] = '\0';
    if (j == 0)
    {
        snprintf(dst, dst_size, "timestamp");
    }
}

int PathTimestampLeafIsSafe(const char *timestamp)
{
    char leaf[MAXNAME];
    if (timestamp == NULL || timestamp[0] == '\0')
    {
        return 0;
    }
    PathSafeTimestampLeaf(leaf, sizeof(leaf), timestamp);
    if (strcmp(timestamp, leaf) != 0)
    {
        return 0;
    }
    if (strcmp(leaf, ".") == 0 || strcmp(leaf, "..") == 0)
    {
        return 0;
    }
    return 1;
}
