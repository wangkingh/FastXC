#include "fs.h"

#include <errno.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int path_is_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        return 0;
    return S_ISDIR(st.st_mode);
}

int mkdir_p(const char *path, mode_t mode)
{
    char tmp[PATH_MAX];
    size_t len = 0;

    if (!path || path[0] == '\0')
    {
        errno = EINVAL;
        return -1;
    }

    len = strnlen(path, sizeof(tmp));
    if (len >= sizeof(tmp))
    {
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(tmp, path, len + 1);

    while (len > 1 && tmp[len - 1] == '/')
        tmp[--len] = '\0';

    if (path_is_dir(tmp))
        return 0;

    for (char *p = tmp + 1; *p; ++p)
    {
        if (*p != '/')
            continue;
        *p = '\0';
        if (tmp[0] != '\0' && mkdir(tmp, mode) != 0)
        {
            if (errno != EEXIST || !path_is_dir(tmp))
            {
                *p = '/';
                return -1;
            }
        }
        *p = '/';
    }

    if (mkdir(tmp, mode) != 0)
    {
        if (errno != EEXIST || !path_is_dir(tmp))
            return -1;
    }
    return 0;
}
