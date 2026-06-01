#include "path_util.h"

#include <errno.h>
#include <limits.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int path_is_dir(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int mkdir_p(const char *dir)
{
    char tmp[PATH_MAX];
    size_t len;

    if (!dir || dir[0] == '\0')
    {
        errno = EINVAL;
        return -1;
    }

    len = strnlen(dir, sizeof(tmp));
    if (len >= sizeof(tmp))
    {
        errno = ENAMETOOLONG;
        return -1;
    }

    memcpy(tmp, dir, len + 1);
    while (len > 1 && tmp[len - 1] == '/')
        tmp[--len] = '\0';

    if (path_is_dir(tmp))
        return 0;

    for (char *p = tmp + 1; *p; ++p)
    {
        if (*p != '/')
            continue;

        *p = '\0';
        if (mkdir(tmp, 0755) != 0 && (errno != EEXIST || !path_is_dir(tmp)))
        {
            *p = '/';
            return -1;
        }
        *p = '/';
    }

    if (mkdir(tmp, 0755) != 0 && (errno != EEXIST || !path_is_dir(tmp)))
        return -1;

    return 0;
}

int ensure_parent_dir(const char *path)
{
    char dir[PATH_MAX];
    const char *last_slash;
    size_t len;

    if (!path || path[0] == '\0')
    {
        errno = EINVAL;
        return -1;
    }

    last_slash = strrchr(path, '/');
    if (!last_slash || last_slash == path)
        return 0;

    len = (size_t)(last_slash - path);
    if (len >= sizeof(dir))
    {
        errno = ENAMETOOLONG;
        return -1;
    }

    memcpy(dir, path, len);
    dir[len] = '\0';

    return mkdir_p(dir);
}
