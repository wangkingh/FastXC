#ifndef FS_H
#define FS_H

#include <sys/stat.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

int mkdir_p(const char *path, mode_t mode);

#ifdef __cplusplus
}
#endif

#endif
