#ifndef SAC2SPEC_PATH_UTILS_H
#define SAC2SPEC_PATH_UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

char *PathJoinAlloc(const char *root, const char *leaf);
char *PathAbsoluteDup(const char *path);
char *PathStringDup(const char *text);
int PathMakeDirectoryRecursive(const char *path);
void PathSafeTimestampLeaf(char *dst, size_t dst_size, const char *timestamp);
int PathTimestampLeafIsSafe(const char *timestamp);

#ifdef __cplusplus
}
#endif

#endif
