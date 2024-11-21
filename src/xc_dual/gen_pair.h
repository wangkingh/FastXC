#ifndef _GEN_PAIR_H
#define _GEN_PAIR_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#define MAX_LINE_LENGTH 1024

typedef struct TimeInfo
{
    int year;
    int jday;
    int hourminute;
} TimeInfo;

typedef struct
{
    char filename[MAX_LINE_LENGTH];
    TimeInfo time;
} FileEntry;

typedef struct
{
    char source_path[MAX_LINE_LENGTH];
    char station_path[MAX_LINE_LENGTH];
    TimeInfo time;
    int index; // 添加以存储对的索引
} FilePair;

void find_matching_files(const char *filelist1, const char *filelist2, FilePair **matches, size_t *match_count);

#endif
