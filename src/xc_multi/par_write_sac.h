#ifndef _PAR_WRITE_H
#define _PAR_WRITE_H
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "node_util.h"
#include "gen_ccfpath.h"

typedef struct
{
    PAIRLIST_MANAGER *manager; // 包含了源台索引
    FilePaths *src_path_list;  // 虚拟源文件路径列表
    FilePaths *sta_path_list;  // 虚拟台文件路径列表
    size_t start;              // 处理的起始索引
    size_t end;                // 处理的结束索引
    float *ncf_buffer;         // 用于存储交叉相关函数结果的缓冲区
    float delta;               // 互相关采样间隔
    int ncc;                   // 交叉相关数据点的数量
    float cc_length;           // lapse time  半个互相关长度
    char *output_dir;          // 输出文件路径
} thread_info_write;

typedef struct
{
    pthread_t *threads;
    thread_info_write *tinfo;
    size_t num_threads;
} ThreadWritePool;

ThreadWritePool *create_threadwrite_pool(size_t num_threads);

void destroy_threadwrite_pool(ThreadWritePool *pool);

int write_pairs_parallel(PAIRLIST_MANAGER *manager, FilePaths *src_path_list, FilePaths *sta_path_list,
                         float *ncf_buffer, float delta, int ncc, float cc_length,
                         char *output_dir, ThreadWritePool *pool);

#endif
