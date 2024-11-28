#ifndef _PAR_FILTER_NODELIST
#define _PAR_FILTER_NODELIST

#include <stdlib.h>
#include <pthread.h>
#include <limits.h>
#include <stdio.h>
#include "node_util.h"
#include "read_segspec.h"
#include "segspec.h"
#include "util.h"
#include "cal_dist.h"

// 定义一个结构体来存储线程需要的数据
typedef struct
{
    FilePaths *srcFileList;    // 指向文件路径列表的指针
    FilePaths *staFileList;    // 指向文件路径列表的指针
    PAIRLIST_MANAGER *manager; // 输入整一个待处理的列表
    size_t start;              // 开始处理的台站对节点索引
    size_t end;                // 结束处理的台站对索引索引
    float max_distance;
} thread_info_filter;

// 线程池结构体定义
typedef struct
{
    pthread_t *threads;
    thread_info_filter *tinfo;
    size_t num_threads;
} ThreadPoolFilter;

ThreadPoolFilter *create_threadpool_filter_nodes(size_t num_threads);

void destroy_threadpool_filter_nodes(ThreadPoolFilter *pool);

int FilterNodeParallel(PAIRLIST_MANAGER *manager, FilePaths *src_paths, FilePaths *sta_paths,
                       ThreadPoolFilter *pool, float max_distance);

void CompressManager(PAIRLIST_MANAGER *manager);
#endif