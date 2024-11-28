#ifndef _PAR_READ_SPEC_H
#define _PAR_READ_SPEC_H

#include <stdlib.h>
#include <pthread.h>
#include <limits.h>
#include <stdio.h>
#include "read_segspec.h"
#include "node_util.h"
#include "complex.h"
#include "sac.h"
#include "segspec.h"

// 定义一个结构体来存储线程需要的数据
typedef struct
{
    FilePaths *pFileList; // 指向文件路径列表的指针
    complex *data_buffer; // 指向数据缓冲区的指针
    size_t start;
    size_t end;
    size_t file_start; // 开始处理的文件索引
    size_t file_end;   // 结束处理的文件索引
    size_t vec_count;  // 每一道数据的大小
} thread_info_read;

// 线程池结构体定义
typedef struct
{
    pthread_t *threads;
    thread_info_read *tinfo;
    size_t num_threads;
} ThreadPoolRead;

ThreadPoolRead *create_threadpool_read(size_t num_threads);
int ReadSpecArrayParallel(FilePaths *pFileList, complex *data_buffer, size_t start, size_t end,
                          size_t vec_count, ThreadPoolRead *pool);
void destroy_threadpool_read(ThreadPoolRead *pool);
#endif