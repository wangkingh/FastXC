#ifndef _READ_SEGSPEC_H
#define _READ_SEGSPEC_H
#include <dirent.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include "complex.h"
#include "segspec.h"
#include "gen_pair.h"

typedef struct thread_info_read
{
    int start;
    int end;
    FilePair *filepairs;
    complex *src_buffer;
    complex *sta_buffer;
    size_t vec_size;
} thread_info_read;

typedef struct ThreadPoolRead
{
    pthread_t *threads;
    thread_info_read *tinfo;
    size_t num_threads;
} ThreadPoolRead;

// 创建读取进程池
ThreadPoolRead *create_threadpool_read(size_t num_threads);

// 销毁读取进程池
void destroy_threadpool_read(ThreadPoolRead *pool);

// 并行读取segspec文件
int parallel_read_segspec(ThreadPoolRead *pool, size_t proccnt, FilePair *pairs,
                          complex *src_buffer, complex *sta_buffer, size_t block_size,
                          int num_threads);

// 读取segspec文件的头
int read_spechead(char *file_path, SEGSPEC *hd);
#endif