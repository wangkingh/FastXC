#ifndef _PARALLEL_RW_DATA_H_
#define _PARALLEL_RW_DATA_H_
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "in_out_node.h"
#include "sac.h"

typedef struct thread_info_read
{
  int start;
  int end;
  InOutNode *InOutList;
} thread_info_read;

typedef struct ThreadPoolRead
{
  pthread_t *threads;
  thread_info_read *tinfo;
  size_t num_threads;
} ThreadPoolRead;

typedef struct thread_info_write
{
  size_t start;
  size_t end;
  InOutNode *InOutList;
} thread_info_write;

typedef struct ThreadPoolWrite
{
  pthread_t *threads;
  thread_info_write *tinfo;
  size_t num_threads;
} ThreadPoolWrite;

ThreadPoolRead *create_threadpool_read(size_t num_threads);

ThreadPoolWrite *create_threadpool_write(size_t num_threads);

int parallel_read_sac(ThreadPoolRead *pool, size_t proccnt, InOutNode *pInOutList, int num_threads);

int parallel_write_spec(ThreadPoolWrite *pool, size_t h_proccnt, InOutNode *pInOutList, int num_threads);

void destroy_threadpool_read(ThreadPoolRead *pool);

void destroy_threadpool_write(ThreadPoolWrite *pool);

#endif
