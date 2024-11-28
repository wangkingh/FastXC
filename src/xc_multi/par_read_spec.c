#include "par_read_spec.h"
#include "util.h"

// 创建线程池的函数
ThreadPoolRead *create_threadpool_read(size_t num_threads)
{
    ThreadPoolRead *pool = malloc(sizeof(ThreadPoolRead));
    pool->threads = malloc(num_threads * sizeof(pthread_t));
    pool->tinfo = malloc(num_threads * sizeof(thread_info_read));
    pool->num_threads = num_threads;
    return pool;
}

// 销毁线程池的函数
void destroy_threadpool_read(ThreadPoolRead *pool)
{
    free(pool->threads);
    free(pool->tinfo);
    free(pool);
}

// 线程函数，用于处理文件
void *read_files(void *arg)
{
    thread_info_read *tinfo = (thread_info_read *)arg;
    SEGSPEC *tmp_hd = NULL;
    CpuMalloc((void **)&tmp_hd, sizeof(SEGSPEC));

    // 为每个线程处理文件
    for (size_t i = tinfo->start; i < tinfo->end; i++)
    {
        size_t file_idx = tinfo->file_start + i;
        size_t offset = tinfo->vec_count * i;
        if (read_spec_buffer(tinfo->pFileList->paths[file_idx], tmp_hd, tinfo->data_buffer + offset) == NULL)
        {
            continue;
        }
    }

    CpuFree((void **)&tmp_hd);
    return NULL;
}

// 并行生成SPECNODE数组的函数
int ReadSpecArrayParallel(FilePaths *pFileList, complex *data_buffer, size_t file_start, size_t file_end,
                          size_t vec_count, ThreadPoolRead *pool)
{
    size_t total_files = file_end - file_start;
    size_t files_per_thread = total_files / pool->num_threads;
    size_t remainder = total_files % pool->num_threads;

    size_t current_file_start = file_start;
    size_t start = 0;
    for (int i = 0; i < pool->num_threads; i++)
    {
        pool->tinfo[i].start = start;
        pool->tinfo[i].end = start + files_per_thread + (i < remainder ? 1 : 0);
        start = pool->tinfo[i].end;
        pool->tinfo[i].pFileList = pFileList;
        pool->tinfo[i].data_buffer = data_buffer;
        pool->tinfo[i].file_start = current_file_start;
        pool->tinfo[i].vec_count = vec_count;

        if (pthread_create(&pool->threads[i], NULL, read_files, &pool->tinfo[i]))
        {
            fprintf(stderr, "Error creating thread\n");
            return -1;
        }
    }
    for (int i = 0; i < pool->num_threads; i++)
    {
        if (pthread_join(pool->threads[i], NULL))
        {
            fprintf(stderr, "Error joining thread\n");
            return -1;
        }
    }

    return 0;
}
