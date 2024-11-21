#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "node_util.h"
#include "par_write_sac.h"

ThreadWritePool *create_threadwrite_pool(size_t num_threads)
{
    ThreadWritePool *pool = malloc(sizeof(ThreadWritePool));
    if (!pool)
    {
        fprintf(stderr, "Memory allocation failed for thread pool\n");
        return NULL;
    }
    pool->threads = malloc(num_threads * sizeof(pthread_t));
    pool->tinfo = malloc(num_threads * sizeof(thread_info_write));
    if (!pool->threads || !pool->tinfo)
    {
        free(pool->threads);
        free(pool->tinfo);
        free(pool);
        fprintf(stderr, "Memory allocation failed for threads or thread info\n");
        return NULL;
    }
    pool->num_threads = num_threads;
    return pool;
}

void destroy_threadwrite_pool(ThreadWritePool *pool)
{
    if (pool)
    {
        free(pool->threads);
        free(pool->tinfo);
        free(pool);
    }
}

void *write_file(void *arg)
{
    thread_info_write *write_info = (thread_info_write *)arg;
    PAIRLIST_MANAGER *manager = write_info->manager;
    FilePaths *src_path_list = write_info->src_path_list;
    FilePaths *sta_path_list = write_info->sta_path_list;
    size_t srcfile_start_idx = manager->src_start_idx;
    size_t stafile_start_idx = manager->sta_start_idx;
    float delta = write_info->delta;
    int ncc = write_info->ncc;
    float cc_length = write_info->cc_length;
    char *output_dir = write_info->output_dir;
    for (size_t node_idx = write_info->start; node_idx < write_info->end; node_idx++)
    {
        char ncf_path[MAXPATH];
        SACHEAD ncf_hd;
        int buffer_offset = node_idx * (write_info->ncc);
        size_t src_idx = srcfile_start_idx + write_info->manager->src_idx_list[node_idx];
        size_t sta_idx = stafile_start_idx + write_info->manager->sta_idx_list[node_idx];
        float stla = manager->stla_list[node_idx];
        float stlo = manager->stlo_list[node_idx];
        float evla = manager->evla_list[node_idx];
        float evlo = manager->evlo_list[node_idx];
        float gcarc = manager->Gcarc_list[node_idx];
        float az = manager->Az_list[node_idx];
        float baz = manager->Baz_list[node_idx];
        float dist = manager->Dist_list[node_idx];
        char *src_path = src_path_list->paths[src_idx];
        char *sta_path = sta_path_list->paths[sta_idx];
        GenCCFPath(ncf_path, src_path, sta_path, output_dir); // 这个函数包含了生成路径和创建父目录
        SacheadProcess(&ncf_hd, stla, stlo, evla, evlo, gcarc, az, baz, dist, delta, ncc, cc_length);
        write_sac(ncf_path, ncf_hd, (write_info->ncf_buffer) + buffer_offset);
    }
    return NULL;
}

int write_pairs_parallel(PAIRLIST_MANAGER *manager, FilePaths *src_path_list, FilePaths *sta_path_list,
                         float *ncf_buffer, float delta, int ncc, float cc_length,
                         char *output_dir, ThreadWritePool *pool)
{
    size_t nodes_count = manager->node_count;
    size_t nodes_per_thread = nodes_count / pool->num_threads;
    size_t remainder = nodes_count % pool->num_threads;

    size_t start = 0;

    for (int i = 0; i < pool->num_threads; i++)
    {
        pool->tinfo[i].manager = manager;
        pool->tinfo[i].src_path_list = src_path_list;
        pool->tinfo[i].sta_path_list = sta_path_list;
        pool->tinfo[i].start = start;
        pool->tinfo[i].end = start + nodes_per_thread + (i < remainder ? 1 : 0);
        pool->tinfo[i].ncf_buffer = ncf_buffer;
        pool->tinfo[i].delta = delta;
        pool->tinfo[i].ncc = ncc;
        pool->tinfo[i].cc_length = cc_length;
        pool->tinfo[i].output_dir = output_dir;
        start = pool->tinfo[i].end;

        if (pthread_create(&pool->threads[i], NULL, write_file, &pool->tinfo[i]))
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
