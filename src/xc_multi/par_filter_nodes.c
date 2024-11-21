#include "par_filter_nodes.h"

ThreadPoolFilter *create_threadpool_filter_nodes(size_t num_threads)
{
    ThreadPoolFilter *pool = malloc(sizeof(ThreadPoolFilter));
    if (pool == NULL)
    {
        fprintf(stderr, "Memory allocation failed for ThreadPoolFilter\n");
        return NULL;
    }
    pool->threads = malloc(num_threads * sizeof(pthread_t));
    pool->tinfo = malloc(num_threads * sizeof(thread_info_filter));
    pool->num_threads = num_threads;
    return pool;
}

// 销毁线程池的函数
void destroy_threadpool_filter_nodes(ThreadPoolFilter *pool)
{
    free(pool->threads);
    free(pool->tinfo);
    free(pool);
}

// 线程函数，用于处理文件
void *filter_nodes(void *arg)
{
    thread_info_filter *tinfo = (thread_info_filter *)arg;
    SEGSPEC *phd_src = NULL;
    SEGSPEC *phd_sta = NULL;
    CpuMalloc((void **)&phd_src, sizeof(SEGSPEC));
    CpuMalloc((void **)&phd_sta, sizeof(SEGSPEC));
    FilePaths *src_path = tinfo->srcFileList;
    FilePaths *sta_path = tinfo->staFileList;
    PAIRLIST_MANAGER *manager = tinfo->manager;
    size_t *src_idx_list = manager->src_idx_list;
    size_t *sta_idx_list = manager->sta_idx_list;
    size_t srcfile_start_idx = manager->src_start_idx;
    size_t stafile_start_idx = manager->sta_start_idx;
    float *stla_list = manager->stla_list;
    float *stlo_list = manager->stlo_list;
    float *evla_list = manager->evla_list;
    float *evlo_list = manager->evlo_list;
    float *Gcarc_list = manager->Gcarc_list;
    float *Az_list = manager->Az_list;
    float *Baz_list = manager->Baz_list;
    float *Dist_list = manager->Dist_list;

    double tempGcarc, tempAz, tempBaz, tempDist;
    float upper_distance_boundary = tinfo->max_distance;

    for (size_t i = tinfo->start; i < tinfo->end; i++)
    {
            size_t src_idx = srcfile_start_idx + src_idx_list[i];
            size_t sta_idx = stafile_start_idx + sta_idx_list[i];
            if (read_spechead(src_path->paths[src_idx], phd_src) == -1)
            {
                continue;
            }
            if (read_spechead(sta_path->paths[sta_idx], phd_sta) == -1)
            {
                continue;
            }
            double stla = phd_sta->stla;
            double stlo = phd_sta->stlo;
            double evla = phd_src->stla;
            double evlo = phd_src->stlo;
            distkm_az_baz_Rudoe(evlo, evla, stlo, stla,
                                &tempGcarc, &tempAz, &tempBaz, &tempDist);
            // Convert back to float after the function call
            evla_list[i] = evla;
            evlo_list[i] = evlo;
            stla_list[i] = stla;
            stlo_list[i] = stlo;
            Gcarc_list[i] = tempGcarc;
            Az_list[i] = tempAz;
            Baz_list[i] = tempBaz;
            Dist_list[i] = tempDist;

            // check if this node is OK
            if (tempDist < upper_distance_boundary)
            {
                manager->ok_flag_list[i] = 1; // if distance < upper boundary , nodes is ok
            }
    }
    CpuFree((void **)&phd_src);
    CpuFree((void **)&phd_sta);
    return NULL;
}

// 并行处理manager里的节点
int FilterNodeParallel(PAIRLIST_MANAGER *manager, FilePaths *src_paths, FilePaths *sta_paths,
                       ThreadPoolFilter *pool, float max_distance)
{
    size_t total_nodes = manager->node_count;
    size_t nodes_per_thread = total_nodes / pool->num_threads;
    size_t nodes_remainder = total_nodes % pool->num_threads;
    size_t start = 0;

    for (int i = 0; i < pool->num_threads; i++)
    {
        pool->tinfo[i].start = start;
        pool->tinfo[i].end = start + nodes_per_thread + (i < nodes_remainder ? 1 : 0);
        pool->tinfo[i].manager = manager;
        pool->tinfo[i].max_distance = max_distance;
        start = pool->tinfo[i].end;
        pool->tinfo[i].srcFileList = src_paths;
        pool->tinfo[i].staFileList = sta_paths;
        if (pthread_create(&pool->threads[i], NULL, filter_nodes, &pool->tinfo[i]))
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

void CompressManager(PAIRLIST_MANAGER *manager)
{
    size_t j = 0; // 新数组的索引
    for (size_t i = 0; i < manager->node_count; i++)
    {
        if (manager->ok_flag_list[i] == 1)
        {
            manager->src_idx_list[j] = manager->src_idx_list[i];
            manager->sta_idx_list[j] = manager->sta_idx_list[i];
            manager->stla_list[j] = manager->stla_list[i];
            manager->stlo_list[j] = manager->stlo_list[i];
            manager->evla_list[j] = manager->evla_list[i];
            manager->evlo_list[j] = manager->evlo_list[i];
            manager->Gcarc_list[j] = manager->Gcarc_list[i];
            manager->Az_list[j] = manager->Az_list[i];
            manager->Baz_list[j] = manager->Baz_list[i];
            manager->Dist_list[j] = manager->Dist_list[i];
            j++;
        }
    }
    manager->node_count = j; // 更新节点数量
}
