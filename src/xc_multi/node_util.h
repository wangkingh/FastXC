#ifndef NODE_UTIL_H
#define NODE_UTIL_H

#include "config.h"
#include "segspec.h"
#include "sac.h"
#include "complex.h"
#include <stdio.h>

typedef struct FilePaths
{
    char **paths;
    int count;
} FilePaths;

// typedef struct SPECNODE
// {
//     int valid;
//     char filepath[MAXLINE];
//     SEGSPEC head;
//     complex *pdata;
// } SPECNODE;

// typedef struct PAIRNODE
// {
//     size_t srcidx;
//     size_t staidx;
// } PAIRNODE;

typedef struct PAIRLIST_MANAGER
{
    size_t *src_idx_list;  // 存储每一源台对的源在输入列表的索引
    size_t *sta_idx_list;  // 存储每一源台对的源在输入列表的索引
    float *stla_list;      // 存储每个节点虚拟台的纬度
    float *stlo_list;      // 存储每个节点虚拟台的经度
    float *evla_list;      // 存储每个节点虚拟源的纬度
    float *evlo_list;      // 存储每个节点虚拟源的经度
    float *Gcarc_list;     // 存储每个节点源-台大圆路径长度
    float *Az_list;        // 方位角
    float *Baz_list;       // 反方位角
    float *Dist_list;      // 存储每个源-台节点的间距
    int *ok_flag_list;     // 每个节点是否可用的标识符
    size_t node_count;        // 这组表里由多少个节点(多少对台站对)
    int single_array_flag; // 是否单一阵列（源台一致）
    size_t src_start_idx;  // 源文件在[本批]输入列表中的[相对]起始索引
    size_t src_end_idx;    // 源文件在[本批]输入列表中的[相对]截止索引+1(满足左闭右开原则)
    size_t sta_start_idx;  // 台文件在[本批]输入列表中的[相对]起始索引
    size_t sta_end_idx;    // 台文件在[本批]输入列表中的[相对]截止索引+1(满足左闭右开原则)
} PAIRLIST_MANAGER;

#endif