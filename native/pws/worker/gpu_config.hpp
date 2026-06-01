#ifndef GPU_CONFIG_HPP
#define GPU_CONFIG_HPP

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "logger.h"
#include "string_util.hpp"

struct PwsGpuWorkerConfig
{
    std::vector<int> gpu_ids;
    std::vector<std::size_t> gpu_memory_limits_mib;
    std::vector<std::size_t> physical_worker_counts;
};

static inline int parse_gpu_list(const char *gpu_list, std::vector<int> *ids)
{
    ids->clear();
    std::string raw = trim_copy(gpu_list ? gpu_list : "0");
    if (raw.empty() || raw[0] == ',' || raw[raw.size() - 1] == ',')
    {
        LOG_ERROR("invalid_gpu_workers", "value=\"%s\"", raw.c_str());
        return 1;
    }

    std::stringstream ss(raw);
    std::string token;

    while (std::getline(ss, token, ','))
    {
        token = trim_copy(token);
        if (token.empty())
        {
            LOG_ERROR("invalid_gpu_workers", "message=\"empty GPU id\"");
            return 1;
        }

        char *end = NULL;
        errno = 0;
        long value = std::strtol(token.c_str(), &end, 10);
        if (errno != 0 || end == token.c_str() || *end != '\0' ||
            value < 0 || value > INT_MAX)
        {
            LOG_ERROR("invalid_gpu_id", "value=\"%s\"", token.c_str());
            return 1;
        }
        ids->push_back((int)value);
    }

    if (ids->empty())
    {
        LOG_ERROR("invalid_gpu_workers", "message=\"no GPU ids\"");
        return 1;
    }

    return 0;
}

static inline int parse_gpu_memory_list(const char *memory_list,
                                        std::size_t expected_count,
                                        std::vector<std::size_t> *limits_mib)
{
    limits_mib->clear();
    if (!memory_list)
    {
        limits_mib->assign(expected_count, 0);
        return 0;
    }

    std::string raw = trim_copy(memory_list);
    if (raw.empty() || raw[0] == ',' || raw[raw.size() - 1] == ',')
    {
        LOG_ERROR("invalid_gpu_memory_mib", "value=\"%s\"", raw.c_str());
        return 1;
    }

    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        token = trim_copy(token);
        if (token.empty())
        {
            LOG_ERROR("invalid_gpu_memory_mib", "message=\"empty MiB value\"");
            return 1;
        }

        char *end = NULL;
        errno = 0;
        unsigned long long value = std::strtoull(token.c_str(), &end, 10);
        if (token[0] == '-' || errno != 0 || end == token.c_str() || *end != '\0')
        {
            LOG_ERROR("invalid_gpu_memory_mib", "value=\"%s\"", token.c_str());
            return 1;
        }
        limits_mib->push_back((std::size_t)value);
    }

    if (limits_mib->size() != expected_count)
    {
        LOG_ERROR("gpu_memory_count_mismatch",
                  "memory_count=%zu worker_count=%zu",
                  limits_mib->size(), expected_count);
        return 1;
    }

    return 0;
}

static inline std::vector<std::size_t> count_physical_gpu_workers(const std::vector<int> &ids)
{
    std::vector<std::size_t> counts(ids.size(), 0);
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        for (std::size_t j = 0; j < ids.size(); ++j)
            if (ids[i] == ids[j])
                ++counts[i];
    }

    return counts;
}

static inline int parse_pws_gpu_worker_config(const char *gpu_list,
                                              const char *gpu_memory_list,
                                              PwsGpuWorkerConfig *config)
{
    config->gpu_ids.clear();
    config->gpu_memory_limits_mib.clear();
    config->physical_worker_counts.clear();

    if (parse_gpu_list(gpu_list, &config->gpu_ids) != 0)
        return 1;

    if (parse_gpu_memory_list(gpu_memory_list,
                              config->gpu_ids.size(),
                              &config->gpu_memory_limits_mib) != 0)
        return 1;

    config->physical_worker_counts = count_physical_gpu_workers(config->gpu_ids);
    return 0;
}

#endif
