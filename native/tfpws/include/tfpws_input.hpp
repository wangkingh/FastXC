#ifndef TFPWS_INPUT_HPP
#define TFPWS_INPUT_HPP

#include <cerrno>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "logger.h"

static inline std::string trim_copy(const std::string &s)
{
    std::size_t first = 0;
    while (first < s.size() && std::isspace((unsigned char)s[first]))
        ++first;

    std::size_t last = s.size();
    while (last > first && std::isspace((unsigned char)s[last - 1]))
        --last;

    return s.substr(first, last - first);
}

static inline std::vector<int> parse_gpu_list(const char *gpu_list)
{
    std::vector<int> ids;
    std::stringstream ss(gpu_list ? gpu_list : "0");
    std::string token;

    while (std::getline(ss, token, ','))
    {
        token = trim_copy(token);
        if (token.empty())
        {
            LOG_ERROR("invalid_gpu_list",
                      "message=\"-G expects comma-separated non-empty GPU IDs\"");
            std::exit(EXIT_FAILURE);
        }

        char *end = NULL;
        errno = 0;
        long id = std::strtol(token.c_str(), &end, 10);
        if (errno != 0 || !end || *end != '\0' || id < 0 || id > INT_MAX)
        {
            LOG_ERROR("invalid_gpu_id",
                      "token=\"%s\"",
                      token.c_str());
            std::exit(EXIT_FAILURE);
        }
        ids.push_back((int)id);
    }

    if (ids.empty())
        ids.push_back(0);

    return ids;
}

static inline std::vector<double> parse_gpu_ram_limit_mib_list(const char *limit_list,
                                                               std::size_t expected_count)
{
    std::vector<double> limits;
    if (!limit_list)
    {
        limits.assign(expected_count, 0.0);
        return limits;
    }

    std::stringstream ss(limit_list);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        token = trim_copy(token);
        if (token.empty())
        {
            LOG_ERROR("invalid_memory_limit_list",
                      "message=\"-M expects comma-separated non-empty MiB limits\"");
            std::exit(EXIT_FAILURE);
        }

        char *end = NULL;
        errno = 0;
        double value = std::strtod(token.c_str(), &end);
        if (errno != 0 || !end || *end != '\0' || !std::isfinite(value) || value < 0.0)
        {
            LOG_ERROR("invalid_memory_limit",
                      "token=\"%s\"",
                      token.c_str());
            std::exit(EXIT_FAILURE);
        }
        limits.push_back(value);
    }

    if (limits.size() != expected_count)
    {
        LOG_ERROR("memory_limit_count_mismatch",
                  "limit_count=%zu worker_count=%zu",
                  limits.size(),
                  expected_count);
        std::exit(EXIT_FAILURE);
    }

    return limits;
}

#endif
