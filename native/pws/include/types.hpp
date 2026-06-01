#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstddef>
#include <string>
#include <vector>

extern "C"
{
#include "sac.h"
}

struct PwsHostItem
{
    std::string input_path;
    unsigned num_segments;
    unsigned nsamples;
    unsigned ngroups;
    SACHEAD header;
    float *prestack_data;
    float *linear_stack;
    float *group_weights;
};

struct PwsHostBatch
{
    unsigned nsamples;
    std::size_t total_groups;
    std::vector<PwsHostItem> items;

    PwsHostBatch() : nsamples(0), total_groups(0) {}
};

#endif
