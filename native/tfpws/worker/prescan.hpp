#ifndef TFPWS_PRESCAN_HPP
#define TFPWS_PRESCAN_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "sourcepack_io.hpp"
#include "tfpws_compute.hpp"
#include "tfpws_types.hpp"

extern "C"
{
#include "arguproc.h"
}

struct TfpwsWorkerWorkspacePlan
{
    bool has_work = false;
    std::size_t group_count = 0;
    std::size_t max_host_workspace_bytes = 0;
    TfpwsDeviceWorkspacePlan device = {};
};

struct TfpwsPrescan
{
    std::size_t total_groups = 0;
    unsigned nsamples = 0;
    std::vector<TfpwsWorkerWorkspacePlan> workers;
};

int estimate_tfpws_group_shape(const std::vector<SourcePackRecord> &records,
                               int sub_stack_size,
                               unsigned *ngroups,
                               unsigned *nsamples);

std::size_t estimate_tfpws_host_workspace_bytes(unsigned ngroups,
                                                unsigned nsamples);

int prescan_tfpws_inputs(const std::vector<std::string> &indexes,
                         const ARGUTYPE *argument,
                         const std::vector<GpuWorkerConfig> &gpu_configs,
                         TfpwsPrescan *prescan);

#endif
