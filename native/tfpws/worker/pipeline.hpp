#ifndef TFPWS_PIPELINE_HPP
#define TFPWS_PIPELINE_HPP

#include <vector>

#include "tfpws_types.hpp"

extern "C"
{
#include "arguproc.h"
}

int run_tfpws_pipeline(const ARGUTYPE *argument,
                       const std::vector<GpuWorkerConfig> &gpu_configs);

#endif
