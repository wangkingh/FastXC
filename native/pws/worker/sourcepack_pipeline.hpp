#ifndef PWS_SOURCEPACK_PIPELINE_HPP
#define PWS_SOURCEPACK_PIPELINE_HPP

#include "gpu_config.hpp"

extern "C"
{
#include "arguproc.h"
}

int run_pws_sourcepack_pipeline(const PwsSourcePackArgs &args,
                                const PwsGpuWorkerConfig &gpu_config);

#endif
