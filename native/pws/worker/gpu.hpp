#ifndef GPU_HPP
#define GPU_HPP

#include <vector>

#include "concurrency.hpp"
#include "types.hpp"

int compute_pws_host_batch(PwsHostBatch *batch,
                           int gpu_id,
                           HostGroupBudget *host_budget,
                           std::vector<SACHEAD> *out_headers,
                           float **out_data);

#endif
