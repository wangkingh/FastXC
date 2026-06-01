#ifndef INPUT_HPP
#define INPUT_HPP

#include "runtime.hpp"

#include <string>
#include <vector>

extern "C"
{
#include "arguproc.h"
}

std::vector<TimestampInput> load_timestamp_inputs(const ARGUTYPE *args);

int open_timestamp_work(const TimestampInput &input,
                        TimestampWork *work,
                        const RuntimeShape *expected,
                        RuntimeShape *shape);

void close_timestamp_work(TimestampWork *work);

bool try_cache_timestamp_payload(TimestampWork *work);

bool finalize_shape(RuntimeShape *shape, float cc_length);

void load_job_step_input(const TimestampWork *work,
                         const RuntimeShape *shape,
                         const std::vector<size_t> &meta_indices,
                         size_t step_idx,
                         complex *dst);

#endif
