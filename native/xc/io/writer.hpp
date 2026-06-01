#ifndef WRITER_HPP
#define WRITER_HPP

#include "runtime.hpp"

#include <vector>

int lazy_writer_init(LazyWriteQueue *queue,
                     const WorkerConfig *cfg,
                     const RuntimeShape *shape,
                     const TimestampWork *timestamp,
                     size_t pair_capacity);

void lazy_writer_close(LazyWriteQueue *queue);

void submit_write_results(LazyWriteQueue *queue,
                          const WorkerConfig *cfg,
                          const RuntimeShape *shape,
                          const TimestampWork *timestamp,
                          const RowBatchJob *job,
                          const std::vector<XcTask> &tasks,
                          const float *cc);

#endif
