#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include "runtime.hpp"

#include <vector>

bool queue_pop(JobQueue *queue, RowBatchJob *job);

std::vector<RowBatchJob> build_jobs(size_t file_count, size_t block_size);

std::vector<XcTask> build_tasks_for_job(const TimestampWork *work,
                                        const AllowedPathTable *paths,
                                        const RowBatchJob *job,
                                        std::vector<size_t> *loaded_meta_indices);

size_t count_units_for_jobs(const TimestampWork *work,
                            const AllowedPathTable *paths,
                            const std::vector<RowBatchJob> &jobs);

#endif
