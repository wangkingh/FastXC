#include "scheduler.hpp"

#include <algorithm>
#include <pthread.h>
#include <vector>

extern "C"
{
#include "path_table.h"
}

bool queue_pop(JobQueue *queue, RowBatchJob *job)
{
  bool ok = false;
  pthread_mutex_lock(&queue->mutex);
  if (queue->next < queue->jobs.size())
  {
    *job = queue->jobs[queue->next++];
    ok = true;
  }
  pthread_mutex_unlock(&queue->mutex);
  return ok;
}

static void fill_row_job(RowBatchJob *job,
                         size_t file_count,
                         size_t block_size,
                         size_t anchor_block,
                         size_t target_begin_block,
                         size_t target_end_block)
{
  job->anchor_block = anchor_block;
  job->target_begin_block = target_begin_block;
  job->target_end_block = target_end_block;
  job->block_size = block_size;
  job->file_count = file_count;
  job->anchor_begin = anchor_block * block_size;
  job->anchor_end = std::min(file_count, job->anchor_begin + block_size);
  job->target_begin = target_begin_block * block_size;
  job->target_end = std::min(file_count, target_end_block * block_size);
}

std::vector<RowBatchJob> build_jobs(size_t file_count, size_t block_size)
{
  std::vector<RowBatchJob> jobs;
  size_t block_count = (file_count + block_size - 1) / block_size;

  for (size_t anchor = 0; anchor < block_count; ++anchor)
  {
    RowBatchJob self_job;
    fill_row_job(&self_job, file_count, block_size, anchor, anchor, anchor + 1);
    jobs.push_back(self_job);

    for (size_t target = anchor + 1; target < block_count; target += kRowTargetBlocks)
    {
      RowBatchJob cross_job;
      size_t target_end = std::min(block_count, target + kRowTargetBlocks);
      fill_row_job(&cross_job, file_count, block_size, anchor, target, target_end);
      jobs.push_back(cross_job);
    }
  }
  return jobs;
}

std::vector<XcTask> build_tasks_for_job(const TimestampWork *work,
                                        const AllowedPathTable *paths,
                                        const RowBatchJob *job,
                                        std::vector<size_t> *loaded_meta_indices)
{
  std::vector<XcTask> tasks;
  loaded_meta_indices->clear();

  for (size_t i = job->anchor_begin; i < job->anchor_end; ++i)
    loaded_meta_indices->push_back(i);
  if (job->target_begin != job->anchor_begin)
  {
    for (size_t i = job->target_begin; i < job->target_end; ++i)
      loaded_meta_indices->push_back(i);
  }

  auto local_idx = [&](size_t meta_idx) -> size_t
  {
    if (meta_idx >= job->anchor_begin && meta_idx < job->anchor_end)
      return meta_idx - job->anchor_begin;
    return (job->anchor_end - job->anchor_begin) + (meta_idx - job->target_begin);
  };

  auto push_task = [&](size_t src_idx, size_t rec_idx,
                       const AllowedPathRecord *record,
                       bool is_autocorr)
  {
    XcTask task;
    task.src_meta_idx = src_idx;
    task.rec_meta_idx = rec_idx;
    task.src_local_idx = local_idx(src_idx);
    task.rec_local_idx = local_idx(rec_idx);
    task.path_id = record->path_id;
    task.path_record = *record;
    task.is_autocorr = is_autocorr;
    tasks.push_back(task);
  };

  auto try_pair = [&](size_t a, size_t b, bool same_block)
  {
    const SpecMeta &ma = work->specs[a];
    const SpecMeta &mb = work->specs[b];
    const bool is_autocorr = (ma.gnsl_id == mb.gnsl_id);

    if (same_block && !is_autocorr && b <= a)
      return;

    const AllowedPathRecord *record =
        find_allowed_path_canonical(paths, ma.gnsl_id, mb.gnsl_id);
    if (!record)
      return;

    if (is_autocorr)
    {
      push_task(a, b, record, true);
      if (!same_block)
        push_task(b, a, record, true);
      return;
    }

    if (ma.gnsl_id < mb.gnsl_id)
      push_task(a, b, record, false);
    else
      push_task(b, a, record, false);
  };

  for (size_t target_block = job->target_begin_block;
       target_block < job->target_end_block;
       ++target_block)
  {
    size_t begin = target_block * job->block_size;
    size_t end = std::min(job->file_count, begin + job->block_size);
    const bool same_block = (target_block == job->anchor_block);

    if (same_block)
    {
      begin = job->anchor_begin;
      end = job->anchor_end;
    }

    for (size_t a = job->anchor_begin; a < job->anchor_end; ++a)
    {
      for (size_t b = begin; b < end; ++b)
        try_pair(a, b, same_block);
    }
  }
  return tasks;
}

size_t count_units_for_jobs(const TimestampWork *work,
                            const AllowedPathTable *paths,
                            const std::vector<RowBatchJob> &jobs)
{
  size_t total = 0;
  std::vector<size_t> loaded_indices;
  for (size_t i = 0; i < jobs.size(); ++i)
  {
    std::vector<XcTask> tasks =
        build_tasks_for_job(work, paths, &jobs[i], &loaded_indices);
    total += tasks.size();
  }
  return total;
}
