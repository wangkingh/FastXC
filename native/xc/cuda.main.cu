#include "async_workers.hpp"
#include "fs.h"
#include "input.hpp"
#include "logger.h"
#include "memory.hpp"
#include "pack_writer.hpp"
#include "runtime.hpp"
#include "scheduler.hpp"
#include "progress_sidecar.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <string>
#include <vector>

extern "C"
{
#include "arguproc.h"
#include "include/write_mode.h"
#include "path_table.h"
}

struct XcRunState
{
  ARGUTYPE args;
  AllowedPathTable allowed_paths;
  std::vector<TimestampInput> timestamp_inputs;
  RuntimeShape shape;
  size_t global_block_size = 0;
  size_t global_pair_capacity = 0;
  std::vector<WorkerConfig> worker_cfgs;
  FastxcProgressSidecar progress;
  ResidentWorkerPool resident_pool;
  std::vector<ResidentWorkerContext> resident_contexts;
  std::vector<pthread_t> resident_threads;
  size_t resident_thread_count = 0;
  size_t resident_active_workers = 0;
  bool allowed_paths_ready = false;
  bool progress_started = false;
  bool resident_pool_ready = false;
};

struct TimestampRunState
{
  TimestampWork work;
  JobQueue queue;
  size_t timestamp_units = 0;
  bool queue_ready = false;
};

static const char *write_mode_name(int write_mode)
{
  switch (write_mode)
  {
  case MODE_APPEND:
    return "append";
  case MODE_AGGREGATE:
    return "aggregate";
  case MODE_PACK:
    return "pack";
  default:
    return "unknown";
  }
}

static void stop_resident_workers(XcRunState *run)
{
  if (!run || !run->resident_pool_ready)
    return;
  if (run->resident_thread_count > 0)
    resident_worker_pool_stop(&run->resident_pool);
  for (size_t i = 0; i < run->resident_thread_count; ++i)
    pthread_join(run->resident_threads[i], NULL);
  resident_worker_pool_destroy(&run->resident_pool);
  run->resident_contexts.clear();
  run->resident_threads.clear();
  run->resident_thread_count = 0;
  run->resident_active_workers = 0;
  run->resident_pool_ready = false;
}

static void cleanup_run_state(XcRunState *run)
{
  stop_resident_workers(run);
  if (run && run->allowed_paths_ready)
  {
    free_allowed_path_table(&run->allowed_paths);
    run->allowed_paths_ready = false;
  }
}

static int prepare_paths_and_output(XcRunState *run)
{
  init_allowed_path_table(&run->allowed_paths);
  run->allowed_paths_ready = true;
  if (load_allowed_path_table(run->args.allowed_paths_file, &run->allowed_paths) != 0)
    return -1;
  if (run->allowed_paths.count == 0)
  {
    LOG_ERROR("allowed_paths_empty", "path=\"%s\"", run->args.allowed_paths_file);
    return -1;
  }
  if (mkdir_p(run->args.ncf_dir, 0755) != 0)
  {
    LOG_ERROR("output_dir_create_failed", "path=\"%s\" error=\"%s\"",
              run->args.ncf_dir, strerror(errno));
    return -1;
  }
  if (run->args.write_mode == MODE_PACK && xc_pack_prepare_root(run->args.ncf_dir) != 0)
    return -1;
  return 0;
}

static int start_resident_workers(XcRunState *run)
{
  resident_worker_pool_init(&run->resident_pool, run->args.gpu_count);
  run->resident_pool_ready = true;
  run->resident_contexts.resize(run->args.gpu_count);
  run->resident_threads.resize(run->args.gpu_count);
  run->resident_thread_count = 0;

  for (size_t i = 0; i < run->args.gpu_count; ++i)
  {
    run->resident_contexts[i].cfg = run->worker_cfgs[i];
    run->resident_contexts[i].shape = &run->shape;
    run->resident_contexts[i].paths = &run->allowed_paths;
    run->resident_contexts[i].pool = &run->resident_pool;
    if (pthread_create(&run->resident_threads[i], NULL,
                       resident_gpu_worker_main,
                       &run->resident_contexts[i]) != 0)
    {
      LOG_ERROR("resident_worker_thread_create_failed",
                "worker=%zu gpu=%zu", i, run->worker_cfgs[i].gpu_id);
      stop_resident_workers(run);
      return -1;
    }
    ++run->resident_thread_count;
  }

  LOG_INFO("resident_workers_wait_ready", "requested_workers=%zu timeout_seconds=%d",
           run->args.gpu_count, kResidentStartupWaitSeconds);
  resident_worker_pool_wait_ready_or_timeout(&run->resident_pool,
                                             kResidentStartupWaitSeconds);
  run->resident_active_workers =
      resident_worker_pool_select_ready(&run->resident_pool,
                                        run->resident_contexts.data(),
                                        run->resident_contexts.size());
  if (run->resident_active_workers == 0)
  {
    LOG_ERROR("resident_no_workers_ready",
              "requested_workers=%zu timeout_seconds=%d",
              run->args.gpu_count, kResidentStartupWaitSeconds);
    stop_resident_workers(run);
    return -1;
  }
  LOG_INFO("resident_pool_start",
           "requested_workers=%zu ready_workers=%zu timeout_seconds=%d",
           run->args.gpu_count, run->resident_active_workers,
           kResidentStartupWaitSeconds);
  return 0;
}

static int load_inputs_and_shape(XcRunState *run)
{
  TimestampWork first;
  RuntimeShape seed_shape;
  size_t files_first = 0;

  run->timestamp_inputs = load_timestamp_inputs(&run->args);
  if (run->timestamp_inputs.empty())
  {
    LOG_ERROR("timestamp_input_empty", "index=\"%s\" single=\"%s\"",
              run->args.timestamp_index_path ? run->args.timestamp_index_path : "",
              run->args.single_timestamp_path ? run->args.single_timestamp_path : "");
    return -1;
  }

  if (open_timestamp_work(run->timestamp_inputs[0], &first, &run->shape, &seed_shape) != 0)
    return -1;
  files_first = first.specs.size();
  if (run->timestamp_inputs.size() == 1 &&
      run->timestamp_inputs[0].file_count_hint == 0)
    run->timestamp_inputs[0].file_count_hint = files_first;
  run->shape = seed_shape;
  close_timestamp_work(&first);
  if (!finalize_shape(&run->shape, run->args.cclength))
  {
    LOG_ERROR("runtime_shape_invalid", "cclength=%g", run->args.cclength);
    return -1;
  }

  LOG_INFO("runtime_shape",
           "nspec=%d nstep=%d nfft=%d dt=%g df=%g num_ch=%zu cc_size=%d source_step_mib=%.3f full_spec_mib=%.3f files_first=%zu",
           run->shape.nspec, run->shape.nstep, run->shape.nfft,
           run->shape.dt, run->shape.df, run->shape.num_channels,
           run->shape.cc_size, bytes_to_mib(run->shape.step_bytes),
           bytes_to_mib(run->shape.vec_bytes), files_first);
  return 0;
}

static void start_progress(XcRunState *run)
{
  run->progress.init(run->args.progress_file);
  run->progress.set_rows({
      {"overall", "RUNNING", 0, run->timestamp_inputs.size(), "timestamps", ""},
      {"current", "PENDING", 0, 0, "xc_units", ""},
  });
  run->progress_started = true;
}

static int plan_gpu_workers(XcRunState *run)
{
  run->global_block_size = (size_t)-1;
  for (size_t i = 0; i < run->args.gpu_count; ++i)
  {
    size_t block_size = estimate_block_files_for_worker(&run->args, i, &run->shape);
    run->global_block_size = std::min(run->global_block_size, block_size);
  }
  if (run->global_block_size == (size_t)-1 || run->global_block_size == 0)
    run->global_block_size = 1;

  size_t hinted = 0;
  size_t max_file_count_hint = 0;
  for (size_t i = 0; i < run->timestamp_inputs.size(); ++i)
  {
    if (run->timestamp_inputs[i].file_count_hint > 0)
    {
      ++hinted;
      max_file_count_hint = std::max(max_file_count_hint,
                                     run->timestamp_inputs[i].file_count_hint);
    }
  }
  if (hinted == run->timestamp_inputs.size() && max_file_count_hint > 0 &&
      run->global_block_size > max_file_count_hint)
  {
    LOG_INFO("worker_plan_file_count_cap",
             "estimated_block_files=%zu max_timestamp_files=%zu source=file_count_hint",
             run->global_block_size, max_file_count_hint);
    run->global_block_size = max_file_count_hint;
  }

  if (!compute_pair_capacity_for_block(run->global_block_size,
                                       &run->global_pair_capacity))
  {
    LOG_ERROR("global_pair_capacity_overflow", "block_files=%zu",
              run->global_block_size);
    return -1;
  }

  const size_t writer_threads =
      std::max((size_t)1, run->args.cpu_count / std::max((size_t)1, run->args.gpu_count));
  const size_t effective_writer_threads =
      run->args.write_mode == MODE_PACK ? (size_t)1 : writer_threads;
  run->worker_cfgs.clear();
  run->worker_cfgs.reserve(run->args.gpu_count);
  for (size_t i = 0; i < run->args.gpu_count; ++i)
  {
    WorkerConfig cfg;
    cfg.worker_id = i;
    cfg.gpu_id = run->args.gpu_ids[i];
    cfg.block_file_count = run->global_block_size;
    cfg.pair_capacity = run->global_pair_capacity;
    cfg.writer_threads = effective_writer_threads;
    cfg.write_mode = run->args.write_mode;
    cfg.lazy_write_depth = run->args.lazy_write_depth;
    cfg.output_dir = run->args.ncf_dir;
    cfg.progress = &run->progress;
    run->worker_cfgs.push_back(cfg);
  }

  LOG_INFO("worker_plan",
           "global_block_files=%zu max_pair_capacity=%zu gpu_workers=%zu writer_threads_per_worker=%zu lazy_write_depth=%zu write_mode=%s",
           run->global_block_size, run->global_pair_capacity,
           run->args.gpu_count, effective_writer_threads, run->args.lazy_write_depth,
           write_mode_name(run->args.write_mode));
  return 0;
}

static void mark_timestamp_skipped(XcRunState *run, size_t ts)
{
  run->progress.update("current", "SKIPPED", 0, 0, "xc_units",
                       run->timestamp_inputs[ts].timestamp);
  run->progress.update("overall",
                       ts + 1 == run->timestamp_inputs.size() ? "DONE" : "RUNNING",
                       ts + 1, run->timestamp_inputs.size(), "timestamps",
                       run->timestamp_inputs[ts].timestamp);
}

static bool prepare_timestamp_run(XcRunState *run,
                                  size_t ts,
                                  TimestampRunState *timestamp)
{
  RuntimeShape check_shape = run->shape;
  if (open_timestamp_work(run->timestamp_inputs[ts], &timestamp->work,
                          &run->shape, &check_shape) != 0)
  {
    LOG_WARN("timestamp_open_failed", "timestamp=\"%s\" xcspec=\"%s\"",
             run->timestamp_inputs[ts].timestamp.c_str(),
             run->timestamp_inputs[ts].xcspec_path.c_str());
    mark_timestamp_skipped(run, ts);
    return false;
  }
  try_cache_timestamp_payload(&timestamp->work);

  size_t active_block_size = std::min(run->global_block_size,
                                      timestamp->work.specs.size());
  active_block_size = std::max((size_t)1, active_block_size);
  size_t active_pair_capacity = 0;
  if (!compute_pair_capacity_for_block(active_block_size, &active_pair_capacity))
  {
    LOG_ERROR("timestamp_pair_capacity_overflow",
              "timestamp=\"%s\" block_files=%zu",
              timestamp->work.timestamp.c_str(), active_block_size);
    close_timestamp_work(&timestamp->work);
    return false;
  }

  timestamp->queue.jobs = build_jobs(timestamp->work.specs.size(), active_block_size);
  timestamp->queue.next = 0;
  pthread_mutex_init(&timestamp->queue.mutex, NULL);
  timestamp->queue_ready = true;
  timestamp->timestamp_units =
      count_units_for_jobs(&timestamp->work, &run->allowed_paths, timestamp->queue.jobs);
  run->progress.update("current",
                       timestamp->timestamp_units == 0 ? "DONE" : "RUNNING",
                       0, timestamp->timestamp_units, "xc_units",
                       timestamp->work.timestamp);

  LOG_INFO("timestamp_plan",
           "timestamp=\"%s\" files=%zu block_files=%zu row_target_blocks=%zu pair_capacity=%zu row_jobs=%zu input=%s units=%zu",
           timestamp->work.timestamp.c_str(), timestamp->work.specs.size(),
           active_block_size, kRowTargetBlocks, active_pair_capacity,
           timestamp->queue.jobs.size(),
           timestamp->work.payload_cache_enabled ? "host_payload" : "pread",
           timestamp->timestamp_units);

  return true;
}

static void cleanup_timestamp_run(TimestampRunState *timestamp)
{
  if (timestamp->queue_ready)
  {
    pthread_mutex_destroy(&timestamp->queue.mutex);
    timestamp->queue_ready = false;
  }
  close_timestamp_work(&timestamp->work);
}

static void finish_timestamp_run(XcRunState *run,
                                 size_t ts,
                                 TimestampRunState *timestamp)
{
  cleanup_timestamp_run(timestamp);
  run->progress.update("current", "DONE",
                       timestamp->timestamp_units, timestamp->timestamp_units,
                       "xc_units", timestamp->work.timestamp);
  run->progress.update("overall",
                       ts + 1 == run->timestamp_inputs.size() ? "DONE" : "RUNNING",
                       ts + 1, run->timestamp_inputs.size(), "timestamps",
                       timestamp->work.timestamp);
}

int main(int argc, char **argv)
{
  XcRunState run;
  ArgumentProcess(argc, argv, &run.args);
  LOG_INFO("run_start", "gpu_workers=%zu output=\"%s\" progress=\"%s\" write_mode=%s",
           run.args.gpu_count, run.args.ncf_dir,
           run.args.progress_file ? run.args.progress_file : "",
           write_mode_name(run.args.write_mode));

  if (prepare_paths_and_output(&run) != 0 ||
      load_inputs_and_shape(&run) != 0)
  {
    cleanup_run_state(&run);
    return 1;
  }

  start_progress(&run);
  if (plan_gpu_workers(&run) != 0)
  {
    run.progress.finish("FAILED", false);
    cleanup_run_state(&run);
    return 1;
  }
  if (start_resident_workers(&run) != 0)
  {
    run.progress.finish("FAILED", false);
    cleanup_run_state(&run);
    return 1;
  }

  int exit_code = 0;
  for (size_t ts = 0; ts < run.timestamp_inputs.size(); ++ts)
  {
    TimestampRunState timestamp;
    if (!prepare_timestamp_run(&run, ts, &timestamp))
      continue;

    LOG_INFO("timestamp_workers_start", "timestamp=\"%s\" workers=%zu requested_workers=%zu row_jobs=%zu resident=yes",
             timestamp.work.timestamp.c_str(), run.resident_active_workers,
             run.args.gpu_count, timestamp.queue.jobs.size());

    resident_worker_pool_submit(&run.resident_pool, &timestamp.work, &timestamp.queue);
    if (!resident_worker_pool_wait_done(&run.resident_pool))
    {
      LOG_ERROR("resident_timestamp_failed", "timestamp=\"%s\"",
                timestamp.work.timestamp.c_str());
      cleanup_timestamp_run(&timestamp);
      exit_code = 1;
      break;
    }

    if (run.args.write_mode == MODE_PACK &&
        xc_pack_write_timestamp_done_for_output(run.args.ncf_dir,
                                                timestamp.work.timestamp.c_str()) != 0)
    {
      LOG_ERROR("xcpack_timestamp_done_failed", "timestamp=\"%s\" output=\"%s\"",
                timestamp.work.timestamp.c_str(), run.args.ncf_dir);
      cleanup_timestamp_run(&timestamp);
      exit_code = 1;
      break;
    }

    finish_timestamp_run(&run, ts, &timestamp);
    LOG_INFO("timestamp_done", "timestamp_index=%zu timestamp=\"%s\"",
             ts, timestamp.work.timestamp.c_str());
  }

  if (exit_code == 0 && run.args.write_mode == MODE_PACK &&
      xc_pack_write_success_for_output(run.args.ncf_dir) != 0)
  {
    LOG_ERROR("xcpack_success_marker_failed", "output=\"%s\"", run.args.ncf_dir);
    exit_code = 1;
  }

  if (exit_code == 0)
    run.progress.finish("DONE", true);
  else if (run.progress_started)
    run.progress.finish("FAILED", false);
  cleanup_run_state(&run);
  LOG_INFO("run_done", "timestamps=%zu", run.timestamp_inputs.size());
  return exit_code;
}
