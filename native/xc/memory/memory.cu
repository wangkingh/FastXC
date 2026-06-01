#include "memory.hpp"

#include "cuda.util.cuh"
#include "cuda.kernels.cuh"
#include "logger.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <sys/sysinfo.h>

static const double kHostRamAutoFraction = 0.80;

static bool kib_to_bytes(unsigned long long kib, size_t *out)
{
  const unsigned long long max_size = (unsigned long long)std::numeric_limits<size_t>::max();
  if (kib > max_size / 1024ULL)
  {
    *out = std::numeric_limits<size_t>::max();
    return true;
  }
  *out = (size_t)(kib * 1024ULL);
  return true;
}

static bool query_available_host_memory(size_t *available_bytes)
{
  FILE *fp = fopen("/proc/meminfo", "r");
  char line[256];

  if (fp)
  {
    while (fgets(line, sizeof(line), fp))
    {
      unsigned long long available_kib = 0;
      if (sscanf(line, "MemAvailable: %llu kB", &available_kib) == 1)
      {
        fclose(fp);
        return kib_to_bytes(available_kib, available_bytes);
      }
    }
    fclose(fp);
  }

  struct sysinfo info;
  if (sysinfo(&info) == 0 && info.mem_unit > 0)
  {
    const unsigned long long free_units = (unsigned long long)info.freeram;
    const unsigned long long unit = (unsigned long long)info.mem_unit;
    const unsigned long long max_size = (unsigned long long)std::numeric_limits<size_t>::max();
    if (free_units > max_size / unit)
      *available_bytes = std::numeric_limits<size_t>::max();
    else
      *available_bytes = (size_t)(free_units * unit);
    return true;
  }

  *available_bytes = 0;
  return false;
}

static bool compute_host_worker_budget(size_t worker_count,
                                       size_t *available_bytes,
                                       size_t *auto_budget_bytes,
                                       size_t *worker_budget_bytes)
{
  if (!query_available_host_memory(available_bytes))
  {
    *auto_budget_bytes = 0;
    *worker_budget_bytes = 0;
    return false;
  }
  if (worker_count == 0)
    worker_count = 1;
  *auto_budget_bytes = (size_t)((long double)(*available_bytes) *
                                (long double)kHostRamAutoFraction);
  *worker_budget_bytes = *auto_budget_bytes / worker_count;
  return true;
}

static size_t cufft_workspace_required(size_t pair_capacity, int nfft, int nspec)
{
  size_t work = 0;
  int rank = 1;
  int n[1] = {nfft};
  int inembed[1] = {nspec};
  int onembed[1] = {nfft};
  cufftHandle plan = 0;
  if (pair_capacity == 0 || pair_capacity > (size_t)INT_MAX)
    return std::numeric_limits<size_t>::max();
  cufftResult rc = cufftCreate(&plan);
  if (rc != CUFFT_SUCCESS)
    return std::numeric_limits<size_t>::max();
  rc = cufftSetAutoAllocation(plan, 0);
  if (rc != CUFFT_SUCCESS)
  {
    cufftDestroy(plan);
    return std::numeric_limits<size_t>::max();
  }
  rc = cufftMakePlanMany(plan, rank, n,
                         inembed, 1, nspec,
                         onembed, 1, nfft,
                         CUFFT_C2R, (int)pair_capacity, &work);
  cufftDestroy(plan);
  if (rc != CUFFT_SUCCESS)
    return std::numeric_limits<size_t>::max();
  return work;
}

bool compute_memory_plan(size_t block_files,
                         const RuntimeShape *shape,
                         size_t lazy_write_depth,
                         MemoryPlan *plan)
{
  MemoryPlan p;
  size_t block_pair_count = 0;
  size_t tmp = 0;
  size_t device_core = 0;
  if (block_files == 0 || shape->step_bytes == 0 || shape->vec_count == 0 || shape->vec_bytes == 0 ||
      shape->nspec <= 0 || shape->nfft <= 0 || shape->cc_size <= 0)
    return false;
  if (!checked_mul_size(block_files, block_files, &block_pair_count))
    return false;
  if (!checked_mul_size(block_pair_count, kRowTargetBlocks, &p.pair_capacity))
    return false;
  if (p.pair_capacity == 0 || p.pair_capacity > (size_t)INT_MAX)
    return false;
  p.block_file_count = block_files;
  if (!checked_mul_size(block_files, 1 + kRowTargetBlocks, &p.max_loaded_files))
    return false;

  if (!checked_mul_size(p.max_loaded_files, shape->step_bytes, &p.host_step_input_bytes))
    return false;
  if (!checked_mul_size(p.pair_capacity, (size_t)shape->cc_size, &tmp) ||
      !checked_mul_size(tmp, sizeof(float), &p.host_cc_bytes))
    return false;
  if (!checked_mul_size(2 * sizeof(size_t), p.pair_capacity, &p.host_index_bytes))
    return false;
  if (!checked_mul_size(p.pair_capacity, sizeof(XcTask), &tmp) ||
      !checked_add_to(&tmp, p.host_cc_bytes) ||
      !checked_mul_size(lazy_write_depth, tmp, &p.host_lazy_bytes))
    return false;
  p.host_active_bytes = p.host_step_input_bytes;
  if (!checked_add_to(&p.host_active_bytes, p.host_cc_bytes) ||
      !checked_add_to(&p.host_active_bytes, p.host_index_bytes) ||
      !checked_add_to(&p.host_active_bytes, p.host_lazy_bytes))
    return false;

  if (!checked_mul_size(p.max_loaded_files, shape->step_bytes, &p.device_step_input_bytes))
    return false;
  p.device_index_bytes = p.host_index_bytes;
  if (!checked_mul_size(p.pair_capacity, (size_t)shape->nspec, &tmp) ||
      !checked_mul_size(tmp, sizeof(cuComplex), &p.device_stack_bytes))
    return false;
  if (!checked_mul_size(p.pair_capacity, (size_t)shape->nfft, &tmp) ||
      !checked_mul_size(tmp, sizeof(float), &p.device_time_bytes))
    return false;
  if (!checked_mul_size((size_t)shape->nspec, sizeof(int), &p.device_sign_bytes))
    return false;

  p.cufft_workspace_bytes = cufft_workspace_required(p.pair_capacity, shape->nfft, shape->nspec);
  if (p.cufft_workspace_bytes == std::numeric_limits<size_t>::max())
    return false;

  device_core = p.device_step_input_bytes;
  if (!checked_add_to(&device_core, p.device_index_bytes) ||
      !checked_add_to(&device_core, p.device_stack_bytes) ||
      !checked_add_to(&device_core, p.device_time_bytes) ||
      !checked_add_to(&device_core, p.device_sign_bytes) ||
      !checked_add_to(&device_core, p.cufft_workspace_bytes))
    return false;
  p.device_safety_bytes = std::max((size_t)128 * kBytesPerMiB, device_core / 20);
  p.device_total_bytes = device_core;
  if (!checked_add_to(&p.device_total_bytes, p.device_safety_bytes))
    return false;

  *plan = p;
  return true;
}

static bool fits_block(size_t block_files,
                       const RuntimeShape *shape,
                       size_t lazy_write_depth,
                       size_t usable_vram,
                       bool enforce_host_budget,
                       size_t usable_host_ram,
                       MemoryPlan *out_plan)
{
  MemoryPlan plan;
  if (!compute_memory_plan(block_files, shape, lazy_write_depth, &plan))
    return false;
  if (out_plan)
    *out_plan = plan;
  if (plan.device_total_bytes > usable_vram)
    return false;
  if (enforce_host_budget && plan.host_active_bytes > usable_host_ram)
    return false;
  return true;
}

bool compute_pair_capacity_for_block(size_t block_files, size_t *pair_capacity)
{
  size_t block_pair_count = 0;
  if (!checked_mul_size(block_files, block_files, &block_pair_count))
    return false;
  return checked_mul_size(block_pair_count, kRowTargetBlocks, pair_capacity);
}

static int report_cuda_failure(cudaError_t rc, const char *event, size_t worker)
{
  if (rc == cudaSuccess)
    return 0;
  LOG_WARN(event, "worker=%zu cuda_status=\"%s\"", worker, cudaGetErrorString(rc));
  return -1;
}

static int report_cufft_failure(cufftResult rc, const char *event, size_t worker)
{
  if (rc == CUFFT_SUCCESS)
    return 0;
  LOG_WARN(event, "worker=%zu cufft_status=%d", worker, (int)rc);
  return -1;
}

static int try_gpu_malloc(void **ptr, size_t bytes, size_t worker, const char *name)
{
  cudaError_t rc = cudaMalloc(ptr, bytes);
  if (rc != cudaSuccess)
  {
    LOG_WARN("gpu_buffer_alloc_failed",
             "worker=%zu buffer=%s bytes_mib=%.3f cuda_status=\"%s\"",
             worker, name, bytes_to_mib(bytes), cudaGetErrorString(rc));
    *ptr = NULL;
    return -1;
  }
  return 0;
}

static int try_gpu_calloc(void **ptr, size_t bytes, size_t worker, const char *name)
{
  if (try_gpu_malloc(ptr, bytes, worker, name) != 0)
    return -1;
  cudaError_t rc = cudaMemset(*ptr, 0, bytes);
  if (rc != cudaSuccess)
  {
    LOG_WARN("gpu_buffer_clear_failed",
             "worker=%zu buffer=%s bytes_mib=%.3f cuda_status=\"%s\"",
             worker, name, bytes_to_mib(bytes), cudaGetErrorString(rc));
    return -1;
  }
  return 0;
}

static size_t count_workers_on_device(const ARGUTYPE *args, size_t gpu_id)
{
  size_t count = 0;
  for (size_t i = 0; i < args->gpu_count; ++i)
    if (args->gpu_ids[i] == gpu_id)
      ++count;
  return count > 0 ? count : 1;
}

static size_t gpu_memory_limit_bytes(double limit_mib)
{
  const double max_bytes = (double)std::numeric_limits<size_t>::max();
  double limit_bytes = 0.0;
  if (limit_mib <= 0.0)
    return 0;
  limit_bytes = limit_mib * (double)kBytesPerMiB;
  if (limit_bytes >= max_bytes)
    return std::numeric_limits<size_t>::max();
  return (size_t)limit_bytes;
}

size_t estimate_block_files_for_worker(const ARGUTYPE *args,
                                       size_t worker_index,
                                       const RuntimeShape *shape)
{
  size_t gpu_id = 0;
  size_t free_ram = 0, total_ram = 0;
  size_t auto_available = 0, usable = 0;
  size_t runtime_limit = 0;
  size_t host_available = 0;
  size_t host_auto_budget = 0;
  size_t host_worker_budget = 0;
  size_t physical_worker_count = 1;
  bool runtime_limit_applied = false;
  bool host_budget_active = false;
  MemoryPlan plan;
  if (worker_index >= args->gpu_count)
  {
    LOG_ERROR("gpu_worker_index_invalid", "worker=%zu gpu_workers=%zu",
              worker_index, args->gpu_count);
    exit(1);
  }
  gpu_id = args->gpu_ids[worker_index];
  CUDACHECK(cudaSetDevice((int)gpu_id));
  CUDACHECK(cudaMemGetInfo(&free_ram, &total_ram));

  physical_worker_count = count_workers_on_device(args, gpu_id);
  auto_available = (size_t)(((double)free_ram * 0.90) /
                            (double)physical_worker_count);
  usable = auto_available;
  runtime_limit = gpu_memory_limit_bytes(args->gpu_mem_limit_mib[worker_index]);
  if (runtime_limit > 0 && usable > runtime_limit)
  {
    usable = runtime_limit;
    runtime_limit_applied = true;
  }
  host_budget_active = compute_host_worker_budget(args->gpu_count,
                                                  &host_available,
                                                  &host_auto_budget,
                                                  &host_worker_budget);
  if (!host_budget_active)
  {
    LOG_WARN("host_memory_query_failed",
             "worker=%zu source=\"/proc/meminfo,sysinfo\" host_budget_active=no",
             worker_index);
  }

  size_t lo = 1, hi = 2;
  while (hi < 4096 && fits_block(hi, shape, args->lazy_write_depth, usable,
                                 host_budget_active, host_worker_budget, NULL))
    hi *= 2;
  while (lo + 1 < hi)
  {
    size_t mid = lo + (hi - lo) / 2;
    if (fits_block(mid, shape, args->lazy_write_depth, usable,
                   host_budget_active, host_worker_budget, NULL))
      lo = mid;
    else
      hi = mid;
  }
  if (shape->num_channels > 1)
    lo = (lo / shape->num_channels) * shape->num_channels;
  if (!fits_block(lo, shape, args->lazy_write_depth, usable,
                  host_budget_active, host_worker_budget, &plan))
  {
    LOG_ERROR("worker_budget_too_small",
              "worker=%zu physical_gpu_id=%zu num_ch=%zu final_worker_gpu_budget_gib=%.3f final_worker_host_budget_gib=%.3f host_budget_active=%s",
              worker_index, gpu_id, shape->num_channels, bytes_to_gib(usable),
              bytes_to_gib(host_worker_budget),
              host_budget_active ? "yes" : "no");
    exit(1);
  }
  LOG_INFO("worker_memory_budget",
           "worker=%zu physical_gpu_id=%zu physical_worker_count=%zu free_gpu_memory_gib=%.3f auto_budget_gib=%.3f manual_budget_from_M_gib=%.3f final_worker_budget_gib=%.3f available_host_memory_gib=%.3f auto_host_budget_gib=%.3f final_worker_host_budget_gib=%.3f selected_block_files=%zu pair_capacity=%zu model=row_batch num_ch=%zu row_target_blocks=%zu manual_applied=%s host_budget_active=%s",
           worker_index, gpu_id, physical_worker_count,
           bytes_to_gib(free_ram), bytes_to_gib(auto_available),
           bytes_to_gib(runtime_limit), bytes_to_gib(usable),
           bytes_to_gib(host_available), bytes_to_gib(host_auto_budget),
           bytes_to_gib(host_worker_budget),
           lo, plan.pair_capacity, shape->num_channels, kRowTargetBlocks,
           runtime_limit_applied ? "yes" : "no",
           host_budget_active ? "yes" : "no");
  LOG_INFO("device_memory_plan",
           "worker=%zu physical_gpu_id=%zu input_step_mib=%.3f stack_mib=%.3f time_mib=%.3f index_mib=%.3f sign_mib=%.3f cufft_workspace_mib=%.3f safety_mib=%.3f total_gib=%.3f",
           worker_index, gpu_id,
           bytes_to_mib(plan.device_step_input_bytes),
           bytes_to_mib(plan.device_stack_bytes),
           bytes_to_mib(plan.device_time_bytes),
           bytes_to_mib(plan.device_index_bytes),
           bytes_to_mib(plan.device_sign_bytes),
           bytes_to_mib(plan.cufft_workspace_bytes),
           bytes_to_mib(plan.device_safety_bytes),
           bytes_to_gib(plan.device_total_bytes));
  LOG_INFO("host_worker_memory_plan",
           "worker=%zu physical_gpu_id=%zu input_step_mib=%.3f cc_mib=%.3f index_mib=%.3f lazy_mib=%.3f total_gib=%.3f budget_gib=%.3f budget_active=%s",
           worker_index, gpu_id,
           bytes_to_mib(plan.host_step_input_bytes),
           bytes_to_mib(plan.host_cc_bytes),
           bytes_to_mib(plan.host_index_bytes),
           bytes_to_mib(plan.host_lazy_bytes),
           bytes_to_gib(plan.host_active_bytes),
           bytes_to_gib(host_worker_budget),
           host_budget_active ? "yes" : "no");
  return std::max((size_t)1, lo);
}

int init_gpu_buffers(GpuBuffers *buf, const WorkerConfig *cfg, const RuntimeShape *shape)
{
  MemoryPlan plan;
  size_t free_before = 0, total_before = 0, free_after = 0, total_after = 0;
  size_t actual_device_total = 0;
  size_t index_one_bytes = 0;
  int rank = 1;
  int n[1] = {shape->nfft};
  int inembed[1] = {shape->nspec};
  int onembed[1] = {shape->nfft};

  if (!compute_memory_plan(cfg->block_file_count, shape, cfg->lazy_write_depth, &plan) ||
      plan.pair_capacity != cfg->pair_capacity)
  {
    LOG_ERROR("memory_plan_invalid",
              "worker=%zu block_files=%zu pair_capacity=%zu",
              cfg->worker_id, cfg->block_file_count, cfg->pair_capacity);
    return -1;
  }
  if (!checked_mul_size(plan.pair_capacity, sizeof(size_t), &index_one_bytes))
  {
    LOG_ERROR("index_buffer_overflow", "worker=%zu pair_capacity=%zu",
              cfg->worker_id, plan.pair_capacity);
    return -1;
  }

  if (report_cuda_failure(cudaSetDevice((int)cfg->gpu_id),
                          "gpu_set_device_failed", cfg->worker_id) != 0 ||
      report_cuda_failure(cudaMemGetInfo(&free_before, &total_before),
                          "gpu_mem_info_failed", cfg->worker_id) != 0)
  {
    return -1;
  }
  buf->h_spec = (complex *)malloc(plan.host_step_input_bytes);
  buf->h_cc = (float *)malloc(plan.host_cc_bytes);
  buf->h_src_idx = (size_t *)malloc(index_one_bytes);
  buf->h_rec_idx = (size_t *)malloc(index_one_bytes);
  if (!buf->h_spec || !buf->h_cc || !buf->h_src_idx || !buf->h_rec_idx)
  {
    LOG_ERROR("host_buffer_alloc_failed", "worker=%zu host_active_gib=%.3f",
              cfg->worker_id, bytes_to_gib(plan.host_active_bytes));
    free_gpu_buffers(buf);
    return -1;
  }

  if (try_gpu_malloc((void **)&buf->d_spec, plan.device_step_input_bytes,
                     cfg->worker_id, "d_spec") != 0 ||
      try_gpu_malloc((void **)&buf->d_src_idx, index_one_bytes,
                     cfg->worker_id, "d_src_idx") != 0 ||
      try_gpu_malloc((void **)&buf->d_rec_idx, index_one_bytes,
                     cfg->worker_id, "d_rec_idx") != 0 ||
      try_gpu_malloc((void **)&buf->d_stack, plan.device_stack_bytes,
                     cfg->worker_id, "d_stack") != 0 ||
      try_gpu_malloc((void **)&buf->d_time, plan.device_time_bytes,
                     cfg->worker_id, "d_time") != 0 ||
      try_gpu_calloc((void **)&buf->d_sign, (size_t)shape->nspec * sizeof(int),
                     cfg->worker_id, "d_sign") != 0)
  {
    free_gpu_buffers(buf);
    return -1;
  }
  buf->pair_capacity = plan.pair_capacity;

  dim3 grid, block;
  DimCompute1D(&grid, &block, (size_t)shape->nspec);
  generateSignVector<<<grid, block>>>(buf->d_sign, (size_t)shape->nspec);
  if (report_cuda_failure(cudaGetLastError(), "gpu_sign_kernel_failed",
                          cfg->worker_id) != 0)
  {
    free_gpu_buffers(buf);
    return -1;
  }

  if (report_cufft_failure(cufftCreate(&buf->plan), "cufft_create_failed",
                           cfg->worker_id) != 0 ||
      report_cufft_failure(cufftSetAutoAllocation(buf->plan, 0),
                           "cufft_auto_alloc_disable_failed", cfg->worker_id) != 0 ||
      report_cufft_failure(cufftMakePlanMany(buf->plan, rank, n,
                                             inembed, 1, shape->nspec,
                                             onembed, 1, shape->nfft,
                                             CUFFT_C2R, (int)plan.pair_capacity,
                                             &buf->cufft_work_bytes),
                           "cufft_plan_failed", cfg->worker_id) != 0)
  {
    free_gpu_buffers(buf);
    return -1;
  }
  if (buf->cufft_work_bytes > 0)
  {
    if (try_gpu_malloc(&buf->d_cufft_work, buf->cufft_work_bytes,
                       cfg->worker_id, "d_cufft_work") != 0 ||
        report_cufft_failure(cufftSetWorkArea(buf->plan, buf->d_cufft_work),
                             "cufft_set_work_area_failed", cfg->worker_id) != 0)
    {
      free_gpu_buffers(buf);
      return -1;
    }
  }
  actual_device_total = plan.device_step_input_bytes;
  if (!checked_add_to(&actual_device_total, plan.device_index_bytes) ||
      !checked_add_to(&actual_device_total, plan.device_stack_bytes) ||
      !checked_add_to(&actual_device_total, plan.device_time_bytes) ||
      !checked_add_to(&actual_device_total, plan.device_sign_bytes) ||
      !checked_add_to(&actual_device_total, buf->cufft_work_bytes))
  {
    LOG_ERROR("device_allocation_accounting_overflow", "worker=%zu", cfg->worker_id);
    free_gpu_buffers(buf);
    return -1;
  }
  if (report_cuda_failure(cudaMemGetInfo(&free_after, &total_after),
                          "gpu_mem_info_failed", cfg->worker_id) != 0)
  {
    free_gpu_buffers(buf);
    return -1;
  }
  LOG_INFO("gpu_allocation",
           "worker=%zu gpu=%zu host_active_gib=%.3f device_core_gib=%.3f cufft_plan_mib=%.3f free_before_gib=%.3f free_after_gib=%.3f",
           cfg->worker_id, cfg->gpu_id,
           bytes_to_gib(plan.host_active_bytes),
           bytes_to_gib(actual_device_total),
           bytes_to_mib(buf->cufft_work_bytes),
           bytes_to_gib(free_before),
           bytes_to_gib(free_after));
  if (buf->cufft_work_bytes > plan.cufft_workspace_bytes)
  {
    LOG_WARN("cufft_workspace_exceeded",
             "worker=%zu actual_mib=%.3f planned_mib=%.3f",
             cfg->worker_id,
             bytes_to_mib(buf->cufft_work_bytes),
             bytes_to_mib(plan.cufft_workspace_bytes));
  }
  return 0;
}

void free_gpu_buffers(GpuBuffers *buf)
{
  free(buf->h_spec);
  buf->h_spec = NULL;
  free(buf->h_cc);
  buf->h_cc = NULL;
  free(buf->h_src_idx);
  buf->h_src_idx = NULL;
  free(buf->h_rec_idx);
  buf->h_rec_idx = NULL;
  if (buf->d_spec)
    GpuFree((void **)&buf->d_spec);
  if (buf->d_src_idx)
    GpuFree((void **)&buf->d_src_idx);
  if (buf->d_rec_idx)
    GpuFree((void **)&buf->d_rec_idx);
  if (buf->d_stack)
    GpuFree((void **)&buf->d_stack);
  if (buf->d_time)
    GpuFree((void **)&buf->d_time);
  if (buf->d_sign)
    GpuFree((void **)&buf->d_sign);
  if (buf->d_cufft_work)
    GpuFree(&buf->d_cufft_work);
  if (buf->plan)
  {
    cufftDestroy(buf->plan);
    buf->plan = 0;
  }
}
