#ifndef RUNTIME_HPP
#define RUNTIME_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <pthread.h>
#include <string>
#include <vector>

#include <cuComplex.h>
#include <cufft.h>

extern "C"
{
#include "include/complex.h"
#include "path_table.h"
}

class FastxcProgressSidecar;

static const size_t kBytesPerGiB = 1ULL << 30;
static const size_t kBytesPerMiB = 1ULL << 20;
static const size_t kRowTargetBlocks = 1;

struct SpecMeta
{
  std::string path;
  std::string group;
  std::string network;
  std::string station;
  std::string location;
  std::string component;
  float stla = 0.0f;
  float stlo = 0.0f;
  int gnsl_id = 0;
};

struct StepackInputFragment
{
  std::string pack_path;
  size_t worker_id = 0;
  size_t batch_seq = 0;
  size_t start_group = 0;
  size_t group_count = 0;
  size_t nslc_start = 0;
  size_t nslc_count = 0;
  size_t batch_nslc_count = 0;
  size_t nstep = 0;
  size_t nspec = 0;
  float dt = 0.0f;
  float df = 0.0f;
  size_t payload_offset = 0;
  size_t payload_bytes = 0;
  size_t step_bytes = 0;
  size_t pitch_step_bytes = 0;
  size_t nslc_step_bytes = 0;
};

struct StepackFragment
{
  std::string pack_path;
  int fd = -1;
  size_t nslc_start = 0;
  size_t nslc_count = 0;
  size_t batch_nslc_count = 0;
  size_t payload_offset = 0;
  size_t payload_bytes = 0;
  size_t step_bytes = 0;
  size_t pitch_step_bytes = 0;
  size_t nslc_step_bytes = 0;
};

struct NslcLocator
{
  size_t fragment_index = 0;
  size_t fragment_local_index = 0;
};

struct TimestampWork
{
  std::string input_path;
  std::string timestamp;
  std::string input_pack_path;
  std::string manifest_path;
  std::vector<SpecMeta> specs;
  size_t num_channels = 0;
  size_t logical_step_bytes = 0; /* all files for one step */
  size_t logical_payload_bytes = 0;
  uint64_t manifest_hash_u64 = 0;
  std::vector<complex> payload_cache;
  bool payload_cache_enabled = false;
  std::vector<StepackFragment> stepack_fragments;
  std::vector<NslcLocator> nslc_locators;
};

struct TimestampInput
{
  std::string timestamp;
  std::string input_pack_path;
  std::string manifest_path;
  size_t file_count_hint = 0;
  std::vector<StepackInputFragment> stepack_fragments;
};

struct RuntimeShape
{
  int nspec = 0;
  int nstep = 0;
  int nfft = 0;
  int half_cc = 0;
  int cc_size = 0;
  float dt = 0.0f;
  float df = 0.0f;
  size_t num_channels = 0;
  size_t step_bytes = 0;
  size_t vec_count = 0;
  size_t vec_bytes = 0;
};

struct MemoryPlan
{
  size_t block_file_count = 0;
  size_t max_loaded_files = 0;
  size_t pair_capacity = 0;
  size_t host_step_input_bytes = 0;
  size_t host_cc_bytes = 0;
  size_t host_index_bytes = 0;
  size_t host_lazy_bytes = 0;
  size_t host_active_bytes = 0;
  size_t device_step_input_bytes = 0;
  size_t device_index_bytes = 0;
  size_t device_stack_bytes = 0;
  size_t device_time_bytes = 0;
  size_t device_sign_bytes = 0;
  size_t cufft_workspace_bytes = 0;
  size_t device_safety_bytes = 0;
  size_t device_total_bytes = 0;
};

struct RowBatchJob
{
  size_t anchor_block = 0;
  size_t target_begin_block = 0;
  size_t target_end_block = 0;
  size_t block_size = 0;
  size_t file_count = 0;
  size_t anchor_begin = 0;
  size_t anchor_end = 0;
  size_t target_begin = 0;
  size_t target_end = 0;
};

struct XcTask
{
  size_t src_meta_idx = 0;
  size_t rec_meta_idx = 0;
  size_t src_local_idx = 0;
  size_t rec_local_idx = 0;
  int path_id = 0;
  AllowedPathRecord path_record;
  bool is_autocorr = false;
};

struct JobQueue
{
  std::vector<RowBatchJob> jobs;
  size_t next = 0;
  pthread_mutex_t mutex;
};

struct WorkerConfig
{
  size_t worker_id = 0;
  size_t gpu_id = 0;
  size_t block_file_count = 1;
  size_t pair_capacity = 1;
  size_t lazy_write_depth = 0;
  const char *output_dir = NULL;
  FastxcProgressSidecar *progress = NULL;
};

struct WorkerContext
{
  WorkerConfig cfg;
  const RuntimeShape *shape = NULL;
  const AllowedPathTable *paths = NULL;
  const TimestampWork *timestamp = NULL;
  JobQueue *queue = NULL;
};

struct GpuBuffers
{
  complex *h_spec = NULL;
  float *h_cc = NULL;
  size_t *h_src_idx = NULL;
  size_t *h_rec_idx = NULL;
  cuComplex *d_spec = NULL;
  size_t *d_src_idx = NULL;
  size_t *d_rec_idx = NULL;
  cuComplex *d_stack = NULL;
  float *d_time = NULL;
  int *d_sign = NULL;
  void *d_cufft_work = NULL;
  size_t cufft_work_bytes = 0;
  size_t pair_capacity = 0;
  cufftHandle plan = 0;
};

struct WriteBatch
{
  RowBatchJob job;
  std::vector<XcTask> tasks;
  std::vector<float> cc;
};

struct LazyWriteQueue
{
  const WorkerConfig *cfg = NULL;
  const RuntimeShape *shape = NULL;
  const TimestampWork *timestamp = NULL;
  size_t pair_capacity = 0;
  size_t max_inflight = 0;
  size_t inflight = 0;
  bool closed = false;
  bool active = false;
  pthread_t thread;
  pthread_mutex_t mutex;
  pthread_cond_t can_push;
  pthread_cond_t can_pop;
  std::vector<WriteBatch> pool;
  std::deque<WriteBatch *> free_batches;
  std::deque<WriteBatch *> pending;
};

static bool checked_mul_size(size_t a, size_t b, size_t *out)
{
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    return false;
  *out = a * b;
  return true;
}

static bool checked_add_size(size_t a, size_t b, size_t *out)
{
  if (b > std::numeric_limits<size_t>::max() - a)
    return false;
  *out = a + b;
  return true;
}

static bool checked_add_to(size_t *acc, size_t value)
{
  return checked_add_size(*acc, value, acc);
}

static double bytes_to_gib(size_t bytes)
{
  return bytes / (double)kBytesPerGiB;
}

static double bytes_to_mib(size_t bytes)
{
  return bytes / (double)kBytesPerMiB;
}

#endif
