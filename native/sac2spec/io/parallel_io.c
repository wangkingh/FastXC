#include "parallel_io.h"

#include "io_pool.h"
#include "logger.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAC_DELTA_ABS_TOL 1.0e-6f
#define SAC_DELTA_REL_TOL 1.0e-5f

struct ThreadPoolRead
{
  IoPool *pool;
};

typedef struct ReadSacContext
{
  InOutNode *items;
  int target_npts;
  float expected_delta;
} ReadSacContext;

static float sacDeltaTolerance(float expected)
{
  float tolerance = fabsf(expected) * SAC_DELTA_REL_TOL;
  if (tolerance < SAC_DELTA_ABS_TOL)
  {
    tolerance = SAC_DELTA_ABS_TOL;
  }
  return tolerance;
}

static int sacDeltaMatches(float observed, float expected, float *diff_out, float *tolerance_out)
{
  float diff = fabsf(observed - expected);
  float tolerance = sacDeltaTolerance(expected);
  if (diff_out != NULL)
  {
    *diff_out = diff;
  }
  if (tolerance_out != NULL)
  {
    *tolerance_out = tolerance;
  }
  return diff <= tolerance;
}

static int readOneSacFile(void *context, size_t index)
{
  ReadSacContext *read_context = (ReadSacContext *)context;
  InOutNode *node = &read_context->items[index];

  if (read_sachead(node->sacpath, node->sac_hd) != 0)
  {
    LOG_ERROR("read_sachead_failed", "path=\"%s\"", node->sacpath);
    return -1;
  }

  int source_npts = node->sac_hd->npts;
  float source_delta = node->sac_hd->delta;
  float delta_diff = 0.0f;
  float delta_tolerance = 0.0f;
  if (!sacDeltaMatches(source_delta, read_context->expected_delta,
                       &delta_diff, &delta_tolerance))
  {
    LOG_ERROR("sac_delta_mismatch",
              "path=\"%s\" expected=%.9g observed=%.9g diff=%.9g tolerance=%.9g",
              node->sacpath, read_context->expected_delta, source_delta,
              delta_diff, delta_tolerance);
    return -1;
  }

  if (read_sac_buffer(node->sacpath, node->sac_hd,
                      node->sac_data, read_context->target_npts) == NULL)
  {
    LOG_ERROR("read_sac_failed", "path=\"%s\"", node->sacpath);
    return -1;
  }

  if (source_npts != read_context->target_npts)
  {
    LOG_WARN("sac_npts_adjusted",
             "path=\"%s\" source_npts=%d target_npts=%d action=%s",
             node->sacpath, source_npts, read_context->target_npts,
             source_npts > read_context->target_npts ? "truncated" : "zero_padded");
  }
  node->sac_hd->npts = read_context->target_npts;
  node->sac_hd->delta = read_context->expected_delta;

  return 0;
}

ThreadPoolRead *CreateReadIoPool(size_t num_threads)
{
  ThreadPoolRead *pool = (ThreadPoolRead *)malloc(sizeof(ThreadPoolRead));
  if (pool == NULL)
  {
    LOG_ERROR("alloc_failed", "target=read_thread_pool");
    return NULL;
  }

  pool->pool = IoPoolCreate(num_threads, "read");
  if (pool->pool == NULL)
  {
    free(pool);
    return NULL;
  }
  return pool;
}

int ReadSacBatchParallel(ThreadPoolRead *pool, size_t item_count,
                         InOutNode *items, int num_threads,
                         int target_npts, float expected_delta)
{
  ReadSacContext context;
  context.items = items;
  context.target_npts = target_npts;
  context.expected_delta = expected_delta;

  return IoPoolRun(pool != NULL ? pool->pool : NULL,
                   item_count, num_threads, "read_sac",
                   "read_sac_start", "read_sac_done", "read_thread_failed",
                   readOneSacFile, &context);
}

void DestroyReadIoPool(ThreadPoolRead *pool)
{
  if (pool == NULL)
  {
    return;
  }
  IoPoolDestroy(pool->pool);
  free(pool);
}
