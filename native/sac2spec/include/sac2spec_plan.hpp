#ifndef SAC2SPEC_PLAN_HPP
#define SAC2SPEC_PLAN_HPP

#include <cstddef>

extern "C"
{
#include "in_out_node.h"
}

typedef struct FilterResp FilterResp;
typedef struct ProgressState ProgressState;
typedef struct TimestampTracker TimestampTracker;

typedef struct Sac2SpecPlan
{
    FilePathArray in_paths;
    SacIndexMetaArray meta;
    ProgressState *progress;
    TimestampTracker *timestamp_tracker;
    const char *stepack_root;

    int num_ch;
    int npts;
    float delta;
    int segment_pts;
    int shift_pts;
    int nstep_valid;
    int output_nfft;
    int nspec_output;
    int filter_nfft;
    int filter_count;

    float freq_low;
    int f_idx1;
    int f_idx2;
    int f_idx3;
    int f_idx4;
    float df_output;

    int wh_before;
    int wh_after;
    int output_phase_only;
    int do_runabs_mf;
    int do_runabs;
    int do_onebit;
    int lazy_async;
    int host_slot_count;
    size_t wh_flag;

    int *valid_steps;
    FilterResp *filter_responses;
    float *freq_lows;

    size_t unit_sacdata_size;
    size_t unit_spectrum_size;
    size_t unit_InOutNode_size;
} Sac2SpecPlan;

#endif
