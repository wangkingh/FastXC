#ifndef TFPWS_COMPUTE_HPP
#define TFPWS_COMPUTE_HPP

#include <cstddef>

#include <cuComplex.h>
#include <cufft.h>

extern "C"
{
#include "arguproc.h"
#include "sac.h"
}

struct TfpwsDeviceWorkspacePlan
{
    unsigned nsamples;
    unsigned max_ngroups;
    std::size_t nfreq;
    std::size_t freq_chunk_size;
    std::size_t num_freq_chunks;
    long double resident_data_bytes;
    std::size_t cufft_workspace_bytes;
    std::size_t runtime_reserve_bytes;
    long double planned_peak_bytes;
    int band_limited;
};

struct TfpwsDeviceWorkspace
{
    TfpwsDeviceWorkspacePlan plan;
    std::size_t worker_index;
    int device_id;
    bool initialized;

    void *d_cufft_workspace;
    float *d_linear_stack;
    float *d_prestack_data;
    float *d_group_trace_weights;
    cuComplex *d_trace_spectrum;
    cufftComplex *d_linear_spectrum;
    cufftComplex *d_out_spectrum;
    cufftComplex *d_tfpw_stack_complex;
    float *d_tfpw_stack;
    cufftComplex *d_stack_tf_chunk;
    cufftComplex *d_chunk_spectrum;
    cuComplex *d_weight_chunk;
    cufftComplex *d_trace_tf_chunk;

    cufftHandle plan_fwd_traces;
    cufftHandle plan_fwd_single_trace;
    cufftHandle plan_inv_stack_chunk;
    cufftHandle plan_inv_trace_chunk;
    cufftHandle plan_inv_final;
    bool fixed_cufft_plans_ready;
    bool group_cufft_plans_ready;
    unsigned cached_ngroups;
    std::size_t fixed_cufft_workspace_bytes;
    std::size_t group_cufft_workspace_bytes;
    std::size_t actual_cufft_workspace_bytes;
};

void init_tfpws_device_workspace_struct(TfpwsDeviceWorkspace *workspace);
int allocate_tfpws_device_workspace(TfpwsDeviceWorkspace *workspace,
                                    const TfpwsDeviceWorkspacePlan *plan,
                                    std::size_t worker_index,
                                    int device_id);
void free_tfpws_device_workspace(TfpwsDeviceWorkspace *workspace);

int compute_tfpws_from_prestack(const char *label,
                                SACHEAD header,
                                float *prestack_data,
                                float *linear_stack,
                                float *group_trace_weights,
                                unsigned num_segments,
                                unsigned ngroups,
                                unsigned nsamples,
                                const ARGUTYPE *argument,
                                std::size_t worker_index,
                                int device_id,
                                TfpwsDeviceWorkspace *workspace,
                                SACHEAD *out_header,
                                float **out_data);

#endif
