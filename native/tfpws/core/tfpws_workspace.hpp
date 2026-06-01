#ifndef TFPWS_WORKSPACE_HPP
#define TFPWS_WORKSPACE_HPP

#include "tfpws_compute.hpp"

int ensure_tfpws_fixed_cufft_plans(TfpwsDeviceWorkspace *workspace);
int ensure_tfpws_group_cufft_plans(TfpwsDeviceWorkspace *workspace,
                                   unsigned ngroups);

#endif
