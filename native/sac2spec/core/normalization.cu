#include "normalization.cuh"
#include "spectrum.cuh"
#include "smoothing.cuh"

#include "../kernels/real_matrix.cuh"
#include "cuda.util.cuh"

#include <limits.h>
#include <stdio.h>

static int cudaOk(cudaError_t err, const char *op)
{
    if (err != cudaSuccess)
    {
        LOG_ERROR("core_cuda_call_failed",
                  "op=%s code=%d name=%s message=\"%s\"",
                  op, (int)err, cudaGetErrorName(err), cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static int cufftOk(cufftResult err, const char *op)
{
    if (err != CUFFT_SUCCESS)
    {
        LOG_ERROR("core_cufft_call_failed", "op=%s code=%d", op, (int)err);
        return -1;
    }
    return 0;
}

static int kernelOk(const char *op)
{
    return cudaOk(cudaGetLastError(), op);
}

static int validateNormalizeArgs(const char *op,
                                 const float *d_data,
                                 const float *d_tmp,
                                 const float *d_weight,
                                 const float *d_tmp_weight,
                                 float freq_low,
                                 float delta,
                                 int frame_count,
                                 int num_ch,
                                 int pitch,
                                 float maxval)
{
    if (d_data == NULL || d_tmp == NULL || d_weight == NULL || d_tmp_weight == NULL ||
        freq_low <= 0.0f || delta <= 0.0f || frame_count <= 0 ||
        num_ch <= 0 || pitch <= 0 || maxval <= 0.0f ||
        frame_count > INT_MAX / num_ch)
    {
        LOG_ERROR("core_normalize_invalid_input",
                  "op=%s data=%p tmp=%p weight=%p tmp_weight=%p freq_low=%.8g delta=%.8g frame_count=%d num_ch=%d pitch=%d maxval=%.8g",
                  op, d_data, d_tmp, d_weight, d_tmp_weight,
                  freq_low, delta, frame_count, num_ch, pitch, maxval);
        return -1;
    }
    return 0;
}

static int validate2D(const void *ptr, size_t pitch, size_t width, size_t height,
                      const char *op)
{
    if (ptr == NULL || pitch == 0 || width == 0 || height == 0 || width > pitch)
    {
        LOG_ERROR("core_normalize_invalid_2d_array",
                  "op=%s ptr=%p pitch=%zu width=%zu height=%zu",
                  op, ptr, pitch, width, height);
        return -1;
    }
    return 0;
}

int time_onebit(float *d_data, size_t pitch,
                                size_t width, size_t height)
{
    if (validate2D(d_data, pitch, width, height, "time_onebit") != 0)
    {
        return -1;
    }

    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);
    onebit2DKernel<<<dimgrd, dimblk>>>(d_data, pitch, width, height);
    return kernelOk("time_onebit");
}

static int runabsNormalizeSharedWeight(float *d_data,
                                              float *d_tmp,
                                              float *d_weight,
                                              float *d_tmp_weight,
                                              float freq_low,
                                              float delta,
                                              int frame_count,
                                              int num_ch,
                                              int pitch,
                                              float maxval)
{
    if (validateNormalizeArgs("runabs", d_data, d_tmp, d_weight, d_tmp_weight,
                              freq_low, delta, frame_count, num_ch,
                              pitch, maxval) != 0)
    {
        return -1;
    }

    size_t twidth = (size_t)pitch;
    size_t frame_cnt = (size_t)frame_count;
    size_t proc_cnt = frame_cnt * (size_t)num_ch;
    dim3 b_tdimgrd, b_tdimblk;
    dim3 c_tdimgrd, c_tdimblk;
    DimCompute(&b_tdimgrd, &b_tdimblk, twidth, frame_cnt);
    DimCompute(&c_tdimgrd, &c_tdimblk, twidth, proc_cnt);

    size_t scratch_bytes = frame_cnt * twidth * sizeof(float);
    if (cudaOk(cudaMemset(d_tmp_weight, 0, scratch_bytes),
               "runabs_memset_tmp_weight") != 0 ||
        cudaOk(cudaMemset(d_tmp, 0, scratch_bytes),
               "runabs_memset_tmp") != 0)
    {
        return -1;
    }

    int winsize = 2 * int(1.0 / (freq_low * delta)) + 1;
    if (winsize < 1)
    {
        winsize = 1;
    }

    for (int k = 0; k < num_ch; k++)
    {
        if (cudaOk(cudaMemcpy2D(d_tmp_weight, twidth * sizeof(float),
                                d_data + (size_t)k * twidth,
                                (size_t)num_ch * twidth * sizeof(float),
                                twidth * sizeof(float), frame_cnt,
                                cudaMemcpyDeviceToDevice),
                   "runabs_copy_channel_to_tmp_weight") != 0)
        {
            return -1;
        }

        abs2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_tmp_weight, twidth,
                                              twidth, frame_cnt);
        if (kernelOk("runabs_abs") != 0)
        {
            return -1;
        }

        if (cudaOk(cudaMemcpy2D(d_weight, twidth * sizeof(float),
                                d_tmp_weight, twidth * sizeof(float),
                                twidth * sizeof(float), frame_cnt,
                                cudaMemcpyDeviceToDevice),
                   "runabs_copy_tmp_weight_to_weight") != 0)
        {
            return -1;
        }

        launch_smooth_rows(d_tmp_weight, pitch, d_weight, pitch,
                           pitch, frame_count, winsize);
        if (kernelOk("runabs_smooth") != 0)
        {
            return -1;
        }

        sum2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_tmp, twidth,
                                              d_tmp_weight, twidth,
                                              twidth, frame_cnt);
        if (kernelOk("runabs_sum_weight") != 0)
        {
            return -1;
        }
    }

    clampmin2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_tmp, twidth, twidth,
                                               frame_cnt, MINVAL);
    if (kernelOk("runabs_clamp_weight") != 0)
    {
        return -1;
    }

    expandSharedWeight2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_weight, twidth,
                                                         d_tmp, twidth,
                                                         twidth, proc_cnt,
                                                         num_ch);
    if (kernelOk("runabs_expand_weight") != 0)
    {
        return -1;
    }

    div2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_data, twidth, d_weight, twidth,
                                          twidth, proc_cnt);
    if (kernelOk("runabs_divide") != 0)
    {
        return -1;
    }

    isnan2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_data, twidth, twidth, proc_cnt);
    if (kernelOk("runabs_isnan") != 0)
    {
        return -1;
    }

    cutmax2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_data, twidth, twidth,
                                             proc_cnt, maxval);
    if (kernelOk("runabs_cutmax") != 0)
    {
        return -1;
    }

    return 0;
}

int runabs(float *d_data,
                           float *d_tmp,
                           float *d_weight,
                           float *d_tmp_weight,
                           float freq_low,
                           float delta,
                           int frame_count,
                           int num_ch,
                           int pitch,
                           float maxval)
{
    return runabsNormalizeSharedWeight(d_data, d_tmp, d_weight,
                                              d_tmp_weight, freq_low, delta,
                                              frame_count, num_ch, pitch,
                                              maxval);
}

int runabs_mf(float *d_sacdata,
                              float *d_filtered_sacdata,
                              float *d_total_sacdata,
                              float *d_padded_sacdata,
                              cuComplex *d_padded_spectrum,
                              cuComplex *d_base_padded_spectrum,
                              float *d_responses,
                              float *d_tmp,
                              float *d_weight,
                              float *d_tmp_weight,
                              float *freq_lows,
                              int filter_count,
                              float delta,
                              int frame_count,
                              int num_ch,
                              float maxval,
                              int segment_pts,
                              int filter_nfft,
                              int plan_batch,
                              cufftHandle *planinv_filter,
                              cufftHandle *planfwd_filter)
{
    if (d_sacdata == NULL || d_filtered_sacdata == NULL ||
        d_total_sacdata == NULL || d_padded_sacdata == NULL ||
        d_padded_spectrum == NULL || d_base_padded_spectrum == NULL ||
        d_responses == NULL || d_tmp == NULL || d_weight == NULL ||
        d_tmp_weight == NULL || freq_lows == NULL ||
        planinv_filter == NULL || planfwd_filter == NULL ||
        filter_count <= 0 || delta <= 0.0f || frame_count <= 0 ||
        num_ch <= 0 || maxval <= 0.0f || segment_pts <= 0 ||
        filter_nfft <= 0 || plan_batch < frame_count ||
        frame_count > INT_MAX / num_ch)
    {
        LOG_ERROR("core_runabs_mf_invalid_input",
                  "sac=%p filtered=%p total=%p padded_sac=%p padded_spectrum=%p base_padded_spectrum=%p responses=%p tmp=%p weight=%p tmp_weight=%p freq_lows=%p filter_count=%d delta=%.8g frame_count=%d num_ch=%d maxval=%.8g segment_pts=%d filter_nfft=%d plan_batch=%d",
                  d_sacdata, d_filtered_sacdata, d_total_sacdata,
                  d_padded_sacdata, d_padded_spectrum, d_base_padded_spectrum,
                  d_responses, d_tmp, d_weight, d_tmp_weight, freq_lows,
                  filter_count, delta, frame_count, num_ch, maxval,
                  segment_pts, filter_nfft, plan_batch);
        return -1;
    }

    size_t twidth = (size_t)segment_pts;
    size_t proc_cnt = (size_t)frame_count * (size_t)num_ch;
    size_t plan_cnt = (size_t)plan_batch * (size_t)num_ch;

    dim3 c_tdimgrd, c_tdimblk;
    DimCompute(&c_tdimgrd, &c_tdimblk, twidth, proc_cnt);

    size_t filter_width = (size_t)filter_nfft;
    size_t filter_fwidth = (size_t)filter_nfft / 2 + 1;

    if (cudaOk(cudaMemset(d_total_sacdata, 0,
                          proc_cnt * twidth * sizeof(float)),
               "runabs_mf_memset_total") != 0 ||
        cudaOk(cudaMemset(d_padded_sacdata, 0,
                          plan_cnt * filter_width * sizeof(float)),
               "runabs_mf_memset_padded_sac") != 0 ||
        cudaOk(cudaMemset(d_base_padded_spectrum, 0,
                          plan_cnt * filter_width * sizeof(cuComplex)),
               "runabs_mf_memset_base_spectrum") != 0 ||
        cudaOk(cudaMemcpy2D(d_padded_sacdata, filter_width * sizeof(float),
                            d_sacdata, twidth * sizeof(float),
                            twidth * sizeof(float), proc_cnt,
                            cudaMemcpyDeviceToDevice),
               "runabs_mf_copy_input_to_filter_buffer") != 0)
    {
        return -1;
    }

    if (cufftOk(cufftExecR2C(*planfwd_filter, (cufftReal *)d_padded_sacdata,
                             (cufftComplex *)d_base_padded_spectrum),
                "runabs_mf_forward_filter_fft") != 0)
    {
        return -1;
    }

    if (fft_forward_normalize(d_base_padded_spectrum,
                                          filter_width,
                                          filter_fwidth,
                                          proc_cnt, delta) != 0)
    {
        return -1;
    }

    if (complex_sanitize(d_base_padded_spectrum,
                                         filter_width,
                                         filter_fwidth,
                                         proc_cnt) != 0)
    {
        return -1;
    }

    for (int i = 1; i < filter_count; i++)
    {
        if (cudaOk(cudaMemset(d_padded_spectrum, 0,
                              plan_cnt * filter_width * sizeof(cuComplex)),
                   "runabs_mf_memset_work_spectrum") != 0 ||
            cudaOk(cudaMemcpy2D(d_padded_spectrum, filter_width * sizeof(cuComplex),
                                d_base_padded_spectrum, filter_width * sizeof(cuComplex),
                                filter_fwidth * sizeof(cuComplex), proc_cnt,
                                cudaMemcpyDeviceToDevice),
                   "runabs_mf_copy_base_spectrum") != 0)
        {
            return -1;
        }

        if (apply_filter_response(d_padded_spectrum,
                                                  d_responses + (size_t)i * filter_width,
                                                  filter_width,
                                                  filter_fwidth,
                                                  proc_cnt) != 0)
        {
            return -1;
        }

        if (cufftOk(cufftExecC2R(*planinv_filter, (cufftComplex *)d_padded_spectrum,
                                 (cufftReal *)d_padded_sacdata),
                    "runabs_mf_inverse_filter_fft") != 0)
        {
            return -1;
        }

        if (fft_inverse_normalize(d_padded_sacdata,
                                              filter_width,
                                              filter_width,
                                              proc_cnt,
                                              delta) != 0)
        {
            return -1;
        }

        if (cudaOk(cudaMemcpy2D(d_filtered_sacdata, twidth * sizeof(float),
                                d_padded_sacdata, filter_width * sizeof(float),
                                twidth * sizeof(float), proc_cnt,
                                cudaMemcpyDeviceToDevice),
                   "runabs_mf_copy_filtered_back") != 0)
        {
            return -1;
        }

        if (runabsNormalizeSharedWeight(d_filtered_sacdata, d_tmp,
                                               d_weight, d_tmp_weight,
                                               freq_lows[i], delta,
                                               frame_count, num_ch,
                                               segment_pts, maxval) != 0)
        {
            return -1;
        }

        sum2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_total_sacdata, twidth,
                                              d_filtered_sacdata, twidth,
                                              twidth, proc_cnt);
        if (kernelOk("runabs_mf_sum_filtered") != 0)
        {
            return -1;
        }
    }

    return 0;
}
