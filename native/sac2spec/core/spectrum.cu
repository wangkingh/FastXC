#include "spectrum.cuh"
#include "smoothing.cuh"

#include "../kernels/complex_matrix.cuh"
#include "../kernels/real_matrix.cuh"
#include "../kernels/taper.cuh"
#include "cuda.util.cuh"

#include <limits.h>

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

static int kernelOk(const char *op)
{
    return cudaOk(cudaGetLastError(), op);
}

static int validate2D(const void *ptr, size_t pitch, size_t width, size_t height,
                      const char *op)
{
    if (ptr == NULL || pitch == 0 || width == 0 || height == 0 || width > pitch)
    {
        LOG_ERROR("core_spectrum_invalid_2d_array",
                  "op=%s ptr=%p pitch=%zu width=%zu height=%zu",
                  op, ptr, pitch, width, height);
        return -1;
    }
    return 0;
}

int fft_forward_normalize(cuComplex *d_data,
                          size_t pitch,
                          size_t width,
                          size_t height,
                          float delta)
{
    if (validate2D(d_data, pitch, width, height, "forward_normalize") != 0 ||
        delta <= 0.0f)
    {
        if (delta <= 0.0f)
        {
            LOG_ERROR("core_spectrum_invalid_delta",
                      "op=forward_normalize delta=%.8g", delta);
        }
        return -1;
    }

    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);
    FwdNormalize2DKernel<<<dimgrd, dimblk>>>(d_data, pitch, width, height, delta);
    return kernelOk("forward_normalize");
}

int fft_inverse_normalize(float *d_data,
                          size_t pitch,
                          size_t width,
                          size_t height,
                          float delta)
{
    if (validate2D(d_data, pitch, width, height, "inverse_normalize") != 0 ||
        delta <= 0.0f)
    {
        if (delta <= 0.0f)
        {
            LOG_ERROR("core_spectrum_invalid_delta",
                      "op=inverse_normalize delta=%.8g", delta);
        }
        return -1;
    }

    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);
    InvNormalize2DKernel<<<dimgrd, dimblk>>>(d_data, pitch, width, height, delta);
    return kernelOk("inverse_normalize");
}

int complex_sanitize(cuComplex *d_data,
                     size_t pitch,
                     size_t width,
                     size_t height)
{
    if (validate2D(d_data, pitch, width, height, "complex_sanitize") != 0)
    {
        return -1;
    }

    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);
    cisnan2DKernel<<<dimgrd, dimblk>>>(d_data, pitch, width, height);
    return kernelOk("complex_sanitize");
}

int apply_filter_response(cuComplex *d_spectrum,
                          const float *d_response,
                          size_t pitch,
                          size_t width,
                          size_t height)
{
    if (validate2D(d_spectrum, pitch, width, height, "filter_response") != 0 ||
        d_response == NULL)
    {
        if (d_response == NULL)
        {
            LOG_ERROR("core_spectrum_invalid_filter_response",
                      "response=%p", d_response);
        }
        return -1;
    }

    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);
    filterKernel<<<dimgrd, dimblk>>>(d_spectrum, d_response, pitch, width, height);
    return kernelOk("filter_response");
}

static int spectral_taper(cuComplex *d_spectrum,
                          size_t pitch,
                          size_t width,
                          size_t height,
                          int np,
                          int idx1,
                          int idx2,
                          int idx3,
                          int idx4)
{
    if (validate2D(d_spectrum, pitch, width, height, "spectral_taper") != 0 ||
        np < 1 || idx1 < 0 || idx2 < idx1 || idx3 < idx2 || idx4 < idx3)
    {
        if (np < 1 || idx1 < 0 || idx2 < idx1 || idx3 < idx2 || idx4 < idx3)
        {
            LOG_ERROR("core_spectrum_invalid_taper",
                      "np=%d idx1=%d idx2=%d idx3=%d idx4=%d",
                      np, idx1, idx2, idx3, idx4);
        }
        return -1;
    }

    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);
    specTaper2DKernel<<<dimgrd, dimblk>>>(d_spectrum, pitch, width, height,
                                          np, idx1, idx2, idx3, idx4);
    return kernelOk("spectral_taper");
}

int phase_only(cuComplex *d_data,
               size_t pitch,
               size_t width,
               size_t height,
               float minval)
{
    if (validate2D(d_data, pitch, width, height, "phase_only") != 0 ||
        minval < 0.0f)
    {
        if (minval < 0.0f)
        {
            LOG_ERROR("core_spectrum_invalid_minval",
                      "op=phase_only minval=%.8g", minval);
        }
        return -1;
    }

    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);
    spectralOnebit2DKernel<<<dimgrd, dimblk>>>(d_data, pitch, width, height, minval);
    return kernelOk("phase_only");
}

int freq_whiten(cuComplex *d_spectrum,
                float *d_weight,
                float *d_tmp_weight,
                float *d_tmp,
                int num_ch,
                int pitch,
                int frame_count,
                float delta,
                int idx1,
                int idx2,
                int idx3,
                int idx4)
{
    if (d_spectrum == NULL || d_weight == NULL || d_tmp_weight == NULL ||
        d_tmp == NULL || num_ch <= 0 || pitch <= 0 || frame_count <= 0 ||
        delta <= 0.0f || idx1 < 0 || idx2 < idx1 || idx3 < idx2 ||
        idx4 < idx3 || frame_count > INT_MAX / num_ch)
    {
        LOG_ERROR("core_freq_whiten_invalid_input",
                  "spectrum=%p weight=%p tmp_weight=%p tmp=%p num_ch=%d pitch=%d frame_count=%d delta=%.8g idx1=%d idx2=%d idx3=%d idx4=%d",
                  d_spectrum, d_weight, d_tmp_weight, d_tmp, num_ch,
                  pitch, frame_count, delta, idx1, idx2, idx3, idx4);
        return -1;
    }

    int proc_cnt = frame_count * num_ch;
    int winsize = int(0.02 * pitch * delta);
    if (winsize < 1)
    {
        winsize = 1;
    }

    size_t big_pitch = (size_t)num_ch * (size_t)pitch;
    size_t fwidth = (size_t)pitch / 2 + 1;
    dim3 b_dimgrd, b_dimblk;
    dim3 c_dimgrd, c_dimblk;
    DimCompute(&b_dimgrd, &b_dimblk, fwidth, frame_count);
    DimCompute(&c_dimgrd, &c_dimblk, fwidth, proc_cnt);

    if (cudaOk(cudaMemset(d_weight, 0,
                          (size_t)proc_cnt * (size_t)pitch * sizeof(float)),
               "freq_whiten_memset_weight") != 0 ||
        cudaOk(cudaMemset(d_tmp_weight, 0,
                          (size_t)frame_count * (size_t)pitch * sizeof(float)),
               "freq_whiten_memset_tmp_weight") != 0 ||
        cudaOk(cudaMemset(d_tmp, 0,
                          (size_t)frame_count * (size_t)pitch * sizeof(float)),
               "freq_whiten_memset_tmp") != 0)
    {
        return -1;
    }

    if (complex_sanitize(d_spectrum, pitch, fwidth, proc_cnt) != 0)
    {
        return -1;
    }

    for (int k = 0; k < num_ch; k++)
    {
        amp2DKernel<<<b_dimgrd, b_dimblk>>>(d_tmp_weight, pitch,
                                            d_spectrum + (size_t)k * (size_t)pitch,
                                            big_pitch, fwidth, frame_count);
        if (kernelOk("freq_whiten_amp") != 0)
        {
            return -1;
        }

        if (cudaOk(cudaMemcpy2D(d_weight, (size_t)pitch * sizeof(float),
                                d_tmp_weight, (size_t)pitch * sizeof(float),
                                fwidth * sizeof(float), frame_count,
                                cudaMemcpyDeviceToDevice),
                   "freq_whiten_copy_amp_to_weight") != 0)
        {
            return -1;
        }

        launch_smooth_rows(d_tmp_weight, pitch, d_weight, pitch,
                           (int)fwidth, frame_count, winsize);
        if (kernelOk("freq_whiten_smooth") != 0)
        {
            return -1;
        }

        sum2DKernel<<<b_dimgrd, b_dimblk>>>(d_tmp, pitch,
                                            d_tmp_weight, pitch,
                                            fwidth, frame_count);
        if (kernelOk("freq_whiten_sum_weight") != 0)
        {
            return -1;
        }
    }

    clampmin2DKernel<<<b_dimgrd, b_dimblk>>>(d_tmp, pitch, fwidth,
                                             frame_count, MINVAL);
    if (kernelOk("freq_whiten_clamp_weight") != 0)
    {
        return -1;
    }

    expandSharedWeight2DKernel<<<c_dimgrd, c_dimblk>>>(d_weight, pitch,
                                                       d_tmp, pitch,
                                                       fwidth, proc_cnt,
                                                       num_ch);
    if (kernelOk("freq_whiten_expand_weight") != 0)
    {
        return -1;
    }

    cdiv2DKernel<<<c_dimgrd, c_dimblk>>>(d_spectrum, pitch, d_weight,
                                         pitch, fwidth, proc_cnt);
    if (kernelOk("freq_whiten_divide") != 0)
    {
        return -1;
    }

    if (spectral_taper(d_spectrum, pitch, fwidth, proc_cnt, 1,
                       idx1, idx2, idx3, idx4) != 0)
    {
        return -1;
    }

    if (complex_sanitize(d_spectrum, pitch, fwidth, proc_cnt) != 0)
    {
        return -1;
    }

    return 0;
}
