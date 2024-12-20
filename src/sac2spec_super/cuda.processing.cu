#include "cuda.processing.cuh"

// pre-processing for sacdat: isnan, demean, detrend
void preprocess(float *d_sacdata, double *d_sum, double *d_isum, int pitch, size_t proccnt, float freq_low, float delta)
{
    size_t width = pitch;
    size_t height = proccnt;
    dim3 dimgrd, dimblk;
    DimCompute(&dimgrd, &dimblk, width, height);

    dim3 dimgrd2, dimblk2;
    dimblk2.x = BLOCKMAX;
    dimblk2.y = 1;
    dimgrd2.x = 1;
    dimgrd2.y = height;

    isnan2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height);

    // demean. First calculate the mean value of each trace
    size_t dpitch = 1;
    size_t spitch = pitch;
    sumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                             dimblk2.x * dimblk2.y * sizeof(double)>>>(
        d_sum, dpitch, d_sacdata, spitch, width, height);

    DimCompute(&dimgrd, &dimblk, width, height);
    rdc2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height, d_sum);

    // detrend. First calculate d_sum and d_isum
    sumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                             dimblk2.x * dimblk2.y * sizeof(double)>>>(
        d_sum, dpitch, d_sacdata, spitch, width, height);

    isumSingleBlock2DKernel<<<dimgrd2, dimblk2,
                              dimblk2.x * dimblk2.y * sizeof(double)>>>(
        d_isum, dpitch, d_sacdata, spitch, width, height);

    rtr2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height, d_sum, d_isum);

    size_t taper_size = 2 * (1 / freq_low) / delta;
    timetaper2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height, taper_size); // taper
}

// multi-frequency run-abs time domain normalization
void runabs_mf(float *d_sacdata, float *d_filtered_sacdata, float *d_total_sacdata,
               float *d_sacdata_2x, cuComplex *d_spectrum_2x,
               cuComplex *d_responses, float *d_tmp,
               float *d_weight, float *d_tmp_weight,
               cufftHandle *planinv, float *freq_lows,
               int filterCount, float delta, int proc_batch, int num_ch, float maxval,
               int nseg_1x, int nseg_2x, cufftHandle *planinv_2x, cufftHandle *planfwd_2x)
{
    size_t twidth = nseg_1x;
    size_t fwidth = nseg_1x * 0.5 + 1;

    size_t big_pitch = num_ch * nseg_1x; // the distance from the start of the current row to the start of the same channel in the next row.
    size_t proc_cnt = proc_batch * num_ch;

    // calculate the grid and block size for time domain and frequency domain
    // b means for batch processing, c means for cnt(channel) processing
    dim3 b_tdimgrd, b_tdimblk, b_fdimgrd, b_fdimblk;
    dim3 c_tdimgrd, c_tdimblk, c_fdimgrd, c_fdimblk;
    DimCompute(&b_tdimgrd, &b_tdimblk, twidth, proc_batch);
    DimCompute(&b_fdimgrd, &b_fdimblk, fwidth, proc_batch);
    DimCompute(&c_tdimgrd, &c_tdimblk, twidth, proc_cnt);
    DimCompute(&c_fdimgrd, &c_fdimblk, fwidth, proc_cnt);

    // calculate the grid and block size for 2x length spectrum
    // for 2x length spectrum, fwidth_2x = nseg_2x / 2 + 1, the pitch here is set as nseg_1x, thus fwidth_2x = pitch + 1
    size_t twidth_2x = nseg_2x;
    size_t fwidth_2x = 0.5 * nseg_2x + 1;
    dim3 c_fdimgrd_2x, c_fdimblk_2x, c_tdimgrd_2x, c_tdimblk_2x;
    DimCompute(&c_fdimgrd_2x, &c_fdimblk_2x, fwidth_2x, proc_cnt);
    DimCompute(&c_tdimgrd_2x, &c_tdimblk_2x, twidth_2x, proc_cnt);

    // clean the total sacdata
    CUDACHECK(cudaMemset(d_total_sacdata, 0, proc_cnt * nseg_1x * sizeof(float)));

    // time domain normalization on different frequency and add them together
    for (int i = 1; i < filterCount; i++)
    {
        //  ============ 零填充滤波开始 ==================
        CUDACHECK(cudaMemset(d_sacdata_2x, 0, proc_cnt * nseg_2x * sizeof(float)));
        CUDACHECK(cudaMemset(d_spectrum_2x, 0, proc_cnt * fwidth_2x * sizeof(cuComplex)));
        CUDACHECK(cudaMemcpy2D(d_sacdata_2x, nseg_2x * sizeof(float),
                               d_sacdata, nseg_1x * sizeof(float),
                               nseg_1x * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice));
        CUFFTCHECK(cufftExecR2C(*planfwd_2x, (cufftReal *)d_sacdata_2x, (cufftComplex *)d_spectrum_2x));
        FwdNormalize2DKernel<<<c_fdimgrd_2x, c_fdimblk_2x>>>(d_spectrum_2x, nseg_2x, nseg_2x, proc_cnt, delta);
        cisnan2DKernel<<<c_fdimgrd_2x, c_fdimblk_2x>>>(d_spectrum_2x, nseg_2x, nseg_2x, proc_cnt);
        filterKernel<<<c_fdimgrd_2x, c_fdimblk_2x>>>(d_spectrum_2x, d_responses + i * nseg_2x, nseg_2x, fwidth_2x, proc_cnt);
        CUFFTCHECK(cufftExecC2R(*planinv_2x, (cufftComplex *)d_spectrum_2x, (cufftReal *)d_sacdata_2x)); // 逆变换回时域(2x)
        InvNormalize2DKernel<<<c_tdimgrd_2x, c_tdimblk_2x>>>(d_sacdata_2x, nseg_2x, nseg_2x, proc_cnt, delta);
        CUDACHECK(cudaMemcpy2D(d_filtered_sacdata, nseg_1x * sizeof(float),
                               d_sacdata_2x, nseg_2x * sizeof(float),
                               nseg_1x * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice)); // 截取回原长度数据
        //  ============ 零填充滤波结束 ==================

        // Time domain run-abs normalization
        CUDACHECK(cudaMemset(d_weight, 0, proc_batch * nseg_1x * sizeof(float)));
        CUDACHECK(cudaMemset(d_tmp_weight, 0, proc_batch * nseg_1x * sizeof(float)));
        CUDACHECK(cudaMemset(d_tmp, 0, proc_batch * nseg_1x * sizeof(float)));
        int winsize = 2 * int(1.0 / (freq_lows[i] * delta)) + 1; // refrence from Yao's code winsize = SampleF * EndT
        for (int k = 0; k < num_ch; k++)
        {
            CUDACHECK(cudaMemcpy2D(d_tmp_weight, nseg_1x * sizeof(float),
                                   d_filtered_sacdata + k * nseg_1x, big_pitch * sizeof(float),
                                   nseg_1x * sizeof(float), proc_batch, cudaMemcpyDeviceToDevice));
            abs2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_tmp_weight, nseg_1x, twidth, proc_batch);
            CUDACHECK(cudaMemcpy2D(d_tmp, nseg_1x * sizeof(float), d_tmp_weight, nseg_1x * sizeof(float), twidth * sizeof(float), proc_batch, cudaMemcpyDeviceToDevice));
            smooth2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_tmp_weight, nseg_1x, d_tmp, nseg_1x, twidth, proc_batch, winsize);
            sum2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_weight, nseg_1x, d_tmp_weight, nseg_1x, twidth, proc_batch);
        }
        clampmin2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_weight, nseg_1x, twidth, proc_batch, MINVAL); // avoid the minimum value

        for (int k = 0; k < num_ch; k++)
        {
            div2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_filtered_sacdata + k * nseg_1x, big_pitch, d_weight, nseg_1x, twidth, proc_batch); // divide
        }

        // Post Processing
        isnan2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_filtered_sacdata, nseg_1x, twidth, proc_cnt);
        cutmax2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_filtered_sacdata, nseg_1x, twidth, proc_cnt, maxval);                // avoid too big value
        sum2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_total_sacdata, nseg_1x, d_filtered_sacdata, nseg_1x, twidth, proc_cnt); // adding [d_filtered_sacdata] of different bands to [d_sacdata]
    }
}

// Wang Weitao's version
void runabs(float *d_sacdata, float *d_tmp, float *d_weight, float *d_tmp_weight,
            float freq_lows_limit, float delta, int proc_batch, int num_ch, int pitch, float maxval)
{
    size_t twidth = pitch;
    size_t fwidth = pitch * 0.5 + 1;
    size_t big_pitch = num_ch * pitch; // the distance from the start of the current row to the start of the same channel in the next row.
    size_t proc_cnt = proc_batch * num_ch;

    // calculate the grid and block size for time domain and frequency domain
    // b means for batch processing, c means for cnt processing
    dim3 b_tdimgrd, b_tdimblk;
    dim3 c_tdimgrd, c_tdimblk;
    DimCompute(&b_tdimgrd, &b_tdimblk, twidth, proc_batch);
    DimCompute(&c_tdimgrd, &c_tdimblk, twidth, proc_cnt);

    // Time domain run-abs normalization
    CUDACHECK(cudaMemset(d_weight, 0, proc_batch * pitch * sizeof(float)));
    CUDACHECK(cudaMemset(d_tmp_weight, 0, proc_batch * pitch * sizeof(float)));
    CUDACHECK(cudaMemset(d_tmp, 0, proc_batch * pitch * sizeof(float)));
    int winsize = 2 * int(1.0 / (freq_lows_limit * delta)) + 1; // refrence from Yao's code winsize = SampleF * EndT
    for (int k = 0; k < num_ch; k++)
    {
        CUDACHECK(cudaMemcpy2D(d_tmp_weight, pitch * sizeof(float),
                               d_sacdata + k * pitch, big_pitch * sizeof(float),
                               pitch * sizeof(float), proc_batch, cudaMemcpyDeviceToDevice));
        abs2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_tmp_weight, pitch, twidth, proc_batch);
        CUDACHECK(cudaMemcpy2D(d_tmp, pitch * sizeof(float), d_tmp_weight, pitch * sizeof(float), twidth * sizeof(float), proc_batch, cudaMemcpyDeviceToDevice));
        smooth2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_tmp_weight, pitch, d_tmp, pitch, twidth, proc_batch, winsize);
        sum2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_weight, pitch, d_tmp_weight, pitch, twidth, proc_batch);
    }
    clampmin2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_weight, pitch, twidth, proc_batch, MINVAL); // avoid the minimum value

    for (int k = 0; k < num_ch; k++)
    {
        div2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_sacdata + k * pitch, big_pitch, d_weight, pitch, twidth, proc_batch); // divide
    }

    // Post Processing
    isnan2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_sacdata, pitch, twidth, proc_cnt);
    cutmax2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_sacdata, pitch, twidth, proc_cnt, maxval); // avoid too big value
}

void freqWhiten(cuComplex *d_spectrum,
                float *d_weight, float *d_tmp_weight, float *d_tmp,
                int num_ch, int pitch, int proc_batch,
                float delta, int idx1, int idx2, int idx3, int idx4)
{
    int proc_cnt = proc_batch * num_ch;
    int winsize = int(0.02 * pitch * delta);
    size_t big_pitch = num_ch * pitch; // the distance from the start of the current row to the start of the same channel in the next row.
    size_t fwidth = pitch * 0.5 + 1;
    dim3 b_dimgrd, b_dimblk, c_dimgrd, c_dimblk; // for batch and for cnt
    DimCompute(&b_dimgrd, &b_dimblk, fwidth, proc_batch);
    DimCompute(&c_dimgrd, &c_dimblk, fwidth, proc_cnt);

    CUDACHECK(cudaMemset(d_weight, 0, proc_batch * pitch * sizeof(float)));
    CUDACHECK(cudaMemset(d_tmp_weight, 0, proc_batch * pitch * sizeof(float)));
    CUDACHECK(cudaMemset(d_tmp, 0, proc_batch * pitch * sizeof(float)));
    cisnan2DKernel<<<b_dimgrd, b_dimblk>>>(d_spectrum, pitch, fwidth, proc_cnt);
    for (size_t k = 0; k < num_ch; k++)
    {
        amp2DKernel<<<b_dimgrd, b_dimblk>>>(d_tmp_weight, pitch, d_spectrum + k * pitch, big_pitch, fwidth, proc_batch);
        CUDACHECK(cudaMemcpy2D(d_tmp, pitch * sizeof(float),
                               d_tmp_weight, pitch * sizeof(float),
                               fwidth * sizeof(float), proc_batch, cudaMemcpyDeviceToDevice));
        smooth2DKernel<<<b_dimgrd, b_dimblk>>>(d_tmp_weight, pitch, d_tmp, pitch, fwidth, proc_batch, winsize);
        sum2DKernel<<<b_dimgrd, b_dimblk>>>(d_weight, pitch, d_tmp_weight, pitch, fwidth, proc_batch);
    }

    for (size_t k = 0; k < num_ch; k++)
    {
        cdiv2DKernel<<<b_dimgrd, b_dimblk>>>(d_spectrum + k * pitch, big_pitch, d_weight, pitch, fwidth, proc_batch);
    }
    clampmin2DKernel<<<b_dimgrd, b_dimblk>>>(d_weight, pitch, fwidth, proc_batch, MINVAL); // avoid the minimum value
    specTaper2DKernel<<<c_dimgrd, c_dimblk>>>(d_spectrum, pitch, fwidth, proc_cnt, 1, idx1, idx2, idx3, idx4);
    cisnan2DKernel<<<b_dimgrd, b_dimblk>>>(d_spectrum, pitch, fwidth, proc_cnt);
}
