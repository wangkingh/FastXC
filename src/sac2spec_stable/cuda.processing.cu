#include "cuda.processing.cuh"

void mycustom_filter(float *d_sacdata, float *d_filtered, float *d_filtered_tmp,
                     double *d_a, double *d_b, float *d_sac_hist, float *d_filtered_hist,
                     size_t width, size_t height)
{
    dim3 dimgrd, dimblk;

    // forward filtering
    DimCompute(&dimgrd, &dimblk, 1, height);
    CUDACHECK(cudaMemset(d_sac_hist, 0, height * 5 * sizeof(float)));
    CUDACHECK(cudaMemset(d_filtered_hist, 0, height * 5 * sizeof(float)));
    filterTKernel<<<dimgrd, dimblk>>>(d_sacdata, d_filtered_tmp, d_a, d_b, d_sac_hist, d_filtered_hist, width, height);

    // reverse the filtered data
    DimCompute(&dimgrd, &dimblk, width, height);
    reverseKernel<<<dimgrd, dimblk>>>(d_filtered_tmp, d_filtered, width, height);

    // backward filtering
    DimCompute(&dimgrd, &dimblk, 1, height);
    CUDACHECK(cudaMemset(d_sac_hist, 0, height * 5 * sizeof(float)));
    CUDACHECK(cudaMemset(d_filtered_hist, 0, height * 5 * sizeof(float)));
    filterTKernel<<<dimgrd, dimblk>>>(d_filtered, d_filtered_tmp, d_a, d_b, d_sac_hist, d_filtered_hist, width, height);

    // reverse the filtered data
    DimCompute(&dimgrd, &dimblk, width, height);
    reverseKernel<<<dimgrd, dimblk>>>(d_filtered_tmp, d_filtered, width, height);
}

// pre-processing for sacdat: isnan, demean, detrend
void preprocess(float *d_sacdata, double *d_sum, double *d_isum, int pitch, size_t proccnt, int taper_percentage)
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

    float taper_fraction = (float)taper_percentage / 100.0;
    size_t taper_size = width * taper_fraction;
    timetaper2DKernel<<<dimgrd, dimblk>>>(d_sacdata, pitch, width, height, taper_size); // taper, taper percentage set in config.h
}

// multi-frequency run-abs time domain normalization
void runabs_mf(float *d_sacdata, float *d_sac_sum,
               float *d_filtered, float *d_filtered_tmp,
               double *d_a, double *d_b,
               float *d_sac_hist, float *d_filtered_hist,
               float *d_tmp, float *d_weight, float *d_tmp_weight,
               ButterworthFilter *filters,
               int filterCount, float delta, int proc_batch, int num_ch, int pitch, float maxval)
{
    size_t twidth = pitch;
    size_t fwidth = pitch * 0.5 + 1;
    size_t big_pitch = num_ch * pitch; // the distance from the start of the current row to the start of the same channel in the next row.
    size_t proc_cnt = proc_batch * num_ch;

    // calculate the grid and block size for time domain and frequency domain
    // b means for batch processing, c means for cnt processing
    dim3 dimgrd, dimblk;
    dim3 b_tdimgrd, b_tdimblk;
    dim3 c_tdimgrd, c_tdimblk;
    DimCompute(&dimgrd, &dimblk, 1, proc_batch);
    DimCompute(&b_tdimgrd, &b_tdimblk, twidth, proc_batch);
    DimCompute(&c_tdimgrd, &c_tdimblk, twidth, proc_cnt);

    // clean d_sac_sum
    CUDACHECK(cudaMemset(d_sac_sum, 0, proc_cnt * pitch * sizeof(float)));

    // time domain normalization on different frequency and add them together
    for (int i = 1; i < filterCount; i++)
    {
        CUDACHECK(cudaMemcpy(d_a, filters[i].a, 5 * sizeof(double), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_b, filters[i].b, 5 * sizeof(double), cudaMemcpyHostToDevice));

        CUDACHECK(cudaMemset(d_filtered, 0, proc_cnt * pitch * sizeof(float)));
        CUDACHECK(cudaMemset(d_filtered_tmp, 0, proc_cnt * pitch * sizeof(float)));
        CUDACHECK(cudaMemset(d_sac_hist, 0, proc_cnt * 5 * sizeof(float)));
        CUDACHECK(cudaMemset(d_filtered_hist, 0, proc_cnt * 4 * sizeof(float)));

        mycustom_filter(d_sacdata, d_filtered, d_filtered_tmp, d_a, d_b, d_sac_hist, d_filtered_hist, twidth, proc_cnt);
        int winsize = 2 * int(1.0 / (filters[i].freq_low * delta)) + 1; // refrence from Yao's code winsize = SampleF * EndT

        CUDACHECK(cudaMemset(d_weight, 0, proc_batch * pitch * sizeof(float)));
        CUDACHECK(cudaMemset(d_tmp_weight, 0, proc_batch * pitch * sizeof(float)));
        CUDACHECK(cudaMemset(d_tmp, 0, proc_batch * pitch * sizeof(float)));

        for (int k = 0; k < num_ch; k++) // iterate over different channels
        {
            CUDACHECK(cudaMemcpy2D(d_tmp_weight, pitch * sizeof(float),
                                   d_filtered + k * pitch, big_pitch * sizeof(float),
                                   pitch * sizeof(float), proc_batch, cudaMemcpyDeviceToDevice));
            abs2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_tmp_weight, pitch, twidth, proc_batch);
            CUDACHECK(cudaMemcpy2D(d_tmp, pitch * sizeof(float), d_tmp_weight, pitch * sizeof(float), twidth * sizeof(float), proc_batch, cudaMemcpyDeviceToDevice));
            smooth2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_tmp_weight, pitch, d_tmp, pitch, twidth, proc_batch, winsize);
            sum2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_weight, pitch, d_tmp_weight, pitch, twidth, proc_batch);
        }
        clampmin2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_weight, pitch, twidth, proc_batch, MINVAL); // avoid the minimum value

        for (int k = 0; k < num_ch; k++)
        {
            div2DKernel<<<b_tdimgrd, b_tdimblk>>>(d_filtered + k * pitch, big_pitch, d_weight, pitch, twidth, proc_batch); // divide
        }

        // Post Processing
        isnan2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_filtered, pitch, twidth, proc_cnt);
        cutmax2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_filtered, pitch, twidth, proc_cnt, maxval);        // avoid too big value
        sum2DKernel<<<c_tdimgrd, c_tdimblk>>>(d_sac_sum, pitch, d_filtered, pitch, twidth, proc_cnt); // adding [d_filtered_sacdata] of different bands to [d_sacdata]
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
