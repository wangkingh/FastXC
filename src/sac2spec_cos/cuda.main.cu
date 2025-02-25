/* last updated by wangjx@20240108 */

#include "cuda.processing.cuh"
#include "cuda.util.cuh"
extern "C"
{
#include "arguproc.h"
#include "design_filter_response.h"
#include "in_out_node.h"
#include "cal_nseg.h"
#include "read_filelist.h"
#include "par_rw_data.h"
#include "sac.h"
#include "util.h"
}

int main(int argc, char **argv)
{
    // Parsing arguments
    ARGUTYPE argument;
    ArgumentProcess(argc, argv, &argument);
    PathNode *pInFileList = readPathList(argument.sac_lst);   // Read in sac path list file
    PathNode *pOutFileList = readPathList(argument.spec_lst); // Read in spec output path list file

    createDirectories(pOutFileList); // Create output_dir for .segspec files

    FilePathArray InPaths = PathList2Array(pInFileList); // Turn file chain input list chain into array
    FilePathArray OutPaths = PathList2Array(pOutFileList);

    int num_ch = argument.num_ch; // Get the number of channels

    // Parsing Whiten and Normalization Type
    int wh_before = 0, wh_after = 0, do_runabs_mf = 0, do_runabs = 0, do_onebit = 0;
    switch (argument.whitenType)
    {
    case 0:
        wh_before = 0, wh_after = 0;
        break;
    case 1:
        wh_before = 1;
        break;
    case 2:
        wh_after = 1;
        break;
    case 3:
        wh_before = 1, wh_after = 1;
        break;
    default:
        printf("Invalid value for bandwhiten\n");
    }

    switch (argument.normalizeType)
    {
    case 0:
        do_runabs_mf = 0, do_onebit = 0, do_runabs = 0;
        break;
    case 1:
        do_runabs_mf = 1;
        break;
    case 2:
        do_onebit = 1;
        break;
    case 3:
        do_runabs = 1;
        break;
    default:
        printf("Invalid value for normalization\n");
    }

    size_t nValid_sacnum = InPaths.count;         // Count the number of input files
    size_t nValid_batch = nValid_sacnum / num_ch; // Count the number of input files/channels

    size_t gpu_id = argument.gpu_id; // Set the gpu_id
    cudaSetDevice(gpu_id);

    SACHEAD sachd;
    if (read_sachead(pInFileList->path, &sachd) != 0)
    {
        fprintf(stderr, "ERROR reading first SACHEAD\n");
        exit(EXIT_FAILURE);
    }
    int npts = sachd.npts; // Read npts and delta form the file file of pInFileList
    float delta = sachd.delta;

    int nseg = cal_nseg(argument.seglen, npts, delta); // Calculate the number of points of sement
    int nstep = npts / nseg;

    int nseg_2x = nseg * 2; // Create nseg_2x for zero padding data
    int nspec_output = nseg_2x / 2 + 1;

    float df_1x = 1.0 / (nseg * delta);
    float df_2x = 1.0 / (nseg_2x * delta);

    // Parse frequcny band, calculate idx of corner and cutoff frequency
    float freq_low = argument.freq_low;
    float freq_high = argument.freq_high;
    int f_idx1 = int(freq_low * 0.667 / df_1x);
    int f_idx2 = int(freq_low / df_1x);
    int f_idx3 = int(freq_high / df_1x);
    int f_idx4 = int(freq_high * 1.333 / df_1x);

    int filter_count = 0;
    ButterworthFilter *filter = readButterworthFilters(argument.filter_file, &filter_count); // read in filter file
    FilterResp *myResp = processButterworthFilters(filter, filter_count, df_1x, nseg);       // Calculate filter f domain response

    float f1[MAXBAND] = {0}, f2[MAXBAND] = {0}, f3[MAXBAND] = {0}, f4[MAXBAND] = {0};

    // 假设 filter_count 不超过 MAXBAND
    for (int i = 0; i < filter_count && i < MAXBAND; i++)
    {
        float orig_freq_low = filter[i].freq_low;
        float orig_freq_high = filter[i].freq_high;

        // 计算修改后的截止频率
        float cutoff_low = orig_freq_low * 0.67f;   // f2拐角频率下限
        float cutoff_high = orig_freq_high * 1.33f; // f2拐角频率上限

        // 根据需要将结果存入相应数组中
        f1[i] = cutoff_low;
        f4[i] = cutoff_high;

        // 若f1或f4需要其他相关处理，也可在此添加逻辑
        // 比如如需存入原始频率，可这样写：
        f2[i] = orig_freq_low;
        f3[i] = orig_freq_high;
    }

    // Parsing skip_steps
    int *skip_steps = argument.skip_steps;
    int skip_step_count = argument.skip_step_count;
    int nstep_valid = nstep;

    int skip_flags[nstep]; // Mark which steps should be skipped
    memset(skip_flags, false, sizeof(skip_flags));

    for (int i = 0; i < skip_step_count; ++i)
    {
        int skip_step = skip_steps[i];
        if (skip_step >= 0 && skip_step < nstep)
        {
            skip_flags[skip_step] = true;
            printf("Step [no. %d] will be skipped \n", skip_step);
        }
    }

    nstep_valid = 0; // Calculate validated steps
    for (int i = 0; i < nstep; ++i)
    {
        if (!skip_flags[i])
        {
            ++nstep_valid;
        }
    }

    // ********* Calculate CPU memory *********** //
    float *h_sacdata = NULL;
    complex *h_spectrum = NULL;

    InOutNode *pInOutList = NULL;

    size_t unit_sacdata_size = npts * sizeof(float);                          // input sac data
    size_t unit_spectrum_size = nstep_valid * nspec_output * sizeof(complex); // output total spectrum
    size_t unit_InOutNode_size = sizeof(InOutNode);                           // contain head/path/data
    size_t unit_thread_write_size = sizeof(thread_info_write);                // contain thread_info_write
    size_t unit_thread_read_size = sizeof(thread_info_read);                  // contain thread_info_read

    size_t unitCpuRam = num_ch * (unit_sacdata_size +
                                  unit_spectrum_size +
                                  unit_InOutNode_size +
                                  unit_thread_write_size +
                                  unit_thread_read_size);

    size_t h_batch = EstimateCpuBatch(unitCpuRam, argument.gpu_num);

    // Allocate memory for GPU
    size_t wh_flag = wh_after || wh_before;
    size_t d_batch = EstimateGpuBatch(gpu_id, npts, nseg, nstep_valid, num_ch, filter_count, wh_flag, do_runabs_mf || do_runabs);
    h_batch = (h_batch > nValid_batch) ? nValid_batch : h_batch;
    d_batch = (d_batch > h_batch) ? h_batch : d_batch;
    h_batch = d_batch;
    size_t batch = 0.7 * h_batch;
    batch = batch < 1 ? 1 : batch;

    printf("[GPU no.%zu]: batch = %ld, num_ch=%d\n", gpu_id, batch, num_ch);

    // Allocate CPU memory
    CpuMalloc((void **)&pInOutList, num_ch * batch * unit_InOutNode_size);
    CpuMalloc((void **)&h_sacdata, num_ch * batch * unit_sacdata_size);
    CpuMalloc((void **)&h_spectrum, num_ch * batch * unit_spectrum_size);

    // Initializing the memory for InOutNode, batch *num_ch datas will be processed at same time
    size_t sacpathSize = MAXPATH * sizeof(char);
    size_t specpathSize = MAXPATH * sizeof(char);
    size_t spechdSize = sizeof(SEGSPEC);
    size_t sachdSize = sizeof(SACHEAD);
    for (size_t i = 0; i < batch * num_ch; i++)
    {
        CpuMalloc((void **)&(pInOutList[i].sacpath), sacpathSize);
        CpuMalloc((void **)&(pInOutList[i].specpath), specpathSize);

        CpuMalloc((void **)&(pInOutList[i].sac_hd), sachdSize);
        CpuMalloc((void **)&(pInOutList[i].segspec_hd), spechdSize);

        pInOutList[i].sac_data = h_sacdata + i * npts;
        pInOutList[i].spectrum = h_spectrum + i * nstep_valid * nspec_output;

        pInOutList[i].nspec = nspec_output;
        pInOutList[i].nstep = nstep_valid;
        pInOutList[i].df = df_2x;
        pInOutList[i].dt = delta;
    }

    // ********** Allocate memory for GPU ************
    float *d_sacdata = NULL;          // segment sacdata in GPU
    float *d_sacdata_2x = NULL;       // 2x  length segment sacdata in GPU
    float *d_filtered_sacdata = NULL; // filteredsegment sacdata in GPU

    cuComplex *d_spectrum = NULL;          // segment spectrum in GPU
    cuComplex *d_spectrum_2x = NULL;       // 2x  length segment spectrum in GPU
    cuComplex *d_filtered_spectrum = NULL; // used in runabs

    float *d_weight = NULL;     // weight of each segment spectrum data and sac data
    float *d_tmp = NULL;        // used in runabs
    float *d_tmp_weight = NULL; // used in runabs and whiten, store weight of single channel data

    cuComplex *d_responses = NULL; // butterwrth 2th filter responses

    double *d_sum = NULL;  // used in rtr and rdc
    double *d_isum = NULL; // used in rtr and rdc

    cufftHandle planfwd;    // forward fft
    cufftHandle planinv;    // inverse fft
    cufftHandle planfwd_2x; // forward fft, for 2x zero-padding series

    // Allocating memory in GPU Device
    AllocateGpuMemory(batch, nseg, num_ch, do_runabs || do_runabs_mf, wh_flag,
                      &d_sacdata, &d_spectrum,
                      &d_sacdata_2x, &d_spectrum_2x,
                      &d_filtered_sacdata, &d_filtered_spectrum,
                      &d_responses, &d_tmp,
                      &d_weight, &d_tmp_weight,
                      filter_count, &d_sum, &d_isum,
                      &planfwd, &planinv, &planfwd_2x);
    // copy each filter response to GPU and store freq_low array
    float freq_lows[filter_count];
    for (int i = 0; i < filter_count; i++)
    {
        freq_lows[i] = myResp[i].freq_low;
        CUDACHECK(cudaMemcpy(d_responses + i * nseg, myResp[i].response, nseg * sizeof(cuComplex), cudaMemcpyHostToDevice));
    }

    // ********** MAIN PROCESSING PART ********** /
    for (size_t finish_batch = 0; finish_batch < nValid_batch; finish_batch += batch)
    {
        size_t proc_batch = (finish_batch + batch > nValid_batch) ? nValid_batch - finish_batch : batch;

        // Setup Input Output Chain
        size_t finish_cnt = finish_batch * num_ch;
        size_t proc_cnt = proc_batch * num_ch;
        for (size_t i = finish_cnt, j = 0; i < finish_cnt + proc_cnt; i++, j++)
        {
            pInOutList[j].sacpath = InPaths.paths[i];
            pInOutList[j].specpath = OutPaths.paths[i];
        }
        memset(h_sacdata, 0, proc_cnt * npts * sizeof(float));
        memset(h_spectrum, 0, proc_cnt * nstep_valid * nspec_output * sizeof(complex));

        ThreadPoolRead *read_pool = create_threadpool_read(argument.thread_num);
        ThreadPoolWrite *write_pool = create_threadpool_write(argument.thread_num);
        parallel_read_sac(read_pool, proc_cnt, pInOutList, argument.thread_num); // Parallel read SAC data

        dim3 dimgrd_1x, dimblk_1x, dimgrd_2x, dimblk_2x;
        DimCompute(&dimgrd_1x, &dimblk_1x, nseg, proc_cnt);
        DimCompute(&dimgrd_2x, &dimblk_2x, nseg_2x, proc_cnt);
        for (int stepidx = 0, done_step = 0; stepidx < nstep; stepidx++)
        {
            int skip_this_step = 0;
            for (int flag_i = 0; flag_i < argument.skip_step_count; flag_i++)
            {
                if (argument.skip_steps[flag_i] == stepidx)
                {
                    skip_this_step = 1;
                    break;
                }
            }

            if (skip_this_step)
            {
                continue;
            }

            // clean d_sacdata_* and d_spectrum_*
            CUDACHECK(cudaMemset(d_sacdata, 0, proc_cnt * nseg * sizeof(float)));
            CUDACHECK(cudaMemset(d_spectrum, 0, proc_cnt * nseg * sizeof(cuComplex)));
            CUDACHECK(cudaMemset(d_sacdata_2x, 0, proc_cnt * nseg_2x * sizeof(float)));
            CUDACHECK(cudaMemset(d_spectrum_2x, 0, proc_cnt * nseg_2x * sizeof(cuComplex)));

            // copy sacdata to GPU
            CUDACHECK(cudaMemcpy2D(d_sacdata, nseg * sizeof(float),
                                   h_sacdata + stepidx * nseg, npts * sizeof(float),
                                   nseg * sizeof(float), proc_cnt, cudaMemcpyHostToDevice));
            preprocess(d_sacdata, d_sum, d_isum, nseg, proc_cnt, TAPER_RATIO); // isnan,rtr,reman,taper

            CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_sacdata, (cufftComplex *)d_spectrum));
            FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_spectrum, nseg, nseg, proc_cnt, delta);

            // Filtering add 20240711 by wangjx
            dim3 c_fdimgrd, c_fdimblk;
            size_t fwidth = 0.5 * nseg + 1;
            DimCompute(&c_fdimgrd, &c_fdimblk, fwidth, proc_cnt);
            // filterKernel<<<c_fdimgrd, c_fdimblk>>>(d_spectrum, d_responses, nseg, fwidth, proc_cnt);
            CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_spectrum, (cufftReal *)d_sacdata));
            InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_sacdata, nseg, nseg, proc_cnt, delta);

            if (wh_before)
            {
                freqWhiten(d_spectrum, d_weight, d_tmp_weight, d_tmp, num_ch, nseg, proc_batch, delta, f_idx1, f_idx2, f_idx3, f_idx4);
                cisnan2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_spectrum, nseg, nseg, proc_cnt);
            }

            //   Time Domain Normalization
            if (do_runabs_mf)
            {
                cisnan2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_spectrum, nseg, nseg, proc_cnt);

                runabs_mf(d_sacdata, d_spectrum,
                          d_filtered_sacdata, d_filtered_spectrum,
                          d_responses, d_tmp,
                          d_weight, d_tmp_weight,
                          &planinv, freq_lows,
                          df_1x, f1, f2, f3, f4,
                          filter_count, delta, proc_batch, num_ch, nseg, MAXVAL);
            }
            else if (do_onebit)
            {
                if (wh_before)
                {
                    CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_spectrum, (cufftReal *)d_sacdata));
                    InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_sacdata, nseg, nseg, proc_cnt, delta);
                }
                onebit2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_sacdata, nseg, nseg, proc_cnt);
            }
            else if (do_runabs)
            {
                float freq_lows_limit = freq_low * 0.667;
                if (wh_before)
                {
                    CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_spectrum, (cufftReal *)d_sacdata));
                    InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_sacdata, nseg, nseg, proc_cnt, delta);
                }
                runabs(d_sacdata, d_tmp, d_weight, d_tmp_weight, freq_lows_limit, delta, proc_batch, num_ch, nseg, MAXVAL);
            }

            if (do_runabs || do_runabs_mf || do_onebit) // trans data from time domain to freq domain if did time domain normalization
            {
                cisnan2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_spectrum, nseg, nseg, proc_cnt);
                CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_sacdata, (cufftComplex *)d_spectrum));
                FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_spectrum, nseg, nseg, proc_cnt, delta);
            }

            if (wh_after)
            {
                freqWhiten(d_spectrum, d_weight, d_tmp_weight, d_tmp, num_ch, nseg, proc_batch, delta, f_idx1, f_idx2, f_idx3, f_idx4);
            }

            // Zero-Padding
            cisnan2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_spectrum, nseg, nseg, proc_cnt);
            CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_spectrum, (cufftReal *)d_sacdata));
            InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_sacdata, nseg, nseg, proc_cnt, delta);
            preprocess(d_sacdata, d_sum, d_isum, nseg, proc_cnt, TAPER_RATIO); // Post process using the preprocess function
            CUDACHECK(cudaMemcpy2D(d_sacdata_2x, nseg_2x * sizeof(float), d_sacdata, nseg * sizeof(float), nseg * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice));
            CUFFTCHECK(cufftExecR2C(planfwd_2x, (cufftReal *)d_sacdata_2x, (cufftComplex *)d_spectrum_2x));
            FwdNormalize2DKernel<<<dimgrd_2x, dimblk_2x>>>(d_spectrum_2x, nseg_2x, nseg_2x, proc_cnt, delta);
            cisnan2DKernel<<<dimgrd_2x, dimblk_2x>>>(d_spectrum_2x, nseg_2x, nseg_2x, proc_cnt);

            // Copy data from d_spectrum back to h_spectrum
            CUDACHECK(cudaMemcpy2D(h_spectrum + done_step * nspec_output, nstep_valid * nspec_output * sizeof(cuComplex),
                                   d_spectrum_2x, nseg_2x * sizeof(cuComplex),
                                   nspec_output * sizeof(complex), proc_cnt, cudaMemcpyDeviceToHost));
            done_step++;
        }

        // End loop of each segment, Writing the output spectrum
        parallel_write_spec(write_pool, proc_cnt, pInOutList, argument.thread_num);
        destroy_threadpool_write(write_pool);
        destroy_threadpool_read(read_pool);
    }
    // Free memory
    cufftDestroy(planfwd);
    cufftDestroy(planinv);
    cufftDestroy(planfwd_2x);
    GpuFree((void **)&d_sacdata);
    GpuFree((void **)&d_spectrum);
    GpuFree((void **)&d_sacdata_2x);
    GpuFree((void **)&d_spectrum_2x);
    GpuFree((void **)&d_filtered_sacdata);
    GpuFree((void **)&d_responses);
    GpuFree((void **)&d_weight);
    GpuFree((void **)&d_tmp);
    GpuFree((void **)&d_filtered_spectrum);
    GpuFree((void **)&d_tmp_weight);
    GpuFree((void **)&d_sum);
    GpuFree((void **)&d_isum);
    CpuFree((void **)&h_sacdata);
    CpuFree((void **)&h_spectrum);
    CpuFree((void **)&pInOutList);
}