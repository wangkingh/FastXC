#include "cuda.util.cuh"

const float RAMUPPERBOUND = 0.9;

void DimCompute(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
{
    pdimblk->x = BLOCKX;
    pdimblk->y = BLOCKY;

    pdimgrd->x = (width + BLOCKX - 1) / BLOCKX;
    pdimgrd->y = (height + BLOCKY - 1) / BLOCKY;
}

void GpuFree(void **pptr)
{
    if (*pptr != NULL)
    {
        cudaFree(*pptr);
        *pptr = NULL;
    }
}

size_t QueryAvailGpuRam(size_t deviceID)
{
    size_t freeram, totalram;
    cudaSetDevice(deviceID);
    cudaMemGetInfo(&freeram, &totalram);
    freeram *= RAMUPPERBOUND;

    const size_t gigabytes = 1L << 30;
    printf("Avail gpu ram on device%zu: %.3f GB\n", deviceID,
           freeram * 1.0 / gigabytes);
    return freeram;
}

size_t EstimateGpuBatch(size_t gpu_id, int npts, int nseg, int nstep, int num_ch,
                        int filter_count, size_t wh_flag, size_t runabs_flag)
{
    int nseg_2x = nseg * 2;
    // CuFFT parameter
    int rank = 1;
    int n[1] = {nseg};
    int inembed[1] = {nseg};
    int onembed[1] = {nseg};
    int istride = 1;
    int idist = nseg;
    int ostride = 1;
    int odist = nseg;

    // CuFFT parameter for 2x zero padding data
    int rank_2x = 1;
    int n_2x[1] = {nseg_2x};
    int inembed_2x[1] = {nseg_2x};
    int onembed_2x[1] = {nseg_2x};
    int istride_2x = 1;
    int idist_2x = nseg_2x;
    int ostride_2x = 1;
    int odist_2x = nseg_2x;

    // unitgpuram setting
    size_t sac_seg_size = nseg * sizeof(float);            // d_segsac
    size_t spec_seg_size = nseg * sizeof(cuComplex);       // d_segspec
    size_t sac_seg_2x_size = nseg_2x * sizeof(float);      // d_segsac_2x for zero padding
    size_t spec_seg_2x_size = nseg_2x * sizeof(cuComplex); // d_segspec_2x for zero padding

    // gpuram for preprocessing rdc and rtr
    size_t pre_process_size = sizeof(double)    // d_sum
                              + sizeof(double); // d_isum

    size_t fixed_size = nseg * sizeof(cuComplex) * (filter_count + 1); // fixed memory for filter response

    // calcaulate gpuram for frequency whiten and time normalization
    size_t whiten_norm_size = 0;
    if (runabs_flag)
    {                                                       // If runabs normalization is applied
        whiten_norm_size = nseg * sizeof(float)             // d_weight
                           + nseg * sizeof(float)           // d_tmp_weight
                           + nseg * sizeof(cuComplex)       // d_tmp_spectrum
                           + nseg * sizeof(double)          // d_tmp
                           + nseg * sizeof(double)          // d_total_sac
                           + num_ch * nseg * sizeof(float); // d_sacdata_filterd
    }
    else if (!runabs_flag && wh_flag)
    {                                              // If only frequency whiten is applied
        whiten_norm_size = nseg * sizeof(float)    // d_weight
                           + nseg * sizeof(float)  // d_tmp_weight
                           + nseg * sizeof(float); // d_tmp
    }

    size_t unitgpuram = num_ch * sac_seg_size       // segment sac data nfft * float (nfft is redundant)
                        + num_ch * spec_seg_size    // segment spectrum data nfft * cuComplexs
                        + num_ch * pre_process_size // d_sum, d_isum
                        + num_ch * sac_seg_2x_size  // zero-padding segment spectrum data nfft * float
                        + num_ch * spec_seg_2x_size // zero-padding segment spectrum data nfft * cuComplex
                        + whiten_norm_size;         // whiten and normalization

    size_t availram = QueryAvailGpuRam(gpu_id);
    size_t reqram = 0;
    size_t tmpram = 0;
    size_t batch = 0;
    size_t step = 100;
    size_t last_valid_batch = 0;
    // print gpu memory usage ofgpu id in GB
    printf("Estimated GPU memory usage: %.3f GB\n", availram * 1.0 / (1L << 30));
    while (true)
    {
        batch += step;

        reqram = fixed_size + batch * unitgpuram;
        // cuFFT memory usage for data fft forward
        cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride,
                          odist, CUFFT_R2C, num_ch * batch, &tmpram);
        reqram += tmpram;
        // cuFFT memory usage for data fft inverse
        cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride,
                          odist, CUFFT_C2R, num_ch * batch, &tmpram);
        reqram += tmpram;

        // cuFFT memory usage for zero padding data fft forward
        cufftEstimateMany(rank_2x, n_2x, inembed_2x, istride_2x, idist_2x, onembed_2x, ostride_2x,
                          odist_2x, CUFFT_R2C, num_ch * batch, &tmpram);
        reqram += tmpram;

        cufftEstimateMany(rank_2x, n_2x, inembed_2x, istride_2x, idist_2x, onembed_2x, ostride_2x,
                          odist_2x, CUFFT_C2R, num_ch * batch, &tmpram);

        reqram += tmpram;

        if (reqram > availram)
        {
            if (step > 1)
            {
                batch -= step;
                step /= 2;
                continue;
            }
            else
            {
                batch = last_valid_batch;
                break;
            }
        }
        else
        {
            last_valid_batch = batch;
        }
    }
    batch = batch > _RISTRICT_MAX_GPU_BATCH ? _RISTRICT_MAX_GPU_BATCH : batch;
    printf("Estimated batch size: %zu\n", batch);
    return batch;
}

void AllocateGpuMemory(int batch, int nseg, int num_ch, int do_runabs, int wh_flag,
                       float **d_sacdata, cuComplex **d_spectrum,
                       float **d_sacdata_2x, cuComplex **d_spectrum_2x,
                       float **d_filtered_sacdata, float **d_total_sacdata,
                       cuComplex **d_filter_responses, float **d_tmp,
                       float **d_weight, float **d_tmp_weight,
                       int filterCount, double **d_sum, double **d_isum,
                       cufftHandle *planfwd, cufftHandle *planinv,
                       cufftHandle *planfwd_2x, cufftHandle *planinv_2x)
{
    int nseg_2x = nseg * 2;

    // Variables for processing segment data
    cudaMalloc((void **)d_sacdata, num_ch * batch * nseg * sizeof(float));
    cudaMalloc((void **)d_spectrum, num_ch * batch * nseg * sizeof(cuComplex));

    cudaMalloc((void **)d_sacdata_2x, num_ch * batch * nseg_2x * sizeof(float));
    cudaMalloc((void **)d_spectrum_2x, num_ch * batch * nseg_2x * sizeof(cuComplex));

    cudaMalloc((void **)d_filter_responses, filterCount * nseg_2x * sizeof(cuComplex));
    cudaMalloc((void **)d_sum, num_ch * batch * sizeof(double));
    cudaMalloc((void **)d_isum, num_ch * batch * sizeof(double));

    // if whiten, allocate memory for weight and tmp, same size as segment data
    if (!do_runabs && wh_flag)
    {
        cudaMalloc((void **)d_weight, batch * nseg * sizeof(float));
        cudaMalloc((void **)d_tmp_weight, batch * nseg * sizeof(float));
        cudaMalloc((void **)d_tmp, batch * nseg * sizeof(float));
    } // if runabs, allocate memory for filterd sac and spec
    else if (do_runabs)
    {
        cudaMalloc((void **)d_filtered_sacdata, num_ch * batch * nseg * sizeof(float));
        cudaMalloc((void **)d_total_sacdata, num_ch * batch * nseg * sizeof(float));
        cudaMalloc((void **)d_weight, batch * nseg * sizeof(float));
        cudaMalloc((void **)d_tmp_weight, batch * nseg * sizeof(float));
        cudaMalloc((void **)d_tmp, batch * nseg * sizeof(float));
    }

    // set up cufft plans
    int rank = 1;
    int n[1] = {nseg};
    int inembed[1] = {nseg};
    int onembed[1] = {nseg};
    int istride = 1;
    int idist = nseg;
    int ostride = 1;
    int odist = nseg;

    cufftPlanMany(planfwd, 1, n, inembed, istride, idist, onembed,
                  ostride, odist, CUFFT_R2C, num_ch * batch);
    cufftPlanMany(planinv, rank, n, inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_C2R, num_ch * batch);

    // set up cufft plans for zero-padding, both forward and inverse
    int rank_2x = 1;
    int n_2x[1] = {nseg_2x};
    int inembed_2x[1] = {nseg_2x};
    int onembed_2x[1] = {nseg_2x};
    int istride_2x = 1;
    int idist_2x = nseg_2x;
    int ostride_2x = 1;
    int odist_2x = nseg_2x;

    cufftPlanMany(planfwd_2x, rank_2x, n_2x, inembed_2x, istride_2x, idist_2x, onembed_2x,
                  ostride_2x, odist_2x, CUFFT_R2C, num_ch * batch);
    cufftPlanMany(planinv_2x, rank_2x, n_2x, inembed_2x, istride_2x, idist_2x, onembed_2x,
                  ostride_2x, odist_2x, CUFFT_C2R, num_ch * batch);
}
