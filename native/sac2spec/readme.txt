/* ************************************************************
 * Jobs of this code are
 * 1. read in three component SAC data
 * 2. split the whole data into segments
 * 3. do runabs time normalization and spectrum whitenning forof E,N,Z data.
 *
 *    the weight used as the divisor is calculated using all three components
 *    both in time domain normalization and frequency domain whitenning.
 *
 * 4. save segment spectra into pitched step-major worker-batch files.
 *
 * The output is a stepack workspace used by FastXC:
 *   stepack/w<worker_id>.b<batch_seq>.stepack
 *   stepack/w<worker_id>.b<batch_seq>.tsv
 *
 * Each stepack file is a batch header + full batch NSLC table +
 * pitched [step][batch_nslc][freq] complex spectrum payload. The TSV sidecar
 * records timestamp-run virtual slices: pack path, byte offset, byte size,
 * nslc_start/nslc_count, batch acquisition order, logical group range, and
 * pitch metadata.
 *
 * Native XC reads the stepack workspace directly and uses the TSV sidecars as
 * virtual timestamp slices, so SAC2SPEC no longer materializes timestamp-local
 * spectrum packs.
 *
 * Note:
 * History:
 * 1. init by wangwt@2015

 * Cuda Version Histroy
 *
 * 1. init by wangjx with the thread distrubution strategy made by wuchao
 * 2. use 2d cuda threads to perform batch processing of list of sac files,
 *    X: segments of one sacfile, Y:list of sacfiles by wuchao@2021
 * 3. use cuda to implement rdc rtr npsmooth spectaper and etc by wuchao@2021 wangjx@2021,2022
 * 4. use cufft to perform FFT/IFFT transform by wuchao@2021
 * last updated by wangjx@20260530
 * *****************************************************************/
