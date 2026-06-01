/* ************************************************************
 * Jobs of this code are
 * 1. read in three component SAC data
 * 2. split the whole data into segments
 * 3. do runabs time normalization and spectrum whitenning forof E,N,Z data.
 *
 *    the weight used as the divisor is calculated using all three components
 *    both in time domain normalization and frequency domain whitenning.
 *
 * 4. save segment spectra into timestamp-local spack shards.
 *
 * The output is a spack workspace used by the FastXC xcache builder:
 *   spack_by_timestamp/<timestamp>/w<worker_id>.p<part_id>.spack
 *   spack_by_timestamp/<timestamp>/w<worker_id>.p<part_id>.tsv
 *
 * Each spack record is a raw SEGSPEC header followed by its complex spectrum payload.
 * The header layout is defined in segspec.h. The TSV sidecar records pack path,
 * byte offset, byte size, timestamp, source id, component, and spectrum metadata.
 *
 * FastXC converts the spack records into step-major xcache files before cross correlation.
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
