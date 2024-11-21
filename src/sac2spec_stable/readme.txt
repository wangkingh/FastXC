/* ************************************************************
 * Jobs of this code are
 * 1. read in three component SAC data
 * 2. split the whole data into segments
 * 3. do runabs time normalization and spectrum whitenning forof E,N,Z data.
 *
 *    the weight used as the divisor is calculated using all three components
 *    both in time domain normalization and frequency domain whitenning.
 *
 * 4. save spectrum of each segments and write to one file.
 *
 * The output would be segment-spectrum file which will be used later to calculate the noise
 * cross correlation.
 *
 * The information of segment and spectrum is saved in the header of each file which can be
 * found in segspec.h
 *
 * Calculated segment spectrum first would require more disk usage but may speed up the Cross
 * Correlation when you have huge dataset eg too many stations.
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
 * last updated by wangjx@20230703
 * *****************************************************************/