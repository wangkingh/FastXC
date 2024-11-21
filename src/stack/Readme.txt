/********************************************************************
 * Stack the daily cross correlation NCF sac file by directory.
 *
 * This code will read in a directory path and iterate through all .sac files
 * under the directory.
 * You can choose to normalize them by divide N or just leave it un-normed.
 *
 * Since we create cross correlation file by outselves, their length and other
 * headers are always same for the same chn and same station. so we do not
 * add heavy check on the header info.
 *
 * Note:
 * 1.change all reference time to 2010/08/02/00:00:00:000 for pretty look
 * 2.number of stacked files are save in sachd.unused27
 * 3.changed from reading in list file to reading in directory path
 *
 *
 * History:
 * 1. many revision by wangwt@2009-2015
 * 2. Use linked list to store the file list so we now have no limit on the
 *    number of files listed. wangwt@20160411
 * 3. add -W options for special check wangwt@20160723
 * 4. revise the rdlist2link so the main code is shorter and we can now
 *    stack -O outputsac  file1 file2 file3...
 *
 * last update  wangwt@20190510
 *
 * OpenMP Version History:
 * 1. init by wuchao@202108
 * 2. use OpenMP multithread to perform parallel stackings
 * 3. get performance improvement on SSD, nearly no improvement on HDD
 * last update by wuchao@20211010
 *************************************************/