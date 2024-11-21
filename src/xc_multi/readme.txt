**************************************************************************
* This code is for Green's function extraction using long time ambient noise
* cross correlation for single component,usually ZZ
*
* Input files are previously created segment-spectrums. This vector version
* requires you specify two input from the same component at one time.
*
* 1.Those file by -A is take as virtual source
*   while file by -B is taken as virtual station
*
*   the NCF.evlo and NCF.evla are set to be position of source
*   and NCF.stlo and NCF.stla are set to be position of station.
*   also we save the knetwk and kstnm of station as NCF's netwk and stnm
*   and  we save the knetwk and kstnm of source  as NCF's kevnm.
*   Since kevnm is 16 characters(include terminate \0), we set evnm to
*   kevnm=A.knetwk space A.kstnm, this is consistent by taking A as source.
*
*   Azimuth is the angle clockwise between vector north and vector
*   Source->Station and back-azimuth is the angle clockwise between vector
*   north and vector Station->Source so you can check the source direction
*   using all NCF ends with B according the baz.   wangwt@20101028
*
*   Also we set the cmpaz and cmpinc for 1st and 2nd station in NCF sacheader
*
* 2.kcomp will be created using the last character in chnnm of two input if
*   it is not esplicitly set by -N
*
* 3.For 2nd station, the cmpaz and cmpinc are stored in sac.cmpaz and sac.cmpinc
*   For 1st station, the original sac header do not have coresponding key, so
*   I change sachd.unused11 to sachd.scmpaz  to store cmpaz for 1st station
*while change   sachd.unused12 to sachd.scmpinc to store cmpinc for 1st station
*   You can use unuser11 or 12 to access those value and they should be used
*   mainly in tensor NCF rotation.  wangwt@20160701
*
*
* History:
* 1. init by wangwt@2015
* 2. add scmpaz and scmpinc by wangwt@20160701
* 3. set khole and kinst so we have complete info of Net.Sta.Loc
*    for both virt source and virt station. wangwt@20160711
* 4. add -D option, very simple but could output NCFs of each small segment@2018
* 5. add -S option, change the target directory style of output ncfs, by
*stations wangjx@2022
*
* last update wangwt@20181125
*
* Cuda Version History:
* 1. init by wuchao@202105
* 2. use 2d cuda threads to perform batch processing of list of pairs of spec
*files, X: segments of spec pair, Y: list of pairs of spec files
* 3. use cuda to implement speccc and etc
* 4. use cufft to perform FFT/IFFT transform
* 5. add -RD option to avoid calculating duplicate NCFs. Like A2B and B2A
*wangjx@2022
*
* last update by wuchao@20211008
* last updated by wangjx@20230716
* **************************************************************************/