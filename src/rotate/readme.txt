/* **************************************************************************
 * This code is to rotate nine NCF tesnor with ENZ basis to nine NCF tensor 
 * with RTZ basis.
 *
 * 1. We assume you have obtained the EE EN EZ NE NN NZ ZE ZN ZZ stacked NCF and
 *    the stlo/stla/evlo/evla are all set correctly by assuming the first station
 *    as source and 2nd station as receiver.
 *    The time seriers and spectrums should be normalized by same norminator so that
 *    the ration between ENZ thus RTZ could be remained even after non-linear 
 *    processings.
 * 2. Then we rotate ENZ to RTZ using one rotation matrix bellow
 * 3. All nine input tensor are required while output are optional controlled by 
 *    -T flag. -T Flag is any combination of 1-9 such as 123,139,159 etc. Only digit allowed
 *       Number             Tensor
 *     | 1 2 3 |          | RR RT RZ |
 *     | 4 5 6 |  <====>  | TR TT TZ |
 *     | 7 8 9 |          | ZR ZT ZZ |
 *
 * History:
 * 1. init by wangwt@2013?
 * 2. revised and add more documents by wangwt@20160525
 * 3. add correction for cmpaz for components like BH1 BH2  etc wangwt@20160710
 * 4. add -H option 20200223
 *
 * last update by wangwt@20200223
 *
 * More Docs
 * -----------------------------------------------------------------------------------
 *  1. Azimuth  is the angle clockwise between vector North and vector Source->Station
 *     Back-Azimuth is angle clockwise between vector North and vectror Station->Source
 *  2. When we rotate EN into RT, the resulting R component will be positive in the 
 *     specified direction indicating by one angle and the T component will be positive 
 *     90 degrees clockwise of R. By taking the angle as theta, theta will be one angle
 *     clockwise from north( If you are rotating E/N).
 *     If the N and E input data are positive to the North and East respectively the 
 *     resulting R component will be positive along the azimuth rotated to and the T 
 *     component will be positive along an axis 90 clockwise of the R component.
 *  3. For the cross correlation, we will take the first station as source and 2nd as
 *     receiver.
 *     For the source, the theta should be azimuth taking the 2nd station as receiver.
 *     so
 *     theta = azi
 *     Rs = E*sin(theta) + N* cos(theta) = E*+sin(azi) + N*+cos(azi)
 *     Ts = E*cos(theta) + N*-sin(theta) = E*+cos(azi) + N*-sin(azi)
 *
 *     For the receiver, the theta should be theta=back-azimuth -180 degree
 *     so
 *     Rr = E*sin(theta) + N* cos(theta) = E*-sin(baz) + N*-cos(baz)
 *     Tr = E*cos(theta) + N*-sin(theta) = E*-cos(baz) + N*+sin(baz)
 * 4.  The relation of nine RTZ tensor and nine ENZ tensor is 
 *     Taking 
 *     Rs = E*+sin(azi) + N*+cos(azi)  Ts = E*+cos(azi) + N*-sin(azi)  Zs = Z
 *     and
 *     Rr = E*-sin(baz) + N*-cos(baz)  Tr = E*-cos(baz) + N*+sin(baz)  Zr = Z 
 *     we will have
 *
 *     RR = EE*-sin(azi)*sin(baz) + EN*-sin(azi)*cos(baz) + NE*-cos(azi)*sin(baz) + NN*-cos(azi)*cos(baz)
 *     RT = EE*-sin(azi)*cos(baz) + EN*+sin(azi)*sin(baz) + NE*-cos(azi)*cos(baz) + NN*cos(azi)*sin(baz)
 *     RZ = EZ*sin(azi) + NZ*cos(azi)  
 *
 *     TR = EE*-cos(azi)sin(baz) + EN*-cos(azi)cos(baz) + NE*+sin(azi)sin(baz) + NN*+sin(azi)cos(baz)
 *     TT = EE*-cos(azi)cos(baz) + EN*+cos(azi)sin(baz) + NE*+sin(azi)cos(baz) + NN*-sin(azi)sin(baz)
 *     TZ = EZ*+cos(azi) + NZ*-sin(azi)
 *
 *     ZR = ZE*-sin(baz) + ZN*-cos(baz)
 *     ZT = ZE*-cos(baz) + ZN*+sin(baz)
 *     ZZ = ZZ
 *
 *     so the rotation matrix is ( Hoho, view this in one wide screen ....)  
 *
 * RR   [ -sin(azi)sin(baz) -sin(azi)cos(baz)      0       -cos(azi)sin(baz)  -cos(azi)cos(baz)        0            0         0        0 ] [ EE ]
 * RT   [ -sin(azi)cos(baz) +sin(azi)sin(baz)      0       -cos(azi)cos(baz)  +cos(azi)sin(baz)        0            0         0        0 ] [ EN ]
 * RZ   [         0                 0           sin(azi)           0                 0              cos(azi)        0         0        0 ] [ EZ ]
 * TR = [ -cos(azi)sin(baz) -cos(azi)cos(baz)      0       +sin(azi)sin(baz)  +sin(azi)cos(baz)        0            0         0        0 ] [ NE ]
 * TT   [ -cos(azi)cos(baz) +cos(azi)sin(baz)      0       +sin(azi)cos(baz)  -sin(azi)sin(baz)        0            0         0        0 ] [ NN ]
 * TZ   [         0                 0           +cos(azi)          0                  0             -sin(azi)       0         0        0 ] [ NZ ]
 * ZR   [         0                 0              0               0                  0                0        -sin(baz)  -cos(baz)   0 ] [ ZE ]
 * ZT   [         0                 0              0               0                  0                0        -cos(baz)  +sin(baz)   0 ] [ ZN ]
 * ZZ   [         0                 0              0               0                  0                0            0         0        1 ] [ ZZ ]
 *
 * 5. If the E/N/Z component are not exactly E/N/Z,i.e  cmpaz is not 0 for N comp, we should
 *    make azimuth correction. We will suppose the E and N are alway orthogonal. According to
 *    setction 3, the R/T/Z of virt-source will be related to azimuth - src_cmpaz_ncomp, the 
 *    R/T/Z of virt-station will be backazimuth-180-sta_cmpaz_ncomp, so we only update the 
 *    two angle by substract cmpaz_src_ncomp and cmpaz_sta_ncomp. Also we alway take the cmpaz
 *    of T comp 90 away from R comp.
 *
 *    wangwt@20160710
 * **************************************************************************/