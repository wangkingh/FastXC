#ifndef _SMOOTHING_CUH
#define _SMOOTHING_CUH

void launch_smooth_rows(float *d_out, int dpitch, const float *d_in,
                        int spitch, int width, int height, int winsize);

#endif
