#ifndef _FFT_LENGTH_H_
#define _FFT_LENGTH_H_

int segment_length_from_seconds(float seconds, int max_samples, float delta);

int next_smooth_2357_length(int min_length);

int next_even_smooth_2357_length(int min_length);

#endif
