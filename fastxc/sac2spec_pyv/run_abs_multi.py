import numpy as np
import scipy.signal as signal
import copy
import pdb


def multi_freq_run_abs(streams, freq_groups):
    sample_rate = 1 / streams[0][0].stats.delta
    processed_streams = copy.deepcopy(streams)
    accumulated_filtered_datas = [np.zeros(len(stream[0].data)) for stream in streams]
    for freq in freq_groups:
        low_f, high_f = freq
        b, a = signal.butter(2, [low_f, high_f], btype="band", fs=sample_rate)

        filtered_datas = []

        # 重置每个频段的平滑幅度数组和窗口大小
        smooth_amp = np.zeros(len(streams[0][0].data))
        winsize = round((1 / low_f) * 2 * sample_rate)
        if winsize % 2 == 0:
            winsize += 1
        window = np.ones(winsize) / winsize

        for stream in streams:
            data = stream[0].data
            filtered_data = signal.filtfilt(b, a, data)
            filtered_datas.append(filtered_data)

            padded_data = np.pad(
                np.abs(filtered_data), pad_width=(winsize, winsize), mode="edge"
            )

            filtered_smooth_amp = signal.lfilter(window, [1], padded_data)
            correct_length_smooth_amp = filtered_smooth_amp[winsize:-winsize]
            smooth_amp += correct_length_smooth_amp

        valid_indices = smooth_amp > 1e-10
        for filtered_data in filtered_datas:
            filtered_data[valid_indices] /= smooth_amp[valid_indices]

        # 累加归一化后的数据
        for i in range(len(accumulated_filtered_datas)):
            accumulated_filtered_datas[i] += filtered_datas[i]

    # 将累加归一化后的数据存回processed_streams中对应的stream
    for i, stream in enumerate(processed_streams):
        stream[0].data = accumulated_filtered_datas[i]

    return processed_streams
