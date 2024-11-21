import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, irfft
import copy
from scipy.signal.windows import hann
import matplotlib.pyplot as plt  # 导入绘图库
import pdb


def freq_whiten(streams, freq_low, freq_high):
    sampling_rate = 1 / streams[0][0].stats.delta
    processed_streams = copy.deepcopy(streams)
    fft_length = len(streams[0][0].data)
    delta_f = sampling_rate / fft_length

    freqs = np.fft.rfftfreq(len(streams[0][0].data), d=streams[0][0].stats.delta)

    # 计算特定白化频率的索引范围
    freq_low = max(freq_low, 0)
    freq_high = max(freq_high, 0)
    freq_low_limit = 0.75 * freq_low
    freq_high_limit = 1.5 * freq_high

    freq_low_limit = max(freq_low_limit, 0)
    freq_high_limit = min(freq_high_limit, sampling_rate / 2 - delta_f)

    # 拐角频率对应的指标
    idx_low = np.searchsorted(freqs, freq_low, side="left")
    idx_high = np.searchsorted(freqs, freq_high, side="right")

    # 截止频率对应的指标
    idx_low_limit = np.searchsorted(freqs, freq_low_limit, side="left")
    idx_high_limit = np.searchsorted(freqs, freq_high_limit, side="right")

    # 白化的指标范围
    nn = slice(idx_low_limit, idx_high_limit + 1)

    winsize = max(int(0.02 / delta_f), 11)
    if winsize % 2 == 0:
        winsize += 1  # 确保窗口大小为奇数

    window = np.ones(winsize) / winsize
    accumulated_smoothed_amp = np.zeros(
        idx_high_limit + 1 - idx_low_limit
    )  # Adjusted size
    b, a = butter(
        4, [freq_low, freq_high], btype="bandpass", fs=1 / streams[0][0].stats.delta
    )

    spectrums = []
    for stream in streams:
        data = stream[0].data
        data = filtfilt(b, a, data)
        spectrum = rfft(data)
        spectrums.append(spectrum)

        spectrum_amp = np.abs(spectrum[nn])
        padded_spectrum_amp = np.pad(
            spectrum_amp, pad_width=(winsize, winsize), mode="edge"
        )
        data_f_amp_smooth = signal.lfilter(window, [1], padded_spectrum_amp)
        data_f_amp_smooth = data_f_amp_smooth[winsize:-winsize]
        accumulated_smoothed_amp += data_f_amp_smooth

    # 使用累加的平滑幅度进行谱白化处理
    for spectrum in spectrums:
        valid_indices_in_weight = np.where(accumulated_smoothed_amp > 1e-7)[0]
        valid_indices = np.where(accumulated_smoothed_amp > 1e-7)[0] + idx_low_limit
        spectrum[valid_indices] /= accumulated_smoothed_amp[valid_indices_in_weight]

        # bandpass filtering
        spectrum[: idx_low_limit + 1] = 0
        spectrum[idx_high_limit:] = 0
        low_transition_npts = idx_low - idx_low_limit
        if low_transition_npts >= 4:
            taperwin_low = hann(2 * low_transition_npts)
            spectrum[idx_low_limit:idx_low] *= taperwin_low[:low_transition_npts]

        high_transition_npts = idx_high_limit - idx_high + 1
        if high_transition_npts >= 4:
            taperwin_high = hann(2 * high_transition_npts)  # 创建一个足够长的汉宁窗
            spectrum[idx_high : idx_high_limit + 1] *= taperwin_high[
                high_transition_npts:
            ]

    for i, stream in enumerate(processed_streams):
        # 转换回时间域并存储处理后的数据
        stream[0].data = irfft(spectrums[i])

    return processed_streams
