import numpy as np
from scipy.signal import detrend, butter, filtfilt
from scipy.fft import rfft
from .spectrum_whiten import freq_whiten
from .run_abs_multi import multi_freq_run_abs
from .write_spectrum import write_output_spectrum
from obspy import read
import sys
import pdb


def read_frequency_ranges(filename):
    # 初始化一个空列表来存储频带范围
    frequency_ranges = []

    # 打开并读取文件
    with open(filename, "r") as file:
        for line in file:
            # 检查行是否以 '#' 开始
            if line.startswith("#"):
                # 提取并清理频带范围，然后添加到列表中
                range_part = line.strip().split("#")[1].strip()
                # 将频带范围分割为起始和结束值，转换为浮点数
                range_numbers = list(map(float, range_part.split("/")))
                frequency_ranges.append(range_numbers)

    # 从第二个频带开始返回
    return frequency_ranges[1:]


def process_sliced_streams(sliced_streams, param_dict):
    filter_file = param_dict["filter_file"]
    freq_groups = read_frequency_ranges(filter_file)
    freq_low, freq_high = param_dict["freq_band"]

    for stream in sliced_streams:
        stream[0].detrend(type="constant")  # 去均值
        stream[0].detrend(type="linear")  # 去趋势

    sliced_streams = freq_whiten(sliced_streams, freq_low, freq_high)
    sliced_streams = multi_freq_run_abs(sliced_streams, freq_groups)

    processed_data = []
    for stream in sliced_streams:
        data = stream[0].data
        data = np.pad(data, (0, len(data)))
        spectrum = rfft(data)
        processed_data.append(spectrum)
    return processed_data


def process_a_group(grouped_file, param_dict):
    slice_length = int(param_dict["seglen"])  # unit in seconds
    input_group, output_group = grouped_file
    streams = []
    for file in input_group:
        try:
            stream = read(file)
            if stream is None:
                print(f"Warning: read function returned None for file {file}")
            else:
                streams.append(stream)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            return

    start_time = streams[0][0].stats.starttime
    end_time = min(stream[0].stats.endtime for stream in streams)  # 找到最短的结束时间
    # 从第一个流中获取采样率（delta）
    dt = streams[0][0].stats.delta  # 采样间隔
    nseg = int(slice_length / dt)  # 在一个slice中的数据点数量
    df = 1 / (nseg * dt)  # 频率分辨率

    nseg_2x = 2 * nseg  # 2倍的nseg
    df_2x = 1 / (nseg_2x * dt)  # 2倍的频率分辨率
    nspec_output = int(nseg_2x / 2) + 1  # 输出的频谱点数

    # 计算能够分多少个slice
    nstep = int((end_time + dt - start_time) / slice_length)
    nstep = nstep - len(param_dict["skip_steps"])
    # 创建SEGSPEC结构体信息，取第一个SAC文件的信息作为示例
    first_sac = streams[0][0].stats.sac
    segspec_hd = {
        "stla": first_sac["stla"],
        "stlo": first_sac["stlo"],
        "nstep": nstep,
        "nspec": nspec_output,
        "df": df_2x,
        "dt": dt,
    }

    total_output = []  # 初始化每个输出流
    step_idx = 0
    while start_time + slice_length - dt <= end_time:
        sliced_streams = []
        # 计算当前循环的结束时间
        current_end_time = start_time + slice_length - dt
        if step_idx in param_dict["skip_steps"]:
            start_time += slice_length
            step_idx += 1
            continue
        else:
            for stream in streams:
                sliced_stream = stream.slice(
                    starttime=start_time, endtime=current_end_time
                )
                sliced_streams.append(sliced_stream)
            output_sliced_spectrum = process_sliced_streams(sliced_streams, param_dict)
            total_output.append(output_sliced_spectrum)
            start_time += slice_length
            step_idx += 1

    # 调用write_output_spectrum来写入数据，传递SEGSPEC信息
    write_output_spectrum(total_output, segspec_hd, output_group)
