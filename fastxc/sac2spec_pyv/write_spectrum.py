import os
import numpy as np
import struct


def write_output_spectrum(total_output, segspec_hd, output_group):
    # 先初始化一个列表，用于存储每个输出文件对应的所有数据拼接结果
    concatenated_data = [np.array([], dtype=np.complex64) for _ in output_group]

    # 遍历每组时间片切割后的数据
    for sliced_spectrum_groups in total_output:
        # 遍历每个输出文件对应的数据组
        for idx, data in enumerate(sliced_spectrum_groups):
            concatenated_data[idx] = np.concatenate(
                [concatenated_data[idx], np.array(data, dtype=np.complex64)]
            )

    # 开始写入到各个文件
    for idx, spec_data in enumerate(concatenated_data):
        specpath = output_group[idx]
        # print(f"Writing to {specpath}")
        parent_dir = os.path.dirname(specpath)
        os.makedirs(parent_dir, exist_ok=True)
        with open(specpath, "wb") as fid:
            # 创建并写入SEGSPEC头信息
            header_format = "ffiiff"  # 对应于4个floats和2个ints的结构
            header_data = struct.pack(
                header_format,
                segspec_hd["stla"],
                segspec_hd["stlo"],
                segspec_hd["nstep"],
                segspec_hd["nspec"],
                segspec_hd["df"],
                segspec_hd["dt"],
            )
            fid.write(header_data)
            fid.write(spec_data.tobytes())  # 写入二进制数据
