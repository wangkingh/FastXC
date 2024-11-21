# Purpose: Generating the xc_list dir for spec cross-correlation
from typing import Dict
from datetime import datetime
import os
import re
import glob
from multiprocessing import Pool
from tqdm import tqdm


def process_channel_spec_dir(args):
    time_spec_dir, xc_list_dir = args
    try:
        info = time_spec_dir.split(os.sep)
        knetwk = info[-2]
        kstnm_cmp = info[-1]
        kstnm, kcmpnm = kstnm_cmp.split(".")
        target_xc_list_name = f"{kstnm}.{kcmpnm}.speclist"
        target_xc_list_path = os.path.join(xc_list_dir, knetwk, target_xc_list_name)

        file_times = []
        all_spec = glob.glob(time_spec_dir + "/*")
        for spec_file in all_spec:
            fname = spec_file.split(os.sep)[-1]
            kstnm, year, jdate, hourmin, _, _ = fname.split(".")
            date_time_str = f"{year}.{jdate}.{hourmin[:2]}.{hourmin[2:]}"
            date_time = datetime.strptime(date_time_str, "%Y.%j.%H.%M")
            file_times.append((spec_file, date_time))

        file_times.sort(key=lambda x: x[1])
        os.makedirs(os.path.join(xc_list_dir, knetwk), exist_ok=True)
        with open(target_xc_list_path, "w") as f:
            for file, _ in file_times:
                f.write(file + "\n")
    except Exception as e:
        print(f"Error processing {time_spec_dir}: {str(e)}")


def natural_sort_key(s):
    # 使用正则表达式将字符串中的数字文本转换为整数
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def process_time_spec_dir(args):
    time_spec_dir, xc_list_dir = args
    try:
        info = time_spec_dir.split(os.sep)
        knetwk = info[-2]
        time_info = info[-1]
        target_xc_list_name = f"{time_info}.speclist"
        target_xc_list_path = os.path.join(xc_list_dir, knetwk, target_xc_list_name)

        all_spec = glob.glob(time_spec_dir + "/*")
        all_spec_sorted = sorted(
            all_spec, key=natural_sort_key
        )  # 使用自然排序键进行排序

        os.makedirs(os.path.join(xc_list_dir, knetwk), exist_ok=True)
        with open(target_xc_list_path, "w") as f:
            for file in all_spec_sorted:
                f.write(file + "\n")
    except Exception as e:
        print(f"Error processing {time_spec_dir}: {str(e)}")


def gen_xc_list_dir(xc_param: Dict):
    output_dir = xc_param["output_dir"]
    num_thread = xc_param["cpu_count"]
    calulate_style = xc_param["calculate_style"]
    segspec_dir = os.path.join(output_dir, "segspec")
    xc_list_dir = os.path.join(output_dir, "xc_list")

    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    description = (
        f"[INFO {current_time}]: Generating SEGSPEC Lists For Cross-Correlation"
    )

    if calulate_style == "MULTI":
        time_spec_dirs = glob.glob(segspec_dir + "/*/*")
        with Pool(processes=num_thread) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        process_time_spec_dir,
                        [(dir, xc_list_dir) for dir in time_spec_dirs],
                    ),
                    total=len(time_spec_dirs),
                    desc=description,
                )
            )
    elif calulate_style == "DUAL":
        channel_spec_dirs = glob.glob(segspec_dir + "/*/*")
        with Pool(processes=num_thread) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        process_channel_spec_dir,
                        [(dir, xc_list_dir) for dir in channel_spec_dirs],
                    ),
                    total=len(channel_spec_dirs),
                    desc=description,
                )
            )

    print(f"Finish generating xc_list dir at {xc_list_dir}\n")
