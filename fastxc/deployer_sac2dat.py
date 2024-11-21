from obspy import read
import numpy as np
import math
import os
import multiprocessing
from tqdm import tqdm
from datetime import datetime


def gen_sac_dat_pair(xc_param):
    do_linear = bool(int(xc_param["todat_flag"][0]))
    do_pws = bool(int(xc_param["todat_flag"][1]))
    do_tfpws = bool(int(xc_param["todat_flag"][2]))
    do_rtz = bool(int(xc_param["todat_flag"][3]))
    output_dir = xc_param["output_dir"]

    # 正确定义处理类型与目录的映射
    process_dirs = {
        "linear": ("linear", "linear_dat"),
        "pws": ("pws", "pws_dat"),
        "tfpws": ("tfpws", "tfpws_dat"),
        "rtz": ("rtz", "rtz_dat"),
    }

    all_pairs = []
    for process_type, (sac_subdir, dat_subdir) in process_dirs.items():
        if (
            (process_type == "linear" and do_linear)
            or (process_type == "pws" and do_pws)
            or (process_type == "tfpws" and do_tfpws)
            or (process_type == "rtz" and do_rtz)
        ):
            sac_path = os.path.join(output_dir, "stack", sac_subdir)
            dat_path = os.path.join(output_dir, "stack", dat_subdir)
            for root, dirs, files in os.walk(sac_path):
                for file in files:
                    if file.endswith(".sac"):
                        full_sac_path = os.path.join(root, file)
                        # 构造对应的 .dat 文件路径
                        dat_file = file.replace(".sac", f".{process_type}.dat")
                        full_dat_path = os.path.join(dat_path, dat_file)
                        os.makedirs(os.path.dirname(full_dat_path),exist_ok=True)
                        all_pairs.append((full_sac_path, full_dat_path))

    return all_pairs


def sac2dat(sac_dat_pair):
    # Unpack the pair
    sac, dat_path = sac_dat_pair

    # Read in sac file
    st = read(sac)
    tr = st[0]  # Only one trace in sac file

    # Extract header information
    delta = tr.stats.delta
    try:
        npts = tr.stats.sac.npts
        stla, stlo, stel = tr.stats.sac.stla, tr.stats.sac.stlo, 0
        evla, evlo, evel = tr.stats.sac.evla, tr.stats.sac.evlo, 0
    except AttributeError as e:
        print(f"Missing SAC header information in {sac}: {e}")
        return

    # Data Processing
    data = tr.data
    half_length = math.ceil(npts / 2)
    data_neg = data[:half_length][::-1]  # Negative half
    data_pos = data[(half_length - 1) :]  # Positive half
    times = np.arange(0, half_length * delta, delta)

    # Write output file
    with open(dat_path, "w") as f:
        # Write station and event location information
        f.write(f"{stlo:.7e} {stla:.7e} {stel:.7e}\n")
        f.write(f"{evlo:.7e} {evla:.7e} {evel:.7e}\n")

        # Write time, negative half, and positive half data
        for i, time in enumerate(times):
            neg = data_neg[i] if i < len(data_neg) else 0  # Pad with zeros if necessary
            pos = data_pos[i] if i < len(data_pos) else 0  # Pad with zeros if necessary
            f.write(f"{time:.7e} {neg:.7e} {pos:.7e}\n")


def sac2dat_deployer(xc_param):
    sac_dat_pairs = gen_sac_dat_pair(xc_param)
    num_processes = xc_param.get("cpu_count", os.cpu_count())
    pbar = tqdm(total=len(sac_dat_pairs))
    pool = multiprocessing.Pool(processes=num_processes)
    for sac_dat_pair in sac_dat_pairs:
        pool.apply_async(
            sac2dat, args=(sac_dat_pair,), callback=lambda _: pbar.update()
        )
    pool.close()
    pool.join()
    output_dir = xc_param["output_dir"]
    process_dirs = ["linear_dat", "pws_dat", "tfpws_dat", "rtz_dat"]
    main_dat_list_path = os.path.join(output_dir, "dat_list.txt")
    with open(main_dat_list_path, "w") as main_dat_list:
        for process_dir in process_dirs:
            destination_dir = os.path.join(output_dir, "stack", process_dir)
            if not os.path.exists(destination_dir):
                print(f"Directory {destination_dir} does not exist...")
                continue
            dat_list = os.path.join(destination_dir, "dat_list.txt")
            with open(dat_list, "w") as f:
                for root, _, files in os.walk(destination_dir):
                    for file in files:
                        if file.endswith(".dat"):
                            fname = os.path.basename(file)
                            stapair = fname.split(".")[0]
                            sta1, sta2 = stapair.split("-")
                            if sta1 != sta2:
                                f.write(file + "\n")
                                relative_path = os.path.relpath(
                                    os.path.join(root, file), output_dir
                                )
                                main_dat_list.write(relative_path + "\n")
