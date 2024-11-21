from typing import List, Dict
from itertools import combinations, product
import os
import glob


# generate cross correlation terminal cmd
def gen_xc_cmd_multi(command_dict: Dict, xc_param: Dict, gpu_info: Dict) -> List[str]:
    output_dir = xc_param["output_dir"]
    xc_list_dir = os.path.join(output_dir, "xc_list")

    # check if there is double array situation
    ideal_array1_dir = os.path.join(xc_list_dir, "array1")
    ideal_array2_dir = os.path.join(xc_list_dir, "array2")
    single_flag = not os.path.exists(ideal_array2_dir)

    # get the numbrt of gpu used
    gpu_num = len(gpu_info["gpu_list"])
    cpu_count = int(xc_param["cpu_count"] / gpu_num)

    # this param_set is for both xc_mono and xc_dual
    ncf_dir = os.path.join(output_dir, "ncf")
    param_set = [
        "-O",
        ncf_dir,
        "-C",
        str(xc_param["max_lag"]),
        "-D",
        xc_param["distance_threshold"],
        "-T",
        str(cpu_count),
    ]

    # set an empty cmd_list
    cmd_list = []
    input_sets = []
    if single_flag:
        xc_list = glob.glob(ideal_array1_dir + "/*.speclist")
        xc_list.sort()
        for src in xc_list:
            input_set = ["-A", src, "-B", src]
            input_sets.append(input_set)
    else:
        xc_list_pair = []
        for xc_list_1 in glob.glob(os.path.join(ideal_array1_dir, "*.speclist")):
            fname = os.path.basename(xc_list_1)
            ideal_xc_list_2 = os.path.join(ideal_array2_dir, fname)
            if os.path.exists(ideal_xc_list_2):
                xc_list_pair.append((xc_list_1, ideal_xc_list_2))

        for src, sta in xc_list_pair:
            input_set = ["-A", src, "-B", sta]
            input_sets.append(input_set)

    # iterate over all input_sets of source and station
    for input_set in input_sets:
        cmd = command_dict["xc_multi"] + " "
        cmd += " ".join(input_set) + " "
        cmd += " ".join(param_set)
        cmd_list.append(cmd)

    cmd_list_dir = os.path.join(output_dir, "cmd_list")
    os.makedirs(cmd_list_dir, exist_ok=True)
    with open(os.path.join(cmd_list_dir, "xc_cmds.txt"), "w") as f:
        f.write("\n".join(cmd_list))

    return cmd_list


# generate cross correlation terminal cmd
def gen_xc_cmd_dual(command_dict: Dict, xc_param: Dict, gpu_info: Dict) -> List[str]:
    output_dir = xc_param["output_dir"]
    xc_list_dir = os.path.join(output_dir, "xc_list")

    # check if there is double array situation
    ideal_array1_dir = os.path.join(xc_list_dir, "array1")
    ideal_array2_dir = os.path.join(xc_list_dir, "array2")
    single_flag = not os.path.exists(ideal_array2_dir)

    # get the numbrt of gpu used
    gpu_num = len(gpu_info["gpu_list"])
    cpu_count = int(xc_param["cpu_count"] / gpu_num)

    # this param_set is for both xc_mono and xc_dual
    ncf_dir = os.path.join(output_dir, "stack")
    param_set = [
        "-O",
        ncf_dir,
        "-C",
        str(xc_param["max_lag"]),
        "-S",
        xc_param["save_flag"],
        "-D",
        xc_param["distance_threshold"],
        "-T",
        str(cpu_count),
    ]

    # set an empty cmd_list
    cmd_list = []
    input_sets = []
    if single_flag:
        xc_list = glob.glob(ideal_array1_dir + "/*.speclist")
        xc_list.sort()
        pair_combinations = list(combinations(xc_list, 2))  # 添加互相关
        # self_pairs = [(file, file) for file in xc_list]  # 添加自相关
        # pair_combinations.extend(self_pairs)

        for src, sta in pair_combinations:
            input_set = ["-A", src, "-B", sta]
            input_sets.append(input_set)
        for src in xc_list:
            input_set = ["-A", src, "-B", src]
            input_sets.append(input_set)
    else:
        xc_list_1 = glob.glob(os.path.join(ideal_array1_dir, "*.speclist"))
        xc_list_1.sort()
        xc_list_2 = glob.glob(os.path.join(ideal_array2_dir, "*.speclist"))
        xc_list_2.sort()

        # 使用 itertools.product 生成 xc_list_1 和 xc_list_2 的全部可能配对
        for src, sta in product(xc_list_1, xc_list_2):
            input_set = ["-A", src, "-B", sta]
            input_sets.append(input_set)

    # iterate over all input_sets of source and station
    for input_set in input_sets:
        cmd = command_dict["xc_dual"] + " "
        cmd += " ".join(input_set) + " "
        cmd += " ".join(param_set)
        cmd_list.append(cmd)

    cmd_list_dir = os.path.join(output_dir, "cmd_list")
    os.makedirs(cmd_list_dir, exist_ok=True)
    with open(os.path.join(cmd_list_dir, "xc_cmds.txt"), "w") as f:
        f.write("\n".join(cmd_list))

    return cmd_list


def gen_xc_cmd(command_dict: Dict, xc_param: Dict, gpu_info: Dict) -> List[str]:
    if xc_param["calculate_style"] == "MULTI":
        return gen_xc_cmd_multi(command_dict, xc_param, gpu_info)
    elif xc_param["calculate_style"] == "DUAL":
        return gen_xc_cmd_dual(command_dict, xc_param, gpu_info)
    else:
        raise ValueError(f"xc_type {xc_param['calculate_style']} not supported")
