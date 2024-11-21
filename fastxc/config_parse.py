import os
import configparser
from datetime import datetime


def check_path_exists(path, description):
    if path != "NONE" and not os.path.exists(path):
        print(f"{description} path '{path}' does not exist.")
        raise FileNotFoundError(f"[Error] {description} path '{path}' does not exist.")


def convert_type(key, value):
    try:
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        if key in [
            "npts",
            "win_len",
            "cpu_count",
            "max_lag",
        ]:
            return int(value)
        if key in ["redundant_ratio", "delta"]:
            return float(value)
        if key in ["gpu_list", "gpu_task_num", "gpu_mem_info"]:
            return [int(x) for x in value.split(",")]
        if key in ["component_list_1", "component_list_2"]:
            return value.split(",")
    except ValueError as e:
        print(f"Error converting {key} with value {value}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error when converting {key} with value {value}: {e}")
        raise
    return value


def parse_and_check_ini_file(file_path):
    print(f"[INFO]: Start checking the configuration file {file_path}.")
    config = configparser.ConfigParser()
    config.read(file_path)

    if not config.sections():
        print("No sections found in the configuration file.")
        raise ValueError("No sections found in the configuration file.")

    # 使用字典推导来统一配置的获取，并应用类型转换
    try:
        SeisArrayInfo = {
            key: convert_type(key, config.get("SeisArrayInfo", key, fallback="NONE"))
            for key in config["SeisArrayInfo"]
        }
        Parameters = {
            key: convert_type(key, config.get("Parameters", key, fallback="NONE"))
            for key in config["Parameters"]
        }
        Command = {
            key: config.get("Command", key, fallback="NONE")
            for key in config["Command"]
        }
        gpu_info = {
            key: convert_type(key, config.get("gpu_info", key, fallback="NONE"))
            for key in config["gpu_info"]
        }
    except Exception as e:
        print(f"Failed to load configuration properly: {e}")
        raise ValueError(f"Failed to load configuration properly: {e}")

    # 检查路径存在性
    paths_to_check = {
        "sac_dir_1": "SAC File directory 1",
        "sac_dir_2": "SAC File directory 2",
        "sta_list_1": "Station list 1",
        "sta_list_2": "Station list 2",
    }

    paths_to_check.update(
        {key: f"Command tool for {key}" for key, _ in Command.items()}
    )

    for key, description in paths_to_check.items():
        check_path_exists(
            (
                config.get("SeisArrayInfo", key)
                if key in config["SeisArrayInfo"]
                else Command[key]
            ),
            description,
        )

    # 检查时间戳的有效性
    for time_key in ["start", "end"]:
        time_value = config.get("SeisArrayInfo", time_key, fallback="NONE")
        try:
            datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError(
                f"[Error] Invalid date format for {time_key}: {time_value}"
            )

    # 检查两个台阵的通道数量是否一致
    if config.get("SeisArrayInfo", "sac_dir_2") != "NONE" and len(
        SeisArrayInfo["component_list_1"]
    ) != len(SeisArrayInfo["component_list_2"]):
        raise ValueError(
            "[Error] component_list_1 and component_list_2 should be the same length."
        )

    # 检查频率域白化,归一化,旋转和计算风格参数的有效性
    if Parameters["whiten"] not in ["OFF", "BEFORE", "AFTER", "BOTH"]:
        raise ValueError(
            f"[Error] Invalid whiten value {Parameters['whiten']} in section Parameters:whiten."
        )
    if Parameters["normalize"] not in ["OFF", "RUN-ABS", "ONE-BIT", "RUN-ABS-MF"]:
        raise ValueError(
            f"[Error] Invalid normalize value {Parameters['normalize']} in section Parameters:normalize."
        )
    if Parameters["rotate_dir"] not in ["NONE", "LINEAR", "PWS", "TFPWS"]:
        raise ValueError(
            f"[Error] Invalid normalize value {Parameters['rotate_dir']} in section Parameters:rotate_dir."
        )
    if Parameters["calculate_style"] not in ["MULTI", "DUAL"]:
        raise ValueError(
            f"[Error] Invalid normalize value {Parameters['calculate_style']} in section Parameters:calculate_style."
        )
    if os.path.isfile(Parameters["log_file_path"]):  # 检查是否为文件
        os.remove(Parameters["log_file_path"])  # 如果是文件，则删除
        print(f"File '{Parameters['log_file_path']}' has been removed.")

    # 创建一个新的空文件
    with open(Parameters["log_file_path"], "w") as file:
        pass  # 使用 'w' 模式创建文件，如果文件存在则清空内容

    print(f"File '{Parameters['log_file_path']}' has been recreated.")

    print("[INFO]: Configuration checked successfully.")

    return SeisArrayInfo, Parameters, Command, gpu_info
